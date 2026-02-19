"""
Maximum-performance ball tracking for padel analytics.

Detection pipeline (per frame, in priority order):
  1. YOLO at high resolution (1280px) on raw frame — best for small ball detection
  2. YOLO at high resolution on CLAHE-enhanced frame (fallback)
  3. Motion detection via triple-frame differencing — catches ball even when YOLO fails
  4. ROI search with CLAHE + upscaling around Kalman-predicted position
  5. Kalman filter prediction to bridge gaps (up to ~2s)

All detections are validated against the Kalman trajectory to reject false positives.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.config import (
    YOLO_BALL_CLASS,
    YOLO_BALL_CONFIDENCE,
)

# Extra COCO classes that sometimes detect ball-like objects
# 29 = frisbee, 33 = kite — occasionally triggered by padel balls
BALL_CLASSES = [YOLO_BALL_CLASS, 29, 33]

# High-resolution inference for better small-object detection
YOLO_BALL_IMGSZ = 1280


@dataclass
class BallDetection:
    """Raw ball detection from YOLO or motion detection."""
    bbox: np.ndarray       # [x1, y1, x2, y2]
    confidence: float
    center: np.ndarray     # [x, y] pixel center


class BallTracker:
    """
    Maximum-performance ball tracker.

    Uses a multi-stage detection cascade with Kalman filter trajectory
    estimation to achieve the highest possible tracking rate.
    """

    def __init__(
        self,
        model,
        confidence: float = YOLO_BALL_CONFIDENCE,
        max_gap: int = 8,
        max_prediction_frames: int = 5,
        search_roi_scale: float = 3.0,
        validation_distance: float = 150.0,
    ):
        self.model = model
        self.confidence = confidence
        self.max_gap = max_gap
        self.max_prediction_frames = max_prediction_frames
        self.search_roi_scale = search_roi_scale
        self.validation_distance = validation_distance

        # Kalman filter state
        self._kf = self._init_kalman()
        self._kf_initialized = False
        self._frames_since_detection = 0
        self._consecutive_detections = 0

        # Detection history + stats
        self._history: List[Optional[np.ndarray]] = []
        self._source_counts = {"detected": 0, "motion": 0, "predicted": 0}
        self._frame_idx = 0

        # Last known bbox
        self._last_bbox: Optional[np.ndarray] = None
        self._last_confidence: float = 0.0

        # Motion detection state (triple-frame buffer)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_prev_gray: Optional[np.ndarray] = None

        # CLAHE for preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Sharpen kernel
        self._sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ], dtype=np.float32)

    def _init_kalman(self) -> cv2.KalmanFilter:
        """Initialize Kalman filter with constant velocity model."""
        kf = cv2.KalmanFilter(4, 2)

        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 50.0
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        return kf

    def reset(self):
        """Reset the tracker state."""
        self._kf = self._init_kalman()
        self._kf_initialized = False
        self._frames_since_detection = 0
        self._consecutive_detections = 0
        self._history.clear()
        self._source_counts = {"detected": 0, "motion": 0, "predicted": 0}
        self._frame_idx = 0
        self._last_bbox = None
        self._prev_gray = None
        self._prev_prev_gray = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame with CLAHE + sharpening for better ball visibility."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_chan)
        enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        sharpened = cv2.filter2D(enhanced, -1, self._sharpen_kernel)
        return sharpened

    def _detect_motion_blobs(self, frame: np.ndarray) -> List[BallDetection]:
        """
        Detect small fast-moving objects via triple-frame differencing.
        More sensitive than before — lower threshold, wider area range.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        detections = []

        if self._prev_gray is not None and self._prev_prev_gray is not None:
            diff1 = cv2.absdiff(gray, self._prev_gray)
            diff2 = cv2.absdiff(self._prev_gray, self._prev_prev_gray)
            combined = cv2.bitwise_and(diff1, diff2)

            # Lower threshold = more sensitive
            _, thresh = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)

            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Wider area range to catch ball at different distances
                if area < 15 or area > 5000:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.2:  # more permissive
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1e-6)
                if aspect > 4.0:  # more permissive
                    continue

                cx = x + w / 2.0
                cy = y + h / 2.0
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)

                detections.append(BallDetection(
                    bbox=bbox,
                    confidence=0.15,
                    center=np.array([cx, cy]),
                ))

        self._prev_prev_gray = self._prev_gray
        self._prev_gray = gray

        return detections

    def update(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, str]:
        """
        Process a frame and return ball position.

        Returns:
            (center, bbox, confidence, source) — source is "detected", "motion", or "predicted"
        """
        # Stage 1: YOLO at high resolution on raw frame
        detection = self._yolo_detect_hires(frame)
        det_source = "detected"

        # Stage 1b: If missed, retry on CLAHE-enhanced frame at high res
        if detection is None:
            enhanced = self._preprocess(frame)
            detection = self._yolo_detect_hires(enhanced)

        # Stage 2: Motion detection via frame differencing
        motion_blobs = self._detect_motion_blobs(frame)
        if detection is None and motion_blobs:
            detection = self._pick_best_motion(motion_blobs)
            if detection is not None:
                det_source = "motion"

        # Stage 3: ROI search with CLAHE + upscaling around Kalman prediction
        if detection is None and self._kf_initialized and self._frames_since_detection < self.max_prediction_frames:
            prediction = self._kf.statePost[:2].flatten()
            roi_det = self._search_roi_upscaled(frame, prediction)
            if roi_det is not None:
                detection = roi_det
                det_source = "detected"

        # Kalman predict step
        predicted_pos = None
        if self._kf_initialized:
            prediction = self._kf.predict()
            predicted_pos = prediction[:2].flatten()

        result_center = None
        result_bbox = None
        result_conf = 0.0
        result_source = ""

        # Validate detection against Kalman prediction
        if detection is not None:
            if predicted_pos is not None and self._consecutive_detections > 2:
                dist = np.linalg.norm(detection.center - predicted_pos)
                if dist > self.validation_distance and self._frames_since_detection < 3:
                    detection = None

        if detection is not None:
            # Good detection — update Kalman
            measurement = detection.center.reshape(2, 1).astype(np.float32)

            if not self._kf_initialized:
                self._kf.statePost = np.array([
                    [detection.center[0]],
                    [detection.center[1]],
                    [0],
                    [0],
                ], dtype=np.float32)
                self._kf_initialized = True
            else:
                self._kf.correct(measurement)

            result_center = detection.center
            result_bbox = detection.bbox
            result_conf = detection.confidence
            result_source = det_source
            self._frames_since_detection = 0
            self._consecutive_detections += 1
            self._last_bbox = detection.bbox
            self._last_confidence = detection.confidence

        elif self._kf_initialized and self._frames_since_detection < self.max_prediction_frames:
            # Use Kalman prediction
            result_center = predicted_pos
            if self._last_bbox is not None:
                w = self._last_bbox[2] - self._last_bbox[0]
                h = self._last_bbox[3] - self._last_bbox[1]
            else:
                w, h = 20, 20
            result_bbox = np.array([
                predicted_pos[0] - w / 2, predicted_pos[1] - h / 2,
                predicted_pos[0] + w / 2, predicted_pos[1] + h / 2,
            ])
            result_conf = max(0.08, self._last_confidence * 0.92 ** self._frames_since_detection)
            result_source = "predicted"
            self._consecutive_detections = 0

        else:
            self._consecutive_detections = 0
            if self._frames_since_detection > self.max_prediction_frames * 2:
                self._kf_initialized = False

        self._frames_since_detection += 1

        # Track stats
        self._history.append(result_center.copy() if result_center is not None else None)
        if result_source:
            self._source_counts[result_source] = self._source_counts.get(result_source, 0) + 1
        self._frame_idx += 1

        if result_center is not None:
            return result_center, result_bbox, result_conf, result_source
        return None, None, 0.0, ""

    def _yolo_detect_hires(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Run YOLO at 1280px resolution with multiple ball-like classes."""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=BALL_CLASSES,
            imgsz=YOLO_BALL_IMGSZ,
            verbose=False,
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()

        detections = []
        for i in range(len(boxes)):
            bbox = boxes[i]
            # Filter out very large detections (not a ball)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w * h > frame.shape[0] * frame.shape[1] * 0.02:
                continue

            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # Boost confidence for the primary ball class
            conf = float(confs[i])
            if int(cls[i]) != YOLO_BALL_CLASS:
                conf *= 0.8  # slight penalty for secondary classes

            detections.append(BallDetection(
                bbox=bbox,
                confidence=conf,
                center=np.array([cx, cy]),
            ))

        if not detections:
            return None
        return self._pick_best(detections)

    def _search_roi_upscaled(self, frame: np.ndarray, predicted_center: np.ndarray) -> Optional[BallDetection]:
        """
        Search for ball in ROI around predicted position.
        Upscales the crop 2x and applies CLAHE for maximum small-object sensitivity.
        """
        fh, fw = frame.shape[:2]
        roi_size = 200

        x1 = max(0, int(predicted_center[0] - roi_size))
        y1 = max(0, int(predicted_center[1] - roi_size))
        x2 = min(fw, int(predicted_center[0] + roi_size))
        y2 = min(fh, int(predicted_center[1] + roi_size))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        roi = frame[y1:y2, x1:x2]

        # CLAHE enhance
        roi = self._preprocess(roi)

        # Upscale 2x so YOLO sees the ball larger
        roi_up = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

        results = self.model.predict(
            roi_up,
            conf=0.05,
            classes=BALL_CLASSES,
            verbose=False,
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        best_idx = int(np.argmax(confs))
        bbox = boxes[best_idx]

        # Convert back from upscaled ROI coords to full-frame coords
        bbox_full = bbox.copy() / 2.0  # downscale from 2x
        bbox_full[0] += x1
        bbox_full[1] += y1
        bbox_full[2] += x1
        bbox_full[3] += y1

        cx = (bbox_full[0] + bbox_full[2]) / 2.0
        cy = (bbox_full[1] + bbox_full[3]) / 2.0

        return BallDetection(
            bbox=bbox_full,
            confidence=float(confs[best_idx]),
            center=np.array([cx, cy]),
        )

    def _pick_best(self, detections: List[BallDetection]) -> BallDetection:
        """Pick the best detection — closest to prediction if available, else highest confidence."""
        if not self._kf_initialized or len(detections) == 1:
            return max(detections, key=lambda d: d.confidence)

        predicted = self._kf.statePost[:2].flatten()

        def score(d):
            dist = np.linalg.norm(d.center - predicted)
            dist_score = max(0, 1.0 - dist / self.validation_distance)
            return d.confidence * 0.4 + dist_score * 0.6

        return max(detections, key=score)

    def _pick_best_motion(self, blobs: List[BallDetection]) -> Optional[BallDetection]:
        """Pick the best motion blob — must be near Kalman prediction if available."""
        if not blobs:
            return None

        if not self._kf_initialized:
            return max(blobs, key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]))

        predicted = self._kf.statePost[:2].flatten()

        valid = []
        for b in blobs:
            dist = np.linalg.norm(b.center - predicted)
            if dist < self.validation_distance:
                valid.append((b, dist))

        if not valid:
            return None

        valid.sort(key=lambda x: x[1])
        return valid[0][0]

    def get_stats(self) -> dict:
        """Return tracking statistics with source breakdown."""
        total = len(self._history)
        detected = sum(1 for h in self._history if h is not None)
        return {
            "total_frames": total,
            "frames_with_ball": detected,
            "detection_rate": detected / total if total > 0 else 0,
            "source_breakdown": dict(self._source_counts),
        }
