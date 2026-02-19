"""
Automatic and manual court detection + homography computation.
Maps video pixel coordinates to 2D court coordinates (meters).

Default mode is fully automatic (no user interaction).
Pass --manual-calibration to fall back to interactive clicking.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.config import (
    COURT_REFERENCE_POINTS_2D,
    DEFAULT_CALIBRATION_POINTS,
    CALIBRATION_POINT_DESCRIPTIONS,
    COURT_LENGTH,
    COURT_WIDTH,
    SERVICE_LINE_DIST,
    NET_Y,
    CENTER_SERVICE_X,
)

# Court corners in meters: near_left, near_right, far_left, far_right
COURT_CORNERS_M = np.array([
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [0.0, COURT_LENGTH],
    [COURT_WIDTH, COURT_LENGTH],
], dtype=np.float32)

# HSV ranges for common padel court surface colors
COURT_COLOR_RANGES = [
    # (name, lower_hsv, upper_hsv)
    ("blue",       (90, 30, 40),  (130, 255, 255)),
    ("light_blue", (85, 20, 80),  (115, 255, 255)),
    ("turquoise",  (75, 25, 50),  (100, 255, 255)),
    ("green",      (35, 30, 40),  (85,  255, 255)),
    ("orange",     (5,  50, 80),  (25,  255, 255)),
]


# ──────────────────────────────────────────────────────
# AUTOMATIC COURT DETECTOR
# ──────────────────────────────────────────────────────
class AutoCourtDetector:
    """
    Fully automatic court detection. No user interaction required.

    Strategies tried in order:
      1. Color segmentation to find court surface → Hough lines inside → corners
      2. Global Hough line detection → cluster → boundary intersections
      3. Edge contour detection → largest quadrilateral
    """

    def __init__(self):
        self.H: Optional[np.ndarray] = None
        self._corners_pixel: Optional[np.ndarray] = None

    # ── public API ───────────────────────────────────

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto-detect the padel court and return the 3×3 homography matrix.
        Raises RuntimeError if all strategies fail.
        """
        h, w = frame.shape[:2]
        print("  [Auto] Detecting court lines...")

        for strategy_name, strategy_fn in [
            ("color + lines", self._strategy_color_and_lines),
            ("hough lines",   self._strategy_hough_only),
            ("contour",       self._strategy_contour),
        ]:
            try:
                corners = strategy_fn(frame)
                if corners is not None and len(corners) == 4:
                    ordered = self._order_corners(corners)
                    H, _ = cv2.findHomography(ordered, COURT_CORNERS_M, cv2.RANSAC, 5.0)
                    if H is not None and self._validate_homography(H, (h, w)):
                        self.H = H
                        self._corners_pixel = ordered
                        print(f"  [Auto] Success via '{strategy_name}'")
                        return self.H
            except Exception as e:
                print(f"  [Auto] Strategy '{strategy_name}' failed: {e}")

        raise RuntimeError(
            "Automatic court detection failed on all strategies. "
            "Use --manual-calibration for interactive point selection."
        )

    def transform_point(self, px: float, py: float) -> Tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Detection not done yet.")
        pt = np.array([px, py, 1.0], dtype=np.float64)
        r = self.H @ pt
        r /= r[2]
        return (float(r[0]), float(r[1]))

    def transform_points(self, pixel_points: np.ndarray) -> np.ndarray:
        if self.H is None:
            raise RuntimeError("Detection not done yet.")
        pts = pixel_points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def save_calibration(self, path: str) -> None:
        if self.H is None:
            raise RuntimeError("No calibration to save.")
        np.savez(path, H=self.H, corners_pixel=self._corners_pixel)
        print(f"  Calibration saved to: {path}")

    def load_calibration(self, path: str) -> np.ndarray:
        data = np.load(path)
        self.H = data["H"]
        if "corners_pixel" in data:
            self._corners_pixel = data["corners_pixel"]
        print(f"  Calibration loaded from: {path}")
        return self.H

    # ── Strategy 1: color segmentation + Hough refinement ──

    def _strategy_color_and_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court surface by color, then find white lines within it."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_mask = None
        best_area = 0

        for name, lower, upper in COURT_COLOR_RANGES:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            area = cv2.countNonZero(mask)
            # Court surface should be 5-70% of frame
            ratio = area / (h * w)
            if 0.05 < ratio < 0.70 and area > best_area:
                best_area = area
                best_mask = mask

        if best_mask is None:
            return None

        # Find boundary of the court surface
        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < h * w * 0.05:
            return None

        # Try to approximate directly to quadrilateral
        peri = cv2.arcLength(largest, True)
        for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.08]:
            approx = cv2.approxPolyDP(largest, eps_mult * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        # Fallback: use the white lines inside the court region for refinement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=best_mask)

        # Detect white lines on the court surface
        _, white = cv2.threshold(masked_gray, 170, 255, cv2.THRESH_BINARY)
        kernel_sm = np.ones((3, 3), np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel_sm, iterations=2)
        edges = cv2.Canny(white, 50, 150)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(80, int(min(h, w) * 0.08)))
        if lines is not None and len(lines) >= 4:
            corners = self._corners_from_hough(lines, h, w)
            if corners is not None:
                return corners

        # Last resort: min-area rect of the largest contour
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        return box

    # ── Strategy 2: pure Hough line detection ──

    def _strategy_hough_only(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court from Hough lines on the full frame."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        thresh = max(100, int(min(h, w) * 0.10))
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh)
        if lines is None or len(lines) < 4:
            return None

        return self._corners_from_hough(lines, h, w)

    # ── Strategy 3: largest quadrilateral contour ──

    def _strategy_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest approximately rectangular contour."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_area = 0
        min_area = h * w * 0.05

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            for eps_mult in [0.02, 0.04, 0.06]:
                approx = cv2.approxPolyDP(contour, eps_mult * peri, True)
                if len(approx) == 4 and area > best_area:
                    best_quad = approx.reshape(4, 2).astype(np.float32)
                    best_area = area

        return best_quad

    # ── Hough line helpers ──

    def _corners_from_hough(self, lines: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        """
        Given raw Hough lines, cluster into two perpendicular groups,
        pick the outermost pair from each group, and return their 4 intersections.
        """
        rho_theta = lines[:, 0]  # (N, 2)
        thetas = rho_theta[:, 1] % np.pi

        # Merge nearby duplicate lines
        merged = self._merge_lines(rho_theta)
        if len(merged) < 4:
            return None

        thetas_m = merged[:, 1] % np.pi

        # Cluster into two groups by angle (horizontal-ish vs other)
        median_theta = np.median(thetas_m)
        group_a = merged[np.abs(thetas_m - median_theta) < np.pi / 6]
        group_b = merged[np.abs(thetas_m - median_theta) >= np.pi / 6]

        if len(group_a) < 2 or len(group_b) < 2:
            # Try splitting at pi/4
            horiz_mask = (thetas_m > np.pi / 4) & (thetas_m < 3 * np.pi / 4)
            group_a = merged[horiz_mask]    # horizontal-ish
            group_b = merged[~horiz_mask]   # vertical-ish

        if len(group_a) < 2 or len(group_b) < 2:
            return None

        # Pick the two most extreme lines from each group by rho
        line_a1 = group_a[np.argmin(group_a[:, 0])]
        line_a2 = group_a[np.argmax(group_a[:, 0])]
        line_b1 = group_b[np.argmin(group_b[:, 0])]
        line_b2 = group_b[np.argmax(group_b[:, 0])]

        # All 4 pairwise intersections
        corners = []
        for la in [line_a1, line_a2]:
            for lb in [line_b1, line_b2]:
                pt = self._line_intersection(la[0], la[1], lb[0], lb[1])
                if pt is not None and -w * 0.1 < pt[0] < w * 1.1 and -h * 0.1 < pt[1] < h * 1.1:
                    corners.append(pt)

        if len(corners) != 4:
            return None

        return np.array(corners, dtype=np.float32)

    def _merge_lines(self, rho_theta: np.ndarray,
                     rho_thr: float = 30, theta_thr: float = 0.15) -> np.ndarray:
        """Merge Hough lines that are very close in (rho, theta) space."""
        if len(rho_theta) == 0:
            return rho_theta
        used = set()
        merged = []
        for i in range(len(rho_theta)):
            if i in used:
                continue
            group_rho = [rho_theta[i, 0]]
            group_theta = [rho_theta[i, 1]]
            for j in range(i + 1, len(rho_theta)):
                if j in used:
                    continue
                if (abs(rho_theta[i, 0] - rho_theta[j, 0]) < rho_thr
                        and abs(rho_theta[i, 1] - rho_theta[j, 1]) < theta_thr):
                    group_rho.append(rho_theta[j, 0])
                    group_theta.append(rho_theta[j, 1])
                    used.add(j)
            merged.append([np.mean(group_rho), np.mean(group_theta)])
        return np.array(merged, dtype=np.float64)

    @staticmethod
    def _line_intersection(rho1, theta1, rho2, theta2) -> Optional[np.ndarray]:
        """Intersection of two Hough lines (rho, theta parameterization)."""
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)],
        ])
        b = np.array([rho1, rho2])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return np.linalg.solve(A, b)

    # ── Corner ordering ──

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as: near_left, near_right, far_left, far_right.
        Near court = bottom of image (higher y). Far = top (lower y).
        """
        sorted_by_y = pts[np.argsort(-pts[:, 1])]   # descending y
        near = sorted_by_y[:2]
        far = sorted_by_y[2:]
        near = near[np.argsort(near[:, 0])]   # left, right
        far = far[np.argsort(far[:, 0])]
        return np.array([near[0], near[1], far[0], far[1]], dtype=np.float32)

    # ── Validation ──

    @staticmethod
    def _validate_homography(H: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """Sanity-check the homography by verifying mapped dimensions look reasonable."""
        h, w = frame_shape
        # Map the 4 frame corners
        test = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(test, H).reshape(-1, 2)
        x_range = mapped[:, 0].max() - mapped[:, 0].min()
        y_range = mapped[:, 1].max() - mapped[:, 1].min()
        if x_range < 1 or y_range < 1:
            return False
        # Court aspect ratio should be vaguely in range
        ratio = y_range / x_range
        if ratio < 0.3 or ratio > 8.0:
            return False
        # The mapped area should overlap with the actual court (0..10, 0..20)
        if mapped[:, 0].max() < 0 or mapped[:, 0].min() > COURT_WIDTH:
            return False
        if mapped[:, 1].max() < 0 or mapped[:, 1].min() > COURT_LENGTH:
            return False
        return True


# ──────────────────────────────────────────────────────
# MANUAL CALIBRATOR (fallback)
# ──────────────────────────────────────────────────────
class ManualCourtCalibrator:
    """
    Interactive calibration: user clicks 12 court landmarks on the video frame.
    Shows a mini court diagram highlighting which point to click next,
    plus a human-readable description of the point's location.

    Controls:
        Left-click  = place point
        r           = undo last point
        q           = finish (needs >= 4 points, all 12 recommended)
    """

    # Mini court diagram dimensions (pixels)
    _DIAGRAM_W = 160
    _DIAGRAM_H = 320
    _DIAGRAM_MARGIN = 20

    def __init__(self):
        self.pixel_points: List[Tuple[float, float]] = []
        self.court_points: List[Tuple[float, float]] = []
        self.H: Optional[np.ndarray] = None
        self._current_frame: Optional[np.ndarray] = None
        self._display_frame: Optional[np.ndarray] = None
        self._point_names: List[str] = []
        self._current_idx: int = 0
        self._done: bool = False
        self._scale: float = 1.0

    def calibrate(self, frame: np.ndarray,
                  reference_point_names: Optional[List[str]] = None) -> np.ndarray:
        if reference_point_names is None:
            reference_point_names = DEFAULT_CALIBRATION_POINTS

        self._point_names = reference_point_names
        self._current_idx = 0
        self._done = False
        self.pixel_points = []
        self.court_points = []
        self._current_frame = frame.copy()
        h, w = frame.shape[:2]
        self._scale = 1280 / w if w > 1280 else 1.0

        # Print calibration guide to console
        self._print_guide()

        self._redraw()

        win = "Court Calibration - Click reference points"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self._on_click)

        while not self._done:
            cv2.imshow(win, self._display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") and len(self.pixel_points) >= 4:
                self._done = True
            elif key == ord("r") and self.pixel_points:
                self.pixel_points.pop()
                self.court_points.pop()
                self._current_idx -= 1
                self._redraw()

        cv2.destroyWindow(win)

        if len(self.pixel_points) < 4:
            raise ValueError(f"Need >=4 points, got {len(self.pixel_points)}")

        px = np.array(self.pixel_points, dtype=np.float32)
        cx = np.array(self.court_points, dtype=np.float32)
        H, _ = cv2.findHomography(px, cx, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Homography computation failed")
        self.H = H
        return self.H

    # ── console guide ──

    def _print_guide(self):
        """Print a text guide to the console explaining all calibration points."""
        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │           MANUAL CALIBRATION GUIDE                  │")
        print("  │                                                     │")
        print("  │  Click 12 court points in order. A mini court       │")
        print("  │  diagram in the top-right shows which point is      │")
        print("  │  next (highlighted in RED).                         │")
        print("  │                                                     │")
        print("  │  Controls: Left-click = place, R = undo, Q = done  │")
        print("  │  (minimum 4 points, all 12 recommended)             │")
        print("  └─────────────────────────────────────────────────────┘")
        print()
        print("  Point order and locations:")
        print("  ─────────────────────────────────────────────────────")
        for i, name in enumerate(self._point_names):
            desc = CALIBRATION_POINT_DESCRIPTIONS.get(name, name)
            print(f"   {i+1:2d}. {name:25s} → {desc}")
        print()
        print("  Court layout (as seen from camera, near = bottom):")
        print()
        print("     FAR LEFT ──────────────────── FAR RIGHT")
        print("       │                              │")
        print("       │  far_svc_L ─ far_svc_C ─ far_svc_R")
        print("       │             │                │")
        print("       │             │  (service      │")
        print("       │             │   boxes)       │")
        print("       │             │                │")
        print("     NET LEFT ─── ═══NET═══ ──── NET RIGHT")
        print("       │             │                │")
        print("       │             │  (service      │")
        print("       │             │   boxes)       │")
        print("       │             │                │")
        print("       │  nr_svc_L ── nr_svc_C ── nr_svc_R")
        print("       │                              │")
        print("     NEAR LEFT ─────────────────── NEAR RIGHT")
        print()

    # ── drawing helpers ──

    def _redraw(self):
        """Redraw the display frame with clicked points, description, and mini court."""
        self._display_frame = (
            cv2.resize(self._current_frame, None, fx=self._scale, fy=self._scale)
            if self._scale != 1.0 else self._current_frame.copy()
        )

        dh, dw = self._display_frame.shape[:2]

        # Draw already-clicked points
        for i, (px, py) in enumerate(self.pixel_points):
            dp = (int(px * self._scale), int(py * self._scale))
            cv2.circle(self._display_frame, dp, 7, (0, 255, 0), -1)
            cv2.circle(self._display_frame, dp, 7, (0, 0, 0), 2)
            label = f"{i+1}"
            cv2.putText(self._display_frame, label,
                        (dp[0] + 10, dp[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Top bar: point name, number, and description
        if self._current_idx < len(self._point_names):
            name = self._point_names[self._current_idx]
            desc = CALIBRATION_POINT_DESCRIPTIONS.get(name, "")
            num = self._current_idx + 1
            total = len(self._point_names)

            # Dark background bar
            cv2.rectangle(self._display_frame, (0, 0), (dw, 80), (30, 30, 30), -1)

            # Title line
            title = f"Point {num}/{total}:  {name}"
            cv2.putText(self._display_frame, title, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Description line
            cv2.putText(self._display_frame, desc, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Controls
            cv2.putText(self._display_frame, "R=undo  Q=done (min 4 pts)", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        else:
            cv2.rectangle(self._display_frame, (0, 0), (dw, 45), (30, 30, 30), -1)
            cv2.putText(self._display_frame, "All 12 points set! Press Q to finish.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw mini court diagram in top-right corner
        self._draw_mini_court()

    def _draw_mini_court(self):
        """Draw a mini court diagram in the top-right showing which point is active."""
        dh, dw = self._display_frame.shape[:2]
        m = self._DIAGRAM_MARGIN
        cw = self._DIAGRAM_W
        ch = self._DIAGRAM_H

        # Position: top-right
        ox = dw - cw - m
        oy = m + 80  # below the top bar

        # Semi-transparent background
        overlay = self._display_frame.copy()
        cv2.rectangle(overlay, (ox - 5, oy - 5), (ox + cw + 5, oy + ch + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self._display_frame, 0.3, 0, self._display_frame)

        # Scale factors: court meters → diagram pixels
        sx = cw / COURT_WIDTH
        sy = ch / COURT_LENGTH

        def court_to_diag(cx, cy):
            """Convert court coords (meters) to diagram pixel coords."""
            # Flip Y so near wall (y=0) is at bottom of diagram
            return (int(ox + cx * sx), int(oy + ch - cy * sy))

        line_color = (200, 200, 200)
        lt = 1

        # Court outline
        cv2.rectangle(self._display_frame,
                       court_to_diag(0, COURT_LENGTH),
                       court_to_diag(COURT_WIDTH, 0), line_color, lt)

        # Near service line
        cv2.line(self._display_frame,
                 court_to_diag(0, SERVICE_LINE_DIST),
                 court_to_diag(COURT_WIDTH, SERVICE_LINE_DIST), line_color, lt)

        # Far service line
        far_sl = COURT_LENGTH - SERVICE_LINE_DIST
        cv2.line(self._display_frame,
                 court_to_diag(0, far_sl),
                 court_to_diag(COURT_WIDTH, far_sl), line_color, lt)

        # Net
        cv2.line(self._display_frame,
                 court_to_diag(0, NET_Y),
                 court_to_diag(COURT_WIDTH, NET_Y), (0, 200, 255), 2)

        # Center service lines (service line to net on each side)
        cv2.line(self._display_frame,
                 court_to_diag(CENTER_SERVICE_X, SERVICE_LINE_DIST),
                 court_to_diag(CENTER_SERVICE_X, NET_Y), line_color, lt)
        cv2.line(self._display_frame,
                 court_to_diag(CENTER_SERVICE_X, far_sl),
                 court_to_diag(CENTER_SERVICE_X, NET_Y), line_color, lt)

        # Draw all reference points
        for i, name in enumerate(self._point_names):
            cx, cy = COURT_REFERENCE_POINTS_2D[name]
            px, py = court_to_diag(cx, cy)

            if i < self._current_idx:
                # Already clicked — green
                cv2.circle(self._display_frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(self._display_frame, str(i + 1), (px + 6, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == self._current_idx:
                # Current — red pulsing
                cv2.circle(self._display_frame, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(self._display_frame, (px, py), 11, (0, 0, 255), 2)
                cv2.putText(self._display_frame, str(i + 1), (px + 12, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            else:
                # Upcoming — gray
                cv2.circle(self._display_frame, (px, py), 4, (120, 120, 120), -1)
                cv2.putText(self._display_frame, str(i + 1), (px + 6, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # Label
        cv2.putText(self._display_frame, "Court Guide", (ox, oy - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def _on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self._current_idx >= len(self._point_names):
            return
        ox, oy = x / self._scale, y / self._scale
        name = self._point_names[self._current_idx]
        self.pixel_points.append((ox, oy))
        self.court_points.append(COURT_REFERENCE_POINTS_2D[name])
        print(f"    Point {self._current_idx + 1}: {name} → pixel ({ox:.0f}, {oy:.0f})")
        self._current_idx += 1
        self._redraw()
        if self._current_idx >= len(self._point_names):
            self._done = True

    def transform_point(self, px, py):
        pt = np.array([px, py, 1.0], dtype=np.float64)
        r = self.H @ pt
        r /= r[2]
        return (float(r[0]), float(r[1]))

    def transform_points(self, pixel_points):
        pts = pixel_points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def save_calibration(self, path):
        np.savez(path, H=self.H,
                 pixel_points=np.array(self.pixel_points),
                 court_points=np.array(self.court_points))
        print(f"  Calibration saved to: {path}")

    def load_calibration(self, path):
        data = np.load(path)
        self.H = data["H"]
        self.pixel_points = data["pixel_points"].tolist()
        self.court_points = data["court_points"].tolist()
        print(f"  Calibration loaded from: {path}")
        return self.H
