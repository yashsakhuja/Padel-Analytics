"""
Multi-object tracking with YOLOv8 + ByteTrack and team assignment via jersey color.

Key improvements for consistent 4-player tracking:
  - Track age: requires N consecutive frames before accepting a new track
  - Sticky lock: once 4 players are established, strongly prefer keeping them
  - Confidence bonus for established tracks prevents flickering to new detections
"""
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

from src.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_PERSON_CLASS,
    TRACKER_CONFIG,
    MAX_BBOX_AREA_RATIO,
    MIN_BBOX_AREA_RATIO,
    MIN_FOOT_Y_RATIO,
    MAX_PLAYERS,
    MIN_TRACK_AGE,
    STICKY_TRACK_BONUS,
    COLOR_CROP_TOP_RATIO,
    COLOR_CROP_BOTTOM_RATIO,
    KMEANS_N_CLUSTERS,
    TEAM_ASSIGNMENT_INTERVAL,
)


@dataclass
class TrackedPlayer:
    """A tracked player with persistent ID, position, and team."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    foot_position: np.ndarray  # [x, y] in pixel coords
    court_position: Optional[np.ndarray] = None  # [x, y] in meters
    team_id: Optional[int] = None
    dominant_color: Optional[np.ndarray] = None  # BGR dominant jersey color


class PlayerTracker:
    """
    Combines YOLO detection + ByteTrack tracking + jersey color team assignment.
    Uses model.track() with persist=True for internal ByteTrack association.

    Track stability features:
      - Track age counting: new tracks need MIN_TRACK_AGE frames to be accepted
      - Sticky player lock: once 4 players are established, they get a confidence
        bonus so they aren't replaced by transient detections
      - Frames-since-seen counter: tracks are dropped after being unseen for a while
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        confidence: float = YOLO_CONFIDENCE,
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.team_assignments: Dict[int, int] = {}
        self._color_samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._frame_count = 0
        self._position_history: Dict[int, List[np.ndarray]] = defaultdict(list)

        # Track persistence state
        self._track_ages: Dict[int, int] = defaultdict(int)       # frames seen
        self._track_last_seen: Dict[int, int] = {}                # last frame seen
        self._active_players: Set[int] = set()                    # locked 4 player IDs
        self._max_unseen_frames = 180  # drop track after 6s unseen (match ByteTrack buffer)

    def update(self, frame: np.ndarray) -> List[TrackedPlayer]:
        """
        Run detection + tracking on a frame.
        Returns list of up to 4 TrackedPlayer.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            classes=[YOLO_PERSON_CLASS],
            tracker=TRACKER_CONFIG,
            verbose=False,
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            self._frame_count += 1
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id

        if track_ids is None:
            self._frame_count += 1
            return []

        track_ids = track_ids.cpu().numpy().astype(int)

        # Build TrackedPlayer list
        players = []
        seen_ids = set()
        for i in range(len(boxes_xyxy)):
            bbox = boxes_xyxy[i]
            tid = int(track_ids[i])
            foot_x = (bbox[0] + bbox[2]) / 2.0
            foot_y = bbox[3]  # bottom of bbox
            players.append(
                TrackedPlayer(
                    track_id=tid,
                    bbox=bbox,
                    confidence=float(confs[i]),
                    foot_position=np.array([foot_x, foot_y]),
                )
            )
            seen_ids.add(tid)

        # Update track ages
        for tid in seen_ids:
            self._track_ages[tid] += 1
            self._track_last_seen[tid] = self._frame_count

        # Expire stale tracks
        stale = [
            tid for tid, last in self._track_last_seen.items()
            if self._frame_count - last > self._max_unseen_frames
        ]
        for tid in stale:
            self._track_ages.pop(tid, None)
            self._track_last_seen.pop(tid, None)
            self._active_players.discard(tid)

        # Filter to court players only (with sticky logic)
        players = self._filter_to_court_players(players, frame.shape)

        # Update team assignments periodically
        if self._frame_count % TEAM_ASSIGNMENT_INTERVAL == 0 and len(players) >= 2:
            self._update_team_assignments(players, frame)

        # Apply stored team assignments
        for player in players:
            if player.track_id in self.team_assignments:
                player.team_id = self.team_assignments[player.track_id]

        self._frame_count += 1
        return players

    def _filter_to_court_players(
        self, players: List[TrackedPlayer], frame_shape: Tuple
    ) -> List[TrackedPlayer]:
        """
        Filter detections to the most likely 4 court players.

        Two-tier system:
          1. Basic spatial/size filters (reject obvious non-players)
          2. Sticky scoring: established tracks get a confidence bonus so
             new transient detections don't displace them
        """
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w
        filtered = []

        for p in players:
            bbox_w = p.bbox[2] - p.bbox[0]
            bbox_h = p.bbox[3] - p.bbox[1]
            bbox_area = bbox_w * bbox_h

            # Reject if bbox is too large (close-up or non-player)
            if bbox_area / frame_area > MAX_BBOX_AREA_RATIO:
                continue

            # Reject if bbox is too small (distant spectators, ball boys)
            if bbox_area / frame_area < MIN_BBOX_AREA_RATIO:
                continue

            # Reject if foot position is in the top portion of frame (spectators)
            if p.foot_position[1] / frame_h < MIN_FOOT_Y_RATIO:
                continue

            filtered.append(p)

        if not filtered:
            return []

        # Score each player: raw confidence + bonus for established tracks
        def player_score(p):
            score = p.confidence
            age = self._track_ages.get(p.track_id, 0)

            # Bonus for being an active (locked) player
            if p.track_id in self._active_players:
                score += STICKY_TRACK_BONUS

            # Small age bonus (capped) for track persistence
            score += min(age / 100.0, 0.1)

            return score

        # Remove tracks that haven't matured yet (unless we need them to fill 4)
        mature = [p for p in filtered if self._track_ages.get(p.track_id, 0) >= MIN_TRACK_AGE]
        immature = [p for p in filtered if self._track_ages.get(p.track_id, 0) < MIN_TRACK_AGE]

        # Prefer mature tracks; only add immature if we don't have enough
        mature.sort(key=player_score, reverse=True)
        selected = mature[:MAX_PLAYERS]

        if len(selected) < MAX_PLAYERS:
            immature.sort(key=player_score, reverse=True)
            selected.extend(immature[:MAX_PLAYERS - len(selected)])

        # Update active player set
        self._active_players = {p.track_id for p in selected}

        return selected

    def _extract_jersey_color(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract dominant jersey color from the torso region of a player bbox."""
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        h, w = crop.shape[:2]
        # Isolate torso region
        top = int(h * COLOR_CROP_TOP_RATIO)
        bottom = int(h * COLOR_CROP_BOTTOM_RATIO)
        torso = crop[top:bottom, :]

        if torso.size == 0:
            return None

        # Resize for efficiency
        torso_resized = cv2.resize(torso, (32, 32))
        pixels = torso_resized.reshape(-1, 3).astype(np.float32)

        # KMeans to find dominant colors
        kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, n_init=3, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Find the largest cluster that isn't skin-toned
        counts = np.bincount(labels, minlength=KMEANS_N_CLUSTERS)
        sorted_indices = np.argsort(-counts)

        for idx in sorted_indices:
            color_bgr = centers[idx]
            # Convert to HSV to check for skin tone
            color_hsv = cv2.cvtColor(
                np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV
            )[0, 0]
            # Skip skin-like colors (low saturation warm hues)
            if color_hsv[0] < 30 and color_hsv[1] < 80:
                continue
            return color_bgr.astype(np.uint8)

        # Fallback: return largest cluster
        return centers[sorted_indices[0]].astype(np.uint8)

    def _update_team_assignments(
        self, players: List[TrackedPlayer], frame: np.ndarray
    ) -> None:
        """Cluster current players into 2 teams based on jersey color."""
        colors = []
        valid_players = []

        for player in players:
            color = self._extract_jersey_color(frame, player.bbox)
            if color is not None:
                colors.append(color)
                valid_players.append(player)
                player.dominant_color = color
                self._color_samples[player.track_id].append(color)

        if len(colors) < 2:
            return

        color_array = np.array(colors, dtype=np.float32)
        n_teams = min(2, len(color_array))
        kmeans = KMeans(n_clusters=n_teams, n_init=5, random_state=42)
        labels = kmeans.fit_predict(color_array)

        for i, player in enumerate(valid_players):
            new_team = int(labels[i])
            old_team = self.team_assignments.get(player.track_id)

            # Only update if no previous assignment or enough samples to be confident
            if old_team is None or len(self._color_samples[player.track_id]) < 3:
                self.team_assignments[player.track_id] = new_team
            elif new_team != old_team:
                # Check if the new assignment is consistent with recent samples
                recent = self._color_samples[player.track_id][-3:]
                recent_arr = np.array(recent, dtype=np.float32)
                dists = np.linalg.norm(
                    recent_arr - kmeans.cluster_centers_[new_team], axis=1
                )
                if np.mean(dists) < 80:  # color distance threshold
                    self.team_assignments[player.track_id] = new_team

    def store_court_position(self, track_id: int, court_pos: np.ndarray):
        """Store a court position for heatmap generation."""
        self._position_history[track_id].append(court_pos.copy())

    def get_position_history(self) -> Dict[int, List[np.ndarray]]:
        """Return accumulated court positions per track ID."""
        return dict(self._position_history)
