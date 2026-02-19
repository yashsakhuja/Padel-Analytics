"""
2D bird's-eye-view court minimap rendering with player positions.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

from src.config import (
    COURT_LENGTH,
    COURT_WIDTH,
    SERVICE_LINE_DIST,
    NET_Y,
    CENTER_SERVICE_X,
    MINIMAP_SCALE,
    MINIMAP_WIDTH,
    MINIMAP_HEIGHT,
    MINIMAP_BG_COLOR,
    MINIMAP_LINE_COLOR,
    MINIMAP_LINE_THICKNESS,
    MINIMAP_NET_COLOR,
    MINIMAP_NET_THICKNESS,
    PLAYER_DOT_RADIUS,
    TEAM_COLORS,
    PLAYER_LABEL_FONT_SCALE,
)
from src.player_tracker import TrackedPlayer


class MinimapRenderer:
    """
    Renders a 2D padel court minimap with player positions overlaid.
    Court origin (0,0) = near back wall bottom-left.
    Y-axis is inverted for image rendering (0m at bottom, 20m at top).
    """

    def __init__(self):
        self._court_template = self._draw_court_template()

    def _court_to_pixel(self, court_x: float, court_y: float) -> Tuple[int, int]:
        """Convert court coordinates (meters) to minimap pixel coordinates."""
        px = int(court_x * MINIMAP_SCALE)
        py = int((COURT_LENGTH - court_y) * MINIMAP_SCALE)  # invert Y
        return (px, py)

    def _draw_court_template(self) -> np.ndarray:
        """Draw the static court diagram with all standard padel markings."""
        court = np.full(
            (MINIMAP_HEIGHT, MINIMAP_WIDTH, 3), MINIMAP_BG_COLOR, dtype=np.uint8
        )
        color = MINIMAP_LINE_COLOR
        thick = MINIMAP_LINE_THICKNESS

        # Court boundary
        tl = self._court_to_pixel(0, COURT_LENGTH)
        br = self._court_to_pixel(COURT_WIDTH, 0)
        cv2.rectangle(court, tl, br, color, thick)

        # Net line
        net_l = self._court_to_pixel(0, NET_Y)
        net_r = self._court_to_pixel(COURT_WIDTH, NET_Y)
        cv2.line(court, net_l, net_r, MINIMAP_NET_COLOR, MINIMAP_NET_THICKNESS)

        # Near service line
        nsl = self._court_to_pixel(0, SERVICE_LINE_DIST)
        nsr = self._court_to_pixel(COURT_WIDTH, SERVICE_LINE_DIST)
        cv2.line(court, nsl, nsr, color, thick)

        # Far service line
        far_service_y = COURT_LENGTH - SERVICE_LINE_DIST
        fsl = self._court_to_pixel(0, far_service_y)
        fsr = self._court_to_pixel(COURT_WIDTH, far_service_y)
        cv2.line(court, fsl, fsr, color, thick)

        # Near center service line (vertical, from near service line to net)
        nc_top = self._court_to_pixel(CENTER_SERVICE_X, NET_Y)
        nc_bot = self._court_to_pixel(CENTER_SERVICE_X, SERVICE_LINE_DIST)
        cv2.line(court, nc_top, nc_bot, color, thick)

        # Far center service line (vertical, from far service line to net)
        fc_top = self._court_to_pixel(CENTER_SERVICE_X, far_service_y)
        fc_bot = self._court_to_pixel(CENTER_SERVICE_X, NET_Y)
        cv2.line(court, fc_top, fc_bot, color, thick)

        return court

    def render(self, players: List[TrackedPlayer]) -> np.ndarray:
        """Render the minimap with current player positions."""
        minimap = self._court_template.copy()

        for player in players:
            if player.court_position is None:
                continue

            cx, cy = player.court_position
            px, py = self._court_to_pixel(cx, cy)

            # Clamp to minimap boundaries
            px = max(PLAYER_DOT_RADIUS, min(MINIMAP_WIDTH - PLAYER_DOT_RADIUS, px))
            py = max(PLAYER_DOT_RADIUS, min(MINIMAP_HEIGHT - PLAYER_DOT_RADIUS, py))

            # Team color (default white)
            color = TEAM_COLORS.get(player.team_id, (255, 255, 255))

            # Draw player dot with contrasting outline
            cv2.circle(minimap, (px, py), PLAYER_DOT_RADIUS, color, -1)
            outline = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
            cv2.circle(minimap, (px, py), PLAYER_DOT_RADIUS, outline, 2)

            # Draw label
            label = f"P{player.track_id}"
            cv2.putText(
                minimap,
                label,
                (px - 10, py - PLAYER_DOT_RADIUS - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                PLAYER_LABEL_FONT_SCALE,
                (255, 255, 255),
                1,
            )

        return minimap

    def composite(
        self,
        video_frame: np.ndarray,
        minimap: np.ndarray,
        position: str = "right",
    ) -> np.ndarray:
        """
        Combine the video frame and minimap.
        position: "right" (side-by-side) or "overlay" (bottom-right corner).
        """
        if position == "overlay":
            return self._composite_overlay(video_frame, minimap)
        else:
            return self._composite_right(video_frame, minimap)

    def _composite_right(
        self, video_frame: np.ndarray, minimap: np.ndarray
    ) -> np.ndarray:
        """Place minimap to the right of the video frame."""
        vh, vw = video_frame.shape[:2]
        mh, mw = minimap.shape[:2]

        # Resize minimap to match video height
        scale = vh / mh
        new_mw = int(mw * scale)
        minimap_resized = cv2.resize(minimap, (new_mw, vh))

        return np.hstack([video_frame, minimap_resized])

    def _composite_overlay(
        self, video_frame: np.ndarray, minimap: np.ndarray
    ) -> np.ndarray:
        """Overlay minimap in the bottom-right corner of the video frame."""
        frame = video_frame.copy()
        vh, vw = frame.shape[:2]

        # Scale minimap to ~30% of video height
        target_h = int(vh * 0.3)
        mh, mw = minimap.shape[:2]
        scale = target_h / mh
        new_mw = int(mw * scale)
        minimap_resized = cv2.resize(minimap, (new_mw, target_h))

        # Position in bottom-right with 10px margin
        margin = 10
        y1 = vh - target_h - margin
        y2 = vh - margin
        x1 = vw - new_mw - margin
        x2 = vw - margin

        # Alpha blend
        roi = frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(roi, 0.3, minimap_resized, 0.7, 0)
        frame[y1:y2, x1:x2] = blended

        return frame
