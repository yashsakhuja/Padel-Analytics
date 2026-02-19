"""
Padel Court Analytics - Configuration & Constants.
All court measurements in meters (per IPF standard rules).
"""

# ──────────────────────────────────────────────
# VIDEO SOURCE
# ──────────────────────────────────────────────
VIDEO_SOURCE = "https://www.youtube.com/live/ALAG6Q_hOFM?si=OCXL8mjp-ZWTzakB"
VIDEO_START_TIME = "2:17:41"   # HH:MM:SS or MM:SS
VIDEO_END_TIME = "2:18:08"     # HH:MM:SS or MM:SS

# ──────────────────────────────────────────────
# PADEL COURT DIMENSIONS (meters)
# ──────────────────────────────────────────────
COURT_LENGTH = 20.0
COURT_WIDTH = 10.0
SERVICE_LINE_DIST = 3.05     # from back wall to service line (6.95m from net)
NET_Y = 10.0                 # net position (center of court length)
CENTER_SERVICE_X = 5.0       # center service line (half court width)

# Reference points on the court (meters, origin = bottom-left corner)
#
#   (0,20) ──────────── (10,20)    <- far back wall
#     |                    |
#     |  (0,16.95) (10,16.95)      <- far service line (3.05m from far wall)
#     |      (5,16.95)             <- far center service
#     |                    |
#     |   ---- NET ----    |        y = 10.0
#     |                    |
#     |  (0,3.05)  (10,3.05)       <- near service line (3.05m from near wall)
#     |      (5,3.05)              <- near center service
#     |                    |
#   (0,0) ───────────── (10,0)     <- near back wall

COURT_REFERENCE_POINTS_2D = {
    "near_left":           (0.0, 0.0),
    "near_right":          (10.0, 0.0),
    "far_left":            (0.0, 20.0),
    "far_right":           (10.0, 20.0),
    "near_service_left":   (0.0, 3.05),
    "near_service_right":  (10.0, 3.05),
    "near_service_center": (5.0, 3.05),
    "far_service_left":    (0.0, 16.95),
    "far_service_right":   (10.0, 16.95),
    "far_service_center":  (5.0, 16.95),
    "net_left":            (0.0, 10.0),
    "net_right":           (10.0, 10.0),
}

# Default calibration point order (user clicks these in sequence)
# Uses all 12 visible court intersections for maximum accuracy with RANSAC.
DEFAULT_CALIBRATION_POINTS = [
    "near_left",
    "near_right",
    "near_service_left",
    "near_service_right",
    "near_service_center",
    "net_left",
    "net_right",
    "far_service_left",
    "far_service_right",
    "far_service_center",
    "far_left",
    "far_right",
]

# Human-readable descriptions for each calibration point.
# Shown in the calibration window to guide the user.
CALIBRATION_POINT_DESCRIPTIONS = {
    "near_left":           "Bottom-left corner (back wall meets left glass)",
    "near_right":          "Bottom-right corner (back wall meets right glass)",
    "near_service_left":   "Left end of NEAR service line (line meets left wall)",
    "near_service_right":  "Right end of NEAR service line (line meets right wall)",
    "near_service_center": "Center T on NEAR service line (center line meets service line)",
    "net_left":            "Left net post (where net meets left glass wall)",
    "net_right":           "Right net post (where net meets right glass wall)",
    "far_service_left":    "Left end of FAR service line (line meets left wall)",
    "far_service_right":   "Right end of FAR service line (line meets right wall)",
    "far_service_center":  "Center T on FAR service line (center line meets service line)",
    "far_left":            "Top-left corner (far back wall meets left glass)",
    "far_right":           "Top-right corner (far back wall meets right glass)",
}

# ──────────────────────────────────────────────
# MINIMAP RENDERING
# ──────────────────────────────────────────────
MINIMAP_SCALE = 30           # pixels per meter
MINIMAP_WIDTH = int(COURT_WIDTH * MINIMAP_SCALE)    # 300
MINIMAP_HEIGHT = int(COURT_LENGTH * MINIMAP_SCALE)  # 600
MINIMAP_BG_COLOR = (180, 120, 40)       # BGR padel court blue
MINIMAP_LINE_COLOR = (255, 255, 255)    # white
MINIMAP_LINE_THICKNESS = 2
MINIMAP_NET_COLOR = (200, 200, 200)     # light gray for net
MINIMAP_NET_THICKNESS = 3

PLAYER_DOT_RADIUS = 10
TEAM_COLORS = {
    0: (255, 255, 255),  # BGR - Team A (white jerseys)
    1: (40, 40, 40),     # BGR - Team B (black jerseys)
}
PLAYER_LABEL_FONT_SCALE = 0.5

# ──────────────────────────────────────────────
# DETECTION / TRACKING
# ──────────────────────────────────────────────
YOLO_MODEL = "yolo11m.pt"    # medium model — much better accuracy on GPU
YOLO_CONFIDENCE = 0.55       # higher threshold to cut false positives
YOLO_PERSON_CLASS = 0        # COCO class index for "person"
YOLO_BALL_CLASS = 32         # COCO class index for "sports ball"
YOLO_BALL_CONFIDENCE = 0.10  # very low threshold — ball is small, Kalman filter rejects false positives
TRACKER_CONFIG = "padel_bytetrack.yaml"

BALL_DOT_RADIUS = 6
BALL_COLOR = (0, 255, 255)   # BGR yellow — high visibility on blue court

# Ball tracking (Kalman filter + interpolation)
BALL_MAX_GAP = 65                 # max frames to predict without detection (~2s at 30fps)
BALL_MAX_PREDICTION_FRAMES = 60   # max frames of Kalman-only prediction (~2s at 30fps)
BALL_SEARCH_ROI_SCALE = 3.0       # ROI size multiplier for local search
BALL_VALIDATION_DISTANCE = 200.0  # max pixel dist between detection and prediction

# ──────────────────────────────────────────────
# PLAYER FILTERING
# ──────────────────────────────────────────────
MAX_BBOX_AREA_RATIO = 0.25   # reject detections > 25% of frame area (allow players near glass)
MIN_BBOX_AREA_RATIO = 0.002  # reject detections < 0.2% of frame area (distant spectators)
MIN_FOOT_Y_RATIO = 0.20      # reject detections with foot_y in top 20%
MAX_PLAYERS = 4
MIN_TRACK_AGE = 15           # require N consecutive frames before accepting a track
STICKY_TRACK_BONUS = 0.5     # confidence bonus for established (sticky) players

# Court bounds for post-homography validation (meters beyond court edge)
COURT_BOUNDS_MARGIN = 2.0

# ──────────────────────────────────────────────
# SCENE CHANGE DETECTION
# ──────────────────────────────────────────────
SCENE_CHANGE_THRESHOLD = 0.4    # histogram correlation below this = camera cut
REFERENCE_MATCH_THRESHOLD = 0.25  # frame vs reference (calibration) frame
SCENE_CHANGE_COOLDOWN = 10      # skip N frames after a scene change

# ──────────────────────────────────────────────
# TEAM ASSIGNMENT
# ──────────────────────────────────────────────
COLOR_CROP_TOP_RATIO = 0.15
COLOR_CROP_BOTTOM_RATIO = 0.55
KMEANS_N_CLUSTERS = 3
TEAM_ASSIGNMENT_INTERVAL = 30  # re-evaluate team colors every N frames

# ──────────────────────────────────────────────
# HEATMAP
# ──────────────────────────────────────────────
HEATMAP_GRID_RESOLUTION = 200
HEATMAP_ALPHA = 0.7
HEATMAP_COURT_COLOR = "#2878B4"  # matches minimap BGR (180,120,40)

# ──────────────────────────────────────────────
# VIDEO OUTPUT
# ──────────────────────────────────────────────
OUTPUT_FPS = 30
OUTPUT_CODEC = "avc1"       # H.264 — browser-playable in Streamlit dashboard
PROCESS_EVERY_N_FRAMES = 1
