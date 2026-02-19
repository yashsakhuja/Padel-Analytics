# Padel Court Analytics

Fully automated computer vision pipeline that downloads a padel match from YouTube, detects and tracks 4 players, maps their positions onto a 2D bird's-eye court minimap, generates post-match heatmaps, and opens an interactive Streamlit dashboard — all with a single command, zero user interaction.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v11n-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## Quick Start

```bash
pip install -r requirements.txt
pip install streamlit
python main.py
```

That's it. The pipeline will:
1. Download the YouTube video automatically
2. Auto-detect the court lines and compute the homography (no clicking)
3. Track all 4 players frame by frame (headless, no windows)
4. Generate heatmaps and analytics
5. Launch the Streamlit dashboard in your browser

For a quick test run:
```bash
python main.py --max-frames 500
```

---

## How It Works

```
 python main.py
      |
      v
 ┌──────────────────┐
 |  YouTube Download  |   yt-dlp, auto 1080p
 └────────┬─────────┘
          v
 ┌──────────────────┐
 |  Auto Court       |   Color segmentation + Hough lines + contour
 |  Detection        |   3 strategies with fallback
 |  (no user input)  |   -> homography matrix H (pixel -> meters)
 └────────┬─────────┘
          v
 ┌──────────── FRAME LOOP (headless) ─────────────┐
 |                                                  |
 |   YOLOv8 nano person detection                   |
 |        |                                         |
 |   ByteTrack multi-object tracking                |
 |        |                                         |
 |   Jersey color KMeans -> team assignment          |
 |        |                                         |
 |   Foot position -> homography -> court (x,y)     |
 |        |                                         |
 |   Render 2D minimap + annotate video              |
 |        |                                         |
 |   Write frame to output video                    |
 |                                                  |
 └──────────────────────────────────────────────────┘
          |
          v
 ┌──────────────────┐
 |  Post-Processing  |
 |  - Gaussian KDE heatmaps per player/team         |
 |  - Analytics: distance, speed, zones, coverage   |
 |  - Save match_data.pkl + heatmap PNGs            |
 └────────┬─────────┘
          v
 ┌──────────────────┐
 |  Streamlit        |   Auto-launches in browser
 |  Dashboard        |   3 tabs: Replay / Heatmaps / Analytics
 └──────────────────┘
```

---

## Project Structure

```
Padel Analytics/
|
├── main.py                 # Automated pipeline — single command, zero interaction
├── dashboard.py            # Streamlit dashboard — 3-tab match analysis UI
├── requirements.txt        # Python dependencies
├── README.md
|
├── src/
|   ├── __init__.py
|   ├── config.py           # Court dimensions (20x10m), rendering params, YOLO settings
|   ├── video_utils.py      # Video loading + YouTube download via yt-dlp
|   ├── court_detector.py   # Auto court detection (3 strategies) + manual fallback
|   ├── player_tracker.py   # YOLOv8 + ByteTrack tracking + jersey color team assignment
|   ├── minimap.py          # 2D bird's-eye court rendering with player dots
|   ├── heatmap.py          # Post-match Gaussian KDE heatmap generation
|   └── analytics.py        # Match stats: distance, speed, zones, coverage
|
├── output/                 # Generated after processing
|   ├── tracked_output.mp4      # Video with bounding boxes + minimap
|   ├── tracked_output_raw.mp4  # Video with bounding boxes only
|   ├── calibration.npz         # Saved homography (reusable)
|   ├── match_data.pkl          # Full match data for dashboard
|   ├── match_summary.json      # Lightweight stats summary
|   ├── heatmap_player_*.png    # Per-player heatmaps
|   ├── heatmap_team_*.png      # Per-team heatmaps
|   └── heatmap_combined.png    # 2x2 grid of all players
|
└── raw/                    # Downloaded source videos
```

---

## Module Breakdown

### `config.py` — Constants
Padel court: 20m x 10m, service lines at 6.95m from each back wall. Minimap: 30 px/m (300x600px). YOLO: `yolo11n.pt` nano model, 0.3 confidence, ByteTrack tracker.

### `video_utils.py` — Video I/O
Auto-detects YouTube URLs, downloads at 1080p via yt-dlp, returns OpenCV `VideoCapture` + metadata dict.

### `court_detector.py` — Court Detection

**Default: fully automatic.** Three detection strategies tried in order:

| Strategy | How it works |
|----------|-------------|
| **Color + Lines** | Segments the court surface by HSV color (blue/green/turquoise/orange), finds the largest blob, approximates to a quadrilateral. Falls back to Hough lines within the detected region for refinement. |
| **Hough Lines** | Runs Canny edge detection on the full frame, finds lines via Hough transform, clusters into two perpendicular groups, picks the outermost pair from each group, computes their 4 intersections. |
| **Contour** | Finds the largest quadrilateral contour in the edge-detected frame. |

Each strategy's result is validated: the homography must produce a reasonable court aspect ratio and overlap with the expected 10x20m area. First valid result wins.

**Fallback**: pass `--manual-calibration` to click 6 court points interactively.

**Why foot position?** The homography maps the court plane (the floor). Player feet touch the floor, so we use `bbox bottom-center`. Using the bbox center would project mid-air and give wrong court coordinates.

### `player_tracker.py` — Detection + Tracking + Teams
Uses `model.track(persist=True)` for YOLOv8 + ByteTrack in one call. Filters to 4 players by rejecting oversized bboxes (>15% frame), top-of-frame detections (spectators), and keeping top-4 by confidence.

**Team assignment** (every 30 frames): crops each player's torso (skip head 15%, legs 45%), KMeans(k=3) on 32x32 resized crop for dominant jersey color, KMeans(k=2) across all 4 players for team split. Temporal smoothing prevents flickering.

### `minimap.py` — 2D Court Rendering
Pre-renders a static court template with all padel markings. Each frame: copies template, draws colored dots at player court positions, composites alongside (or overlaid on) the video frame. Y-axis inverted for image coordinates.

### `heatmap.py` — Post-Match Heatmaps
`scipy.stats.gaussian_kde` with Scott bandwidth for smooth density estimates. Generates per-player, per-team, and combined 2x2 grid figures. Filters out-of-bounds noise.

### `analytics.py` — Match Statistics
Computed from position history:
- **Distance covered** — euclidean frame-to-frame sum (filters teleport glitches >5m)
- **Average/max speed** — m/s and km/h
- **Zone distribution** — % time in back court / mid court / net zone / opponent back
- **Court coverage** — % of 1mx1m grid cells visited
- **Side preference** — left vs right court time

### `dashboard.py` — Streamlit Dashboard
Auto-launched after processing. Three tabs:

| Tab | Contents |
|-----|----------|
| **Match Replay** | Video player with minimap, overall heatmap, per-player quick stats (distance, speed, coverage) |
| **Player Heatmaps** | Individual KDE heatmaps, team combined heatmaps, 2x2 combined grid |
| **Full Match Analytics** | Stats table, distance bar chart, zone distribution pie charts, left/right preference bars, team comparison cards |

Sidebar: match metadata + player filter (multi-select).

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --source TEXT            YouTube URL or local video path
                           (default: built-in padel match)
  --calibration TEXT       Load a saved .npz calibration file
  --manual-calibration     Use interactive point clicking instead of auto-detect
  --output TEXT            Output video path (default: output/tracked_output.mp4)
  --display-mode TEXT      Minimap position: "right" or "overlay" (default: right)
  --skip-frames INT        Process every Nth frame (default: 1)
  --max-frames INT         Stop after N frames, 0 = all (default: 0)
  --no-dashboard           Don't auto-launch Streamlit after processing
```

### Examples

```bash
# Full automated run (download + detect + track + heatmap + dashboard)
python main.py

# Quick test on first 500 frames
python main.py --max-frames 500

# Process without launching dashboard
python main.py --no-dashboard

# Reuse calibration from a previous run
python main.py --calibration output/calibration.npz

# Use manual calibration if auto-detect fails
python main.py --manual-calibration

# Custom YouTube video
python main.py --source "https://youtu.be/your_video_id"

# Skip frames for faster processing (2x speed)
python main.py --skip-frames 2

# Launch dashboard separately (after processing)
streamlit run dashboard.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Auto court detection (3 strategies) | Zero user interaction; color segmentation handles most broadcast courts |
| Foot position (bbox bottom-center) | Feet touch the court plane for accurate homography projection |
| YOLOv8 nano (`yolo11n.pt`) | Best speed/accuracy trade-off for real-time tracking |
| ByteTrack (via ultralytics) | Handles brief occlusions, keeps consistent track IDs |
| Jersey color via KMeans | No training needed; torso crop isolates jersey from skin/court |
| Gaussian KDE for heatmaps | Smooth continuous density; Scott bandwidth avoids manual tuning |
| Headless processing | No OpenCV windows; runs on any machine including servers |
| Auto-launch dashboard | Single command from video to interactive analysis |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8/v11 detection + ByteTrack tracking |
| `opencv-python` | Video I/O, image processing, homography, Hough lines |
| `numpy` | Array math, coordinate transforms |
| `yt-dlp` | YouTube video download |
| `matplotlib` | Heatmap rendering, analytics charts |
| `scipy` | Gaussian KDE for heatmaps |
| `scikit-learn` | KMeans for jersey color + team clustering |
| `streamlit` | Interactive web dashboard |

Install everything:
```bash
pip install -r requirements.txt
pip install streamlit
```

---

## Troubleshooting

**Auto court detection fails** — Use `--manual-calibration` to click court points interactively. The auto-detector works best on broadcast-style videos with clear court lines and a colored playing surface.

**"No match data found" in dashboard** — Run `python main.py` first to generate `output/match_data.pkl`.

**Players not detected** — Adjust `YOLO_CONFIDENCE` in `src/config.py` (lower = more detections, more false positives).

**Tracking IDs keep changing** — ByteTrack keeps lost tracks for 2 seconds. For longer occlusions (player behind glass), IDs may reassign. Team colors still recover correctly.

**Slow processing** — Use `--skip-frames 2` or `--skip-frames 3`. GPU (CUDA) gives 5-10x speedup over CPU.

**Dashboard won't launch** — Run it manually: `streamlit run dashboard.py`. Make sure `output/match_data.pkl` exists.
