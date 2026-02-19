"""
Padel Court Analytics - Fully Automated Pipeline.

Downloads the YouTube video, auto-detects the court, tracks players,
generates heatmaps, and launches the Streamlit dashboard.
Zero user interaction required.

Usage:
    python main.py
    python main.py --source "https://youtu.be/..."
    python main.py --manual-calibration
    python main.py --max-frames 500
"""
import argparse
import cv2
import json
import numpy as np
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict

from src.config import (
    VIDEO_SOURCE,
    VIDEO_START_TIME,
    VIDEO_END_TIME,
    PROCESS_EVERY_N_FRAMES,
    OUTPUT_FPS,
    OUTPUT_CODEC,
    TEAM_COLORS,
    COURT_LENGTH,
    COURT_WIDTH,
    COURT_BOUNDS_MARGIN,
    NET_Y,
    SCENE_CHANGE_THRESHOLD,
    REFERENCE_MATCH_THRESHOLD,
    SCENE_CHANGE_COOLDOWN,
    MAX_PLAYERS,
)
from src.video_utils import load_video
from src.court_detector import AutoCourtDetector, ManualCourtCalibrator
from src.player_tracker import PlayerTracker
from src.minimap import MinimapRenderer
from src.heatmap import HeatmapGenerator



def parse_time(time_str):
    """Parse MM:SS or M:SS or raw seconds to float seconds."""
    if time_str is None:
        return None
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(parts[0])


def compute_frame_histogram(frame):
    """Compute normalized HSV histogram for scene change detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def is_on_court(court_xy):
    """Check if a court position is within valid court bounds."""
    x, y = court_xy
    return (
        -COURT_BOUNDS_MARGIN <= x <= COURT_WIDTH + COURT_BOUNDS_MARGIN
        and -COURT_BOUNDS_MARGIN <= y <= COURT_LENGTH + COURT_BOUNDS_MARGIN
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Padel Court Analytics (Automated)")
    parser.add_argument(
        "--source", default=VIDEO_SOURCE,
        help="Video source — YouTube URL or local file (default: built-in YouTube match)",
    )
    parser.add_argument(
        "--calibration", default=None,
        help="Load a saved .npz calibration file (skips court detection)",
    )
    parser.add_argument(
        "--manual-calibration", action="store_true",
        help="Use interactive manual calibration instead of auto-detection",
    )
    parser.add_argument(
        "--output", default="output/tracked_output.mp4",
        help="Output video path (default: output/tracked_output.mp4)",
    )
    parser.add_argument(
        "--display-mode", choices=["right", "overlay"], default="right",
        help="Minimap position in output video (default: right)",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=PROCESS_EVERY_N_FRAMES,
        help=f"Process every Nth frame (default: {PROCESS_EVERY_N_FRAMES})",
    )
    parser.add_argument(
        "--start-time", default=VIDEO_START_TIME,
        help="Start processing at this timestamp HH:MM:SS or MM:SS",
    )
    parser.add_argument(
        "--end-time", default=VIDEO_END_TIME,
        help="Stop processing at this timestamp HH:MM:SS or MM:SS",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--no-scene-filter", action="store_true",
        help="Disable scene change detection (process all frames as gameplay)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip auto-launching the Streamlit dashboard after processing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ──────────────────────────────────────────
    # PHASE 1: LOAD VIDEO
    # ──────────────────────────────────────────
    print("=" * 60)
    print("  PADEL COURT ANALYTICS  (Automated)")
    print("=" * 60)

    # Parse time range before download so we can clip during download
    start_s = parse_time(args.start_time) if args.start_time else 0
    end_s = parse_time(args.end_time) if args.end_time else None

    # Pass time range to load_video so YouTube clips are downloaded efficiently
    from src.video_utils import is_youtube_url
    dl_start = start_s if is_youtube_url(args.source) and start_s > 0 else None
    dl_end = end_s if is_youtube_url(args.source) and end_s is not None else None

    cap, meta = load_video(args.source, start_time=dl_start, end_time=dl_end)
    print(f"\nVideo loaded: {meta['path']}")
    print(f"  Resolution : {meta['width']}x{meta['height']}")
    print(f"  FPS        : {meta['fps']:.1f}")
    print(f"  Duration   : {meta['duration_s']:.1f}s  ({meta['total_frames']} frames)")

    fps = meta["fps"] if meta["fps"] > 0 else OUTPUT_FPS

    # If we downloaded a clipped section, the video starts at 0 — reset offsets
    if dl_start is not None:
        print(f"\n  Clipped download: {args.start_time} - {args.end_time}")
        print(f"  Downloaded clip starts at 0:00 in the file")
        start_s = 0
        end_s = None

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps) if end_s else meta["total_frames"]

    if start_s > 0 or end_s is not None:
        start_label = args.start_time or "0:00"
        end_label = args.end_time or "end"
        clip_dur = (end_s if end_s else meta["duration_s"]) - start_s
        print(f"\n  Time range : {start_label} - {end_label}")
        print(f"  Frames     : {start_frame} - {end_frame} ({end_frame - start_frame} frames, {clip_dur:.1f}s)")

    # Seek to start time and read first frame from there (for calibration)
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame at {args.start_time}")
    # Seek back to start so the main loop reads from the right position
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)

    os.makedirs("output", exist_ok=True)

    # ──────────────────────────────────────────
    # PHASE 2: COURT CALIBRATION (auto or manual)
    # ──────────────────────────────────────────
    print("\n--- Court Calibration ---")
    calib_path = "output/calibration.npz"

    if args.calibration and os.path.exists(args.calibration):
        # Load explicitly specified calibration
        calibrator = AutoCourtDetector()
        calibrator.load_calibration(args.calibration)
        print(f"  Loaded calibration from: {args.calibration}")
    elif os.path.exists(calib_path):
        # Auto-load previously saved calibration
        calibrator = AutoCourtDetector()
        calibrator.load_calibration(calib_path)
        print(f"  Loaded saved calibration from: {calib_path}")
    elif args.manual_calibration:
        # Interactive fallback
        print("  Manual mode: click 12 court points. See console guide + mini court diagram.")
        calibrator = ManualCourtCalibrator()
        calibrator.calibrate(first_frame)
        calibrator.save_calibration(calib_path)
    else:
        # Fully automatic (default)
        calibrator = AutoCourtDetector()
        calibrator.detect(first_frame)
        calibrator.save_calibration(calib_path)

    print(f"\n  Homography matrix:\n{calibrator.H}")

    # ──────────────────────────────────────────
    # PHASE 3: INITIALIZE COMPONENTS
    # ──────────────────────────────────────────
    print("\n--- Initializing ---")
    tracker = PlayerTracker()
    minimap_renderer = MinimapRenderer()
    heatmap_gen = HeatmapGenerator()

    os.makedirs(os.path.dirname(args.output) or "output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)

    writer = None
    writer_raw = None

    # Data accumulators
    all_positions = defaultdict(list)
    team_assignments = {}
    # Live display ID mapping: ByteTrack track_id → display label (P1/P2 for Team A, P3/P4 for Team B)
    display_ids = {}          # track_id → display number (1-4)
    _team_counters = {0: 0, 1: 0}  # next slot per team (Team A: 1,2  Team B: 3,4)
    frame_count = start_frame
    processed_count = 0
    clip_frames = end_frame - start_frame
    total_frames = meta["total_frames"]
    processing_times = []
    per_frame_data = []

    # Scene change detection — reference histogram from calibration frame
    ref_hist = compute_frame_histogram(first_frame)
    prev_hist = ref_hist.copy()
    scene_cooldown = 0
    skipped_scenes = 0

    print(f"  Tracker      : YOLOv8 + ByteTrack")
    print(f"  Frame skip   : every {args.skip_frames} frame(s)")
    scene_label = "OFF" if args.no_scene_filter else "ON (skips replays, close-ups, highlights)"
    print(f"  Scene filter : {scene_label}")
    if args.start_time or args.end_time:
        print(f"  Clip         : {args.start_time or '0:00'} - {args.end_time or 'end'} ({clip_frames} frames)")
    else:
        print(f"  Clip         : Full video ({clip_frames} frames)")
    print(f"  Minimap mode : {args.display_mode}")
    print(f"  Output       : {args.output}")
    if args.max_frames > 0:
        print(f"  Max frames   : {args.max_frames}")
    print(f"\n  Processing... (headless, no window)\n")

    # ──────────────────────────────────────────
    # PHASE 4: FRAME-BY-FRAME PROCESSING
    # ──────────────────────────────────────────
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Stop at end time
            if frame_count >= end_frame:
                print(f"\n  Reached end time ({args.end_time}). Stopping.")
                break

            if args.max_frames > 0 and processed_count >= args.max_frames:
                print(f"\n  Reached max frames ({args.max_frames}). Stopping.")
                break

            if frame_count % args.skip_frames != 0:
                frame_count += 1
                continue

            frame_start = time.time()

            # ── Scene change detection ──
            is_gameplay = True
            if not args.no_scene_filter:
                curr_hist = compute_frame_histogram(frame)
                frame_to_frame = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                frame_to_ref = cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_CORREL)

                # Detect camera cuts (sudden histogram change)
                if frame_to_frame < SCENE_CHANGE_THRESHOLD:
                    scene_cooldown = SCENE_CHANGE_COOLDOWN

                # Detect non-gameplay camera angle (doesn't match reference)
                if frame_to_ref < REFERENCE_MATCH_THRESHOLD:
                    scene_cooldown = max(scene_cooldown, SCENE_CHANGE_COOLDOWN)

                is_gameplay = scene_cooldown <= 0

            if not is_gameplay:
                scene_cooldown -= 1
                skipped_scenes += 1
                # Still write the raw frame but with no tracking
                annotated = frame.copy()
                cv2.putText(annotated, "NON-GAMEPLAY", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if writer_raw is None:
                    rh, rw = annotated.shape[:2]
                    raw_path = args.output.replace(".mp4", "_raw.mp4")
                    writer_raw = cv2.VideoWriter(raw_path, fourcc, fps, (rw, rh))
                writer_raw.write(annotated)

                minimap = minimap_renderer.render([])
                combined = minimap_renderer.composite(annotated, minimap, args.display_mode)
                if writer is None:
                    ch, cw = combined.shape[:2]
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (cw, ch))
                writer.write(combined)

                frame_count += 1
                processed_count += 1
                continue

            # Update prev_hist only for gameplay frames
            if not args.no_scene_filter:
                prev_hist = curr_hist.copy()

            # Detect + track players
            players = tracker.update(frame)

            # Homography transform each player + court bounds validation
            frame_players = []
            valid_players = []
            for player in players:
                if player.foot_position is not None:
                    court_xy = calibrator.transform_point(
                        player.foot_position[0], player.foot_position[1]
                    )

                    # Reject players whose court position is outside bounds
                    # (touchline spectators, coaches, ball boys)
                    if not is_on_court(court_xy):
                        continue

                    player.court_position = np.array(court_xy)
                    all_positions[player.track_id].append(player.court_position.copy())
                    tracker.store_court_position(player.track_id, player.court_position)

                    # Assign team by court side: near side = Team A (0), far side = Team B (1)
                    player.team_id = 0 if court_xy[1] < NET_Y else 1
                    team_assignments[player.track_id] = player.team_id

                    # Assign stable display ID: Team A → P1,P2  Team B → P3,P4
                    if player.track_id not in display_ids:
                        team = player.team_id
                        if _team_counters[team] < 2:
                            base = 1 if team == 0 else 3  # Team A: 1,2  Team B: 3,4
                            display_ids[player.track_id] = base + _team_counters[team]
                            _team_counters[team] += 1

                    frame_players.append({
                        "track_id": int(player.track_id),
                        "court_x": float(court_xy[0]),
                        "court_y": float(court_xy[1]),
                        "team_id": int(player.team_id) if player.team_id is not None else -1,
                    })
                    valid_players.append(player)

            # Use only court-validated players for rendering
            players = valid_players

            per_frame_data.append({
                "frame": frame_count,
                "time_s": frame_count / fps if fps > 0 else 0,
                "players": frame_players,
            })

            # Render minimap
            minimap = minimap_renderer.render(players)

            # Annotate video frame
            annotated = frame.copy()
            for player in players:
                color = TEAM_COLORS.get(player.team_id, (255, 255, 255))
                x1, y1, x2, y2 = player.bbox.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                disp = display_ids.get(player.track_id, player.track_id)
                team_letter = "A" if player.team_id == 0 else "B"
                label = f"P{disp} T{team_letter}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                fp = player.foot_position.astype(int)
                cv2.circle(annotated, (fp[0], fp[1]), 4, (0, 255, 0), -1)

            # Write raw annotated video (no minimap)
            if writer_raw is None:
                rh, rw = annotated.shape[:2]
                raw_path = args.output.replace(".mp4", "_raw.mp4")
                writer_raw = cv2.VideoWriter(raw_path, fourcc, fps, (rw, rh))
            writer_raw.write(annotated)

            # Composite video + minimap
            combined = minimap_renderer.composite(annotated, minimap, args.display_mode)

            # FPS + player count overlay
            elapsed = time.time() - frame_start
            processing_times.append(elapsed)
            fps_text = f"FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else "FPS: --"
            cv2.putText(combined, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, f"Players: {len(players)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if writer is None:
                ch, cw = combined.shape[:2]
                writer = cv2.VideoWriter(args.output, fourcc, fps, (cw, ch))
            writer.write(combined)

            processed_count += 1
            frame_count += 1

            # Progress bar
            if processed_count % 100 == 0:
                frames_done = frame_count - start_frame
                pct = (frames_done / clip_frames * 100) if clip_frames > 0 else 0
                avg_fps = 1.0 / np.mean(processing_times[-100:])
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"  [{bar}] {pct:5.1f}%  "
                    f"frame {frames_done}/{clip_frames}  "
                    f"FPS {avg_fps:.1f}  "
                    f"tracks {len(all_positions)}  "
                    f"skipped {skipped_scenes}"
                )

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    # ──────────────────────────────────────────
    # PHASE 5: CLEANUP, SAVE DATA, HEATMAPS
    # ──────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    if writer_raw:
        writer_raw.release()

    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Frames processed : {processed_count}")
    print(f"  Non-gameplay skip: {skipped_scenes} frames")
    if processing_times:
        print(f"  Average FPS      : {1.0 / np.mean(processing_times):.1f}")
    print(f"  Raw tracks       : {len(all_positions)}")
    print(f"  Output video     : {args.output}")

    # ── Post-processing: keep only top 4 tracks by sample count ──
    if len(all_positions) > MAX_PLAYERS:
        sorted_tracks = sorted(all_positions.keys(), key=lambda k: len(all_positions[k]), reverse=True)
        top4 = set(sorted_tracks[:MAX_PLAYERS])
        dropped = set(sorted_tracks[MAX_PLAYERS:])
        print(f"\n  Keeping top {MAX_PLAYERS} tracks: {sorted(top4)}")
        print(f"  Dropping {len(dropped)} minor tracks: {sorted(dropped)}")
        for tid in dropped:
            del all_positions[tid]
            team_assignments.pop(tid, None)
        # Also clean per_frame_data
        for frame_data in per_frame_data:
            frame_data["players"] = [p for p in frame_data["players"] if p["track_id"] in top4]

    # ── Remap track IDs to P1, P2, P3, P4 ──
    # Team A (0) gets P1, P2; Team B (1) gets P3, P4.
    # Within each team, sorted by sample count (most tracked first).
    team_a_ids = sorted(
        [k for k in all_positions if team_assignments.get(k) == 0],
        key=lambda k: len(all_positions[k]), reverse=True,
    )
    team_b_ids = sorted(
        [k for k in all_positions if team_assignments.get(k) == 1],
        key=lambda k: len(all_positions[k]), reverse=True,
    )
    sorted_final_ids = team_a_ids + team_b_ids
    id_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_final_ids, start=1)}
    print(f"  ID remapping     : {' | '.join(f'{old}->P{new}' for old, new in id_remap.items())}")

    remapped_positions = {id_remap[k]: v for k, v in all_positions.items()}
    remapped_teams = {id_remap[k]: v for k, v in team_assignments.items() if k in id_remap}
    for frame_data in per_frame_data:
        for p in frame_data["players"]:
            if p["track_id"] in id_remap:
                p["track_id"] = id_remap[p["track_id"]]
    all_positions = remapped_positions
    team_assignments = remapped_teams

    print(f"  Final players    : {len(all_positions)}")

    # ── Save match data ──
    positions_ser = {int(k): [p.tolist() for p in v] for k, v in all_positions.items()}
    teams_ser = {int(k): int(v) for k, v in team_assignments.items()}

    match_data = {
        "video_meta": meta,
        "player_positions": positions_ser,
        "team_assignments": teams_ser,
        "per_frame_data": per_frame_data,
        "stats": {
            "frames_processed": processed_count,
            "avg_fps": float(1.0 / np.mean(processing_times)) if processing_times else 0,
            "unique_tracks": len(all_positions),
            "duration_s": processed_count / fps if fps > 0 else 0,
        },
        "output_video": os.path.abspath(args.output),
        "output_video_raw": os.path.abspath(args.output.replace(".mp4", "_raw.mp4")),
    }

    with open("output/match_data.pkl", "wb") as f:
        pickle.dump(match_data, f)
    print(f"\n  Match data saved : output/match_data.pkl")

    json_summary = {
        "video_path": meta["path"],
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "frames_processed": processed_count,
        "unique_tracks": len(all_positions),
        "team_assignments": teams_ser,
        "player_position_counts": {int(k): len(v) for k, v in all_positions.items()},
    }
    with open("output/match_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"  Match summary    : output/match_summary.json")

    # ── Generate heatmaps ──
    if all_positions:
        print("\n--- Generating Heatmaps ---")
        saved = heatmap_gen.generate_all_heatmaps(positions_ser, teams_ser, save_dir="output")
        print(f"\n  Heatmaps saved: {len(saved)} files")
        for p in saved:
            print(f"    - {p}")
    else:
        print("\n  No position data. Skipping heatmaps.")

    # ── Auto-launch dashboard ──
    if not args.no_dashboard:
        print("\n" + "=" * 60)
        print("  LAUNCHING DASHBOARD")
        print("=" * 60)
        print("  Opening in browser at http://localhost:8501 ...")
        print("  Press Ctrl+C to stop the dashboard.\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                        "--server.headless", "true"])
    else:
        print("\n  Done! Launch the dashboard manually with:")
        print("    streamlit run dashboard.py")


if __name__ == "__main__":
    main()
