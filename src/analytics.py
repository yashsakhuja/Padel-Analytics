"""
Match analytics computed from player position data.
Provides stats like distance covered, zone distribution, court coverage, speed.
"""
import numpy as np
from typing import Dict, List, Tuple

from src.config import COURT_LENGTH, COURT_WIDTH, SERVICE_LINE_DIST, NET_Y


# Court zones per half (y-ranges in meters)
# Team A plays on the near side (0–10m), Team B on the far side (10–20m).
# Net zone = ~2m either side of the net (8–12m).
NET_ZONE_DEPTH = 2.0  # meters from net on each side
ZONES_NEAR = {
    "Back Court": (0.0, SERVICE_LINE_DIST),                          # 0 – 3.05m  (glass wall)
    "Mid Court":  (SERVICE_LINE_DIST, NET_Y - NET_ZONE_DEPTH),       # 3.05 – 8m  (transition)
    "Net Zone":   (NET_Y - NET_ZONE_DEPTH, NET_Y + NET_ZONE_DEPTH),  # 8 – 12m    (at the net)
}
ZONES_FAR = {
    "Net Zone":   (NET_Y - NET_ZONE_DEPTH, NET_Y + NET_ZONE_DEPTH),  # 8 – 12m    (at the net)
    "Mid Court":  (NET_Y + NET_ZONE_DEPTH, COURT_LENGTH - SERVICE_LINE_DIST),  # 12 – 16.95m (transition)
    "Back Court": (COURT_LENGTH - SERVICE_LINE_DIST, COURT_LENGTH),   # 16.95 – 20m (glass wall)
}
ZONE_NAMES = ["Back Court", "Mid Court", "Net Zone"]


def compute_player_stats(
    positions: List[list],
    fps: float = 30.0,
    skip_frames: int = 1,
    team_id: int = -1,
) -> Dict:
    """
    Compute analytics for a single player from their court position history.

    Args:
        positions: List of [x, y] court positions in meters.
        fps: Video frames per second.
        skip_frames: How many frames were skipped between samples.

    Returns:
        Dict with computed statistics.
    """
    if len(positions) < 2:
        return _empty_stats()

    pts = np.array(positions)

    # Filter out-of-bounds noise
    mask = (
        (pts[:, 0] >= -1)
        & (pts[:, 0] <= COURT_WIDTH + 1)
        & (pts[:, 1] >= -1)
        & (pts[:, 1] <= COURT_LENGTH + 1)
    )
    pts = pts[mask]

    if len(pts) < 2:
        return _empty_stats()

    # Time between position samples
    dt = skip_frames / fps if fps > 0 else 1.0 / 30.0

    # Distances between consecutive positions
    diffs = np.diff(pts, axis=0)
    frame_distances = np.linalg.norm(diffs, axis=1)

    # Filter out teleport glitches (>5m in one frame = noise)
    valid_dists = frame_distances[frame_distances < 5.0]

    total_distance = float(np.sum(valid_dists))
    avg_speed = total_distance / (len(pts) * dt) if len(pts) > 0 else 0.0
    max_speed = float(np.max(valid_dists) / dt) if len(valid_dists) > 0 else 0.0

    # Average position (center of activity)
    avg_x = float(np.mean(pts[:, 0]))
    avg_y = float(np.mean(pts[:, 1]))

    # Zone distribution — pick zones based on which half the player's team plays on
    # Team A (0) = near side, Team B (1) = far side
    zones = ZONES_FAR if team_id == 1 else ZONES_NEAR
    zone_counts = {}
    for zone_name, (y_min, y_max) in zones.items():
        in_zone = np.sum((pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
        zone_counts[zone_name] = int(in_zone)
    total_in_zones = sum(zone_counts.values())
    zone_pct = {}
    for zone_name in ZONE_NAMES:
        count = zone_counts.get(zone_name, 0)
        zone_pct[zone_name] = round(count / total_in_zones * 100, 1) if total_in_zones > 0 else 0.0

    # Court coverage: divide court into grid cells, count unique cells visited
    grid_size = 1.0  # 1m x 1m cells
    grid_x = np.clip((pts[:, 0] / grid_size).astype(int), 0, int(COURT_WIDTH / grid_size) - 1)
    grid_y = np.clip((pts[:, 1] / grid_size).astype(int), 0, int(COURT_LENGTH / grid_size) - 1)
    unique_cells = len(set(zip(grid_x.tolist(), grid_y.tolist())))
    total_cells = int(COURT_WIDTH / grid_size) * int(COURT_LENGTH / grid_size)
    coverage_pct = round(unique_cells / total_cells * 100, 1) if total_cells > 0 else 0.0

    # Left vs right preference
    left_count = int(np.sum(pts[:, 0] < COURT_WIDTH / 2))
    right_count = int(np.sum(pts[:, 0] >= COURT_WIDTH / 2))
    total_lr = left_count + right_count
    left_pct = round(left_count / total_lr * 100, 1) if total_lr > 0 else 50.0
    right_pct = round(right_count / total_lr * 100, 1) if total_lr > 0 else 50.0

    return {
        "total_distance_m": round(total_distance, 1),
        "avg_speed_ms": round(avg_speed, 2),
        "max_speed_ms": round(max_speed, 2),
        "avg_speed_kmh": round(avg_speed * 3.6, 1),
        "max_speed_kmh": round(max_speed * 3.6, 1),
        "avg_position": {"x": round(avg_x, 2), "y": round(avg_y, 2)},
        "zone_distribution": zone_pct,
        "zone_counts": zone_counts,
        "court_coverage_pct": coverage_pct,
        "unique_cells_visited": unique_cells,
        "total_cells": total_cells,
        "side_preference": {"left_pct": left_pct, "right_pct": right_pct},
        "total_samples": len(pts),
        "tracking_duration_s": round(len(pts) * dt, 1),
    }


def _empty_stats() -> Dict:
    """Return empty stats dict when insufficient data."""
    return {
        "total_distance_m": 0.0,
        "avg_speed_ms": 0.0,
        "max_speed_ms": 0.0,
        "avg_speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "avg_position": {"x": 0.0, "y": 0.0},
        "zone_distribution": {z: 0.0 for z in ZONE_NAMES},
        "zone_counts": {z: 0 for z in ZONE_NAMES},
        "court_coverage_pct": 0.0,
        "unique_cells_visited": 0,
        "total_cells": 0,
        "side_preference": {"left_pct": 50.0, "right_pct": 50.0},
        "total_samples": 0,
        "tracking_duration_s": 0.0,
    }


def compute_all_stats(
    player_positions: Dict[int, List[list]],
    team_assignments: Dict[int, int],
    fps: float = 30.0,
    skip_frames: int = 1,
) -> Dict:
    """
    Compute analytics for all players.

    Returns:
        Dict with per-player stats and team aggregates.
    """
    player_stats = {}
    for tid, positions in player_positions.items():
        team_id = team_assignments.get(tid, -1)
        stats = compute_player_stats(positions, fps, skip_frames, team_id=team_id)
        team_label = "A" if team_id == 0 else "B" if team_id == 1 else "?"
        stats["team"] = team_label
        stats["team_id"] = team_id
        player_stats[tid] = stats

    # Team aggregates
    team_stats = {}
    for team_label in ["A", "B"]:
        team_players = {
            tid: s for tid, s in player_stats.items() if s["team"] == team_label
        }
        if team_players:
            total_dist = sum(s["total_distance_m"] for s in team_players.values())
            avg_coverage = np.mean(
                [s["court_coverage_pct"] for s in team_players.values()]
            )
            team_stats[team_label] = {
                "total_distance_m": round(total_dist, 1),
                "avg_coverage_pct": round(float(avg_coverage), 1),
                "player_count": len(team_players),
                "player_ids": list(team_players.keys()),
            }

    return {
        "players": player_stats,
        "teams": team_stats,
    }
