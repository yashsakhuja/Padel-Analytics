"""
Post-match heatmap generation using Gaussian KDE.
Classic warm heatmap (yellow → orange → red) on a blue padel court.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional

from src.config import (
    COURT_LENGTH,
    COURT_WIDTH,
    SERVICE_LINE_DIST,
    NET_Y,
    CENTER_SERVICE_X,
    HEATMAP_GRID_RESOLUTION,
    HEATMAP_ALPHA,
    HEATMAP_COURT_COLOR,
)

# Classic warm heatmap colormap — same as reference image
HEATMAP_CMAP = "YlOrRd"


class HeatmapGenerator:
    """Generates 2D kernel density estimation heatmaps for player positions."""

    def generate_player_heatmap(
        self,
        positions: List[np.ndarray],
        player_label: str = "Player",
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Generate a heatmap for a single player's court positions.
        Returns the matplotlib Figure or None if insufficient data.
        """
        if len(positions) < 5:
            print(f"  Skipping heatmap for {player_label}: only {len(positions)} points")
            return None

        pts = np.array(positions)
        mask = (
            (pts[:, 0] >= -1)
            & (pts[:, 0] <= COURT_WIDTH + 1)
            & (pts[:, 1] >= -1)
            & (pts[:, 1] <= COURT_LENGTH + 1)
        )
        pts = pts[mask]

        if len(pts) < 5:
            print(f"  Skipping heatmap for {player_label}: too few in-bounds points")
            return None

        fig, ax = plt.subplots(figsize=(5, 10))
        ax.set_xlim(-0.5, COURT_WIDTH + 0.5)
        ax.set_ylim(-0.5, COURT_LENGTH + 0.5)
        ax.set_aspect("equal")

        self._draw_court_lines(ax)
        self._render_kde(ax, pts)

        ax.set_title(player_label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Court Width (m)")
        ax.set_ylabel("Court Length (m)")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")

        plt.close(fig)
        return fig

    def generate_all_heatmaps(
        self,
        player_positions: Dict[int, List[np.ndarray]],
        team_assignments: Dict[int, int],
        save_dir: str = "output",
    ) -> List[str]:
        """
        Generate heatmaps for each player and each team.
        Returns list of saved file paths.
        """
        os.makedirs(save_dir, exist_ok=True)
        saved = []

        track_ids = sorted(player_positions.keys())
        print(f"\nGenerating heatmaps for {len(track_ids)} tracked players...")

        # Per-player heatmaps
        for tid in track_ids:
            positions = player_positions[tid]
            team = team_assignments.get(tid, -1)
            label = f"P{tid} (Team {'A' if team == 0 else 'B' if team == 1 else '?'})"
            path = os.path.join(save_dir, f"heatmap_player_{tid}.png")
            fig = self.generate_player_heatmap(positions, label, path)
            if fig is not None:
                saved.append(path)

        # Per-team heatmaps
        for team_id, team_name in [(0, "A"), (1, "B")]:
            team_positions = []
            for tid in track_ids:
                if team_assignments.get(tid) == team_id:
                    team_positions.extend(player_positions[tid])

            if team_positions:
                label = f"Team {team_name} (Combined)"
                path = os.path.join(save_dir, f"heatmap_team_{team_name}.png")
                fig = self.generate_player_heatmap(team_positions, label, path)
                if fig is not None:
                    saved.append(path)

        # Combined 2x2 grid if we have exactly 4 players
        if len(track_ids) >= 4:
            self._generate_combined_grid(
                player_positions, team_assignments, track_ids[:4], save_dir
            )
            saved.append(os.path.join(save_dir, "heatmap_combined.png"))

        return saved

    def _generate_combined_grid(
        self,
        player_positions: Dict[int, List[np.ndarray]],
        team_assignments: Dict[int, int],
        track_ids: List[int],
        save_dir: str,
    ):
        """Generate a 2x2 grid showing all 4 players."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 20))
        axes_flat = axes.flatten()

        for i, tid in enumerate(track_ids[:4]):
            ax = axes_flat[i]
            positions = player_positions[tid]
            team = team_assignments.get(tid, -1)
            team_label = "A" if team == 0 else "B" if team == 1 else "?"

            ax.set_xlim(-0.5, COURT_WIDTH + 0.5)
            ax.set_ylim(-0.5, COURT_LENGTH + 0.5)
            ax.set_aspect("equal")
            self._draw_court_lines(ax)

            pts = np.array(positions)
            if len(pts) < 5:
                ax.set_title(f"P{tid} (Team {team_label}) - No data")
                continue

            mask = (
                (pts[:, 0] >= -1)
                & (pts[:, 0] <= COURT_WIDTH + 1)
                & (pts[:, 1] >= -1)
                & (pts[:, 1] <= COURT_LENGTH + 1)
            )
            pts = pts[mask]

            if len(pts) >= 5:
                self._render_kde(ax, pts)

            ax.set_title(f"P{tid} (Team {team_label})", fontsize=12)

        plt.suptitle("All Players - Court Heatmaps", fontsize=16, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, "heatmap_combined.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    def _render_kde(self, ax: plt.Axes, pts: np.ndarray) -> None:
        """Render KDE heatmap with classic YlOrRd colormap."""
        try:
            xy = np.vstack([pts[:, 0], pts[:, 1]])
            kde = gaussian_kde(xy, bw_method="scott")

            x_grid = np.linspace(0, COURT_WIDTH, HEATMAP_GRID_RESOLUTION)
            y_grid = np.linspace(0, COURT_LENGTH, HEATMAP_GRID_RESOLUTION)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_coords = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(grid_coords).reshape(X.shape)

            ax.pcolormesh(X, Y, Z, cmap=HEATMAP_CMAP, alpha=HEATMAP_ALPHA, shading="auto")
        except np.linalg.LinAlgError:
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=5, c="red")

    def _draw_court_lines(self, ax: plt.Axes) -> None:
        """Draw padel court markings on a matplotlib axes with blue background."""
        line_kw = dict(color="white", linewidth=2, zorder=1)

        court_rect = Rectangle(
            (0, 0),
            COURT_WIDTH,
            COURT_LENGTH,
            linewidth=2,
            edgecolor="white",
            facecolor=HEATMAP_COURT_COLOR,
            zorder=0,
        )
        ax.add_patch(court_rect)

        # Net
        ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], color="lightgray", linewidth=3, zorder=1)

        # Near service line
        ax.plot([0, COURT_WIDTH], [SERVICE_LINE_DIST, SERVICE_LINE_DIST], **line_kw)

        # Far service line
        far_sl = COURT_LENGTH - SERVICE_LINE_DIST
        ax.plot([0, COURT_WIDTH], [far_sl, far_sl], **line_kw)

        # Near center service line
        ax.plot([CENTER_SERVICE_X, CENTER_SERVICE_X], [SERVICE_LINE_DIST, NET_Y], **line_kw)

        # Far center service line
        ax.plot([CENTER_SERVICE_X, CENTER_SERVICE_X], [NET_Y, far_sl], **line_kw)
