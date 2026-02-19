"""
Padel Court Analytics - Interactive Dashboard.

Launch with:  streamlit run dashboard.py
"""
import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from src.config import (
    COURT_LENGTH, COURT_WIDTH, SERVICE_LINE_DIST, NET_Y, CENTER_SERVICE_X,
    HEATMAP_GRID_RESOLUTION, HEATMAP_ALPHA, HEATMAP_COURT_COLOR,
)
from src.analytics import compute_all_stats

# ────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Padel Analytics Dashboard",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MATCH_DATA_PATH = "output/match_data.pkl"


# ────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────
@st.cache_data
def load_match_data():
    """Load the processed match data from pickle."""
    if not os.path.exists(MATCH_DATA_PATH):
        return None
    with open(MATCH_DATA_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_analytics(player_positions, team_assignments, fps, skip_frames):
    """Compute and cache match analytics."""
    return compute_all_stats(player_positions, team_assignments, fps, skip_frames)


def draw_court_lines(ax):
    """Draw padel court markings on a matplotlib axes."""
    court_rect = Rectangle(
        (0, 0), COURT_WIDTH, COURT_LENGTH,
        linewidth=2, edgecolor="white", facecolor=HEATMAP_COURT_COLOR, zorder=0,
    )
    ax.add_patch(court_rect)
    lkw = dict(color="white", linewidth=2, zorder=1)
    ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], color="lightgray", linewidth=3, zorder=1)
    ax.plot([0, COURT_WIDTH], [SERVICE_LINE_DIST, SERVICE_LINE_DIST], **lkw)
    far_sl = COURT_LENGTH - SERVICE_LINE_DIST
    ax.plot([0, COURT_WIDTH], [far_sl, far_sl], **lkw)
    ax.plot([CENTER_SERVICE_X, CENTER_SERVICE_X], [SERVICE_LINE_DIST, NET_Y], **lkw)
    ax.plot([CENTER_SERVICE_X, CENTER_SERVICE_X], [NET_Y, far_sl], **lkw)


def make_heatmap_figure(positions, title="Heatmap", figsize=(4, 8)):
    """Generate a single heatmap figure from position data."""
    pts = np.array(positions)
    mask = (
        (pts[:, 0] >= -1) & (pts[:, 0] <= COURT_WIDTH + 1)
        & (pts[:, 1] >= -1) & (pts[:, 1] <= COURT_LENGTH + 1)
    )
    pts = pts[mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, COURT_WIDTH + 0.5)
    ax.set_ylim(-0.5, COURT_LENGTH + 0.5)
    ax.set_aspect("equal")
    draw_court_lines(ax)

    if len(pts) >= 5:
        try:
            xy = np.vstack([pts[:, 0], pts[:, 1]])
            kde = gaussian_kde(xy, bw_method="scott")
            xg = np.linspace(0, COURT_WIDTH, HEATMAP_GRID_RESOLUTION)
            yg = np.linspace(0, COURT_LENGTH, HEATMAP_GRID_RESOLUTION)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.pcolormesh(X, Y, Z, cmap="YlOrRd", alpha=HEATMAP_ALPHA, shading="auto")
        except np.linalg.LinAlgError:
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=5, c="red")
    else:
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=10, c="red")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Court Width (m)")
    ax.set_ylabel("Court Length (m)")
    plt.tight_layout()
    return fig


# ────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric > div { background: #1e1e2e; border-radius: 10px; padding: 15px; }
    .stMetric label { color: #a0a0b0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }

    /* ── Push tabs down so they're not clipped ── */
    .stTabs {
        margin-top: 1.5rem !important;
    }
    /* ── Tab buttons: force visible text ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 28px !important;
        min-height: 48px !important;
    }
    /* Target ALL child elements of tab buttons */
    .stTabs [data-baseweb="tab-list"] button div,
    .stTabs [data-baseweb="tab-list"] button span,
    .stTabs [data-baseweb="tab-list"] button p,
    .stTabs [data-baseweb="tab-list"] button label,
    .stTabs [data-baseweb="tab-list"] button * {
        color: #cccccc !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1a2744 !important;
        border-bottom: 3px solid #4a9eff !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] div,
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] span,
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p,
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] label,
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] * {
        color: #4a9eff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    /* Override Streamlit tab highlight bar */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4a9eff !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: #3a3a4a !important;
    }

    /* ── Sidebar controls ── */
    .stMultiSelect span, .stMultiSelect label,
    .stSelectbox label, .stSelectbox span { color: #ffffff !important; }

    .team-a { color: #4a9eff; font-weight: bold; }
    .team-b { color: #ff5a5a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────────
def main():
    data = load_match_data()

    if data is None:
        st.title("🎾 Padel Court Analytics")
        st.error(
            "**No match data found.**  \n"
            "Run the processing pipeline first:  \n"
            "```\npython main.py\n```\n"
            "Then relaunch the dashboard."
        )
        return

    player_positions = data["player_positions"]
    team_assignments = data["team_assignments"]
    meta = data["video_meta"]
    stats = data["stats"]
    fps = meta.get("fps", 30)

    analytics = get_analytics(
        player_positions, team_assignments, fps, skip_frames=1
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/tennis-racquet.png", width=60)
        st.title("Padel Analytics")
        st.markdown("---")
        st.markdown(f"**Video FPS:** {fps:.0f}")
        st.markdown(f"**Duration:** {stats['duration_s']:.1f}s")
        st.markdown(f"**Frames processed:** {stats['frames_processed']}")
        st.markdown(f"**Unique tracks:** {stats['unique_tracks']}")
        st.markdown("---")

        # Player selector for filtering
        all_track_ids = sorted(player_positions.keys(), key=int)
        player_labels = []
        for tid in all_track_ids:
            team = team_assignments.get(int(tid), team_assignments.get(str(tid), -1))
            t = "A" if team == 0 else "B" if team == 1 else "?"
            player_labels.append(f"P{tid} (Team {t})")

        st.markdown("**Filter Players:**")
        selected_labels = st.multiselect(
            "Select players to show",
            player_labels,
            default=player_labels,
            label_visibility="collapsed",
        )
        selected_ids = [
            all_track_ids[i] for i, lbl in enumerate(player_labels) if lbl in selected_labels
        ]

    # ─────── TABS ───────
    tab_replay, tab_heatmaps, tab_analytics = st.tabs([
        "📹  Match Replay",
        "🔥  Player Heatmaps",
        "📊  Full Match Analytics",
    ])

    # ────────────────────────────────────────
    # TAB 1: MATCH REPLAY
    # ────────────────────────────────────────
    with tab_replay:
        st.header("Match Replay with Live Minimap")

        video_path = data.get("output_video", "output/tracked_output.mp4")
        raw_video_path = data.get("output_video_raw", "output/tracked_output_raw.mp4")

        col_vid, col_mini = st.columns([3, 1])

        with col_vid:
            # Read video as bytes for reliable browser playback
            vid_file = None
            if os.path.exists(video_path):
                vid_file = video_path
            elif os.path.exists(raw_video_path):
                vid_file = raw_video_path

            if vid_file:
                with open(vid_file, "rb") as vf:
                    video_bytes = vf.read()
                st.video(video_bytes)
            else:
                st.warning("Output video not found. Run `python main.py` first.")

        with col_mini:
            st.markdown("#### Court Minimap")
            # Show a static heatmap of all players as a summary minimap
            all_pts = []
            for tid in selected_ids:
                all_pts.extend(player_positions[tid])
            if all_pts:
                fig = make_heatmap_figure(all_pts, "All Players", figsize=(3, 6))
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No position data for selected players.")

        # Quick stats row below video
        st.markdown("---")
        st.subheader("Quick Stats")
        cols = st.columns(4)
        for i, tid in enumerate(selected_ids[:4]):
            tid_key = tid if tid in analytics["players"] else int(tid)
            if tid_key not in analytics["players"]:
                continue
            ps = analytics["players"][tid_key]
            team_label = ps["team"]
            with cols[i % 4]:
                team_color = "team-a" if team_label == "A" else "team-b"
                st.markdown(
                    f'<span class="{team_color}">P{tid} - Team {team_label}</span>',
                    unsafe_allow_html=True,
                )
                st.metric("Distance", f"{ps['total_distance_m']:.0f} m")
                st.metric("Avg Speed", f"{ps['avg_speed_kmh']:.1f} km/h")
                st.metric("Coverage", f"{ps['court_coverage_pct']:.0f}%")

    # ────────────────────────────────────────
    # TAB 2: PLAYER HEATMAPS
    # ────────────────────────────────────────
    with tab_heatmaps:
        st.header("Player Position Heatmaps")
        st.markdown("Gaussian KDE density maps showing where each player spent the most time on court.")

        # Individual player heatmaps
        num_selected = len(selected_ids)
        if num_selected == 0:
            st.info("Select players from the sidebar to view heatmaps.")
        else:
            cols_per_row = min(num_selected, 4)
            cols = st.columns(cols_per_row)
            for i, tid in enumerate(selected_ids):
                positions = player_positions[tid]
                team = team_assignments.get(int(tid), team_assignments.get(str(tid), -1))
                t = "A" if team == 0 else "B" if team == 1 else "?"
                with cols[i % cols_per_row]:
                    if len(positions) >= 5:
                        fig = make_heatmap_figure(
                            positions, f"P{tid} (Team {t})", figsize=(4, 8)
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning(f"P{tid}: insufficient data ({len(positions)} pts)")

            # Team heatmaps
            st.markdown("---")
            st.subheader("Team Heatmaps")
            team_col_a, team_col_b = st.columns(2)

            for team_id, team_name, col in [(0, "A", team_col_a), (1, "B", team_col_b)]:
                team_pts = []
                for tid in selected_ids:
                    t = team_assignments.get(int(tid), team_assignments.get(str(tid), -1))
                    if t == team_id:
                        team_pts.extend(player_positions[tid])
                with col:
                    if len(team_pts) >= 5:
                        fig = make_heatmap_figure(
                            team_pts, f"Team {team_name} (Combined)", figsize=(4, 8)
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info(f"Team {team_name}: insufficient data")

    # ────────────────────────────────────────
    # TAB 3: FULL MATCH ANALYTICS
    # ────────────────────────────────────────
    with tab_analytics:
        st.header("Full Match Analytics")

        # ── Summary cards ──
        st.subheader("Match Overview")
        ov_cols = st.columns(4)
        with ov_cols[0]:
            st.metric("Match Duration", f"{stats['duration_s']:.0f}s")
        with ov_cols[1]:
            st.metric("Frames Processed", f"{stats['frames_processed']:,}")
        with ov_cols[2]:
            st.metric("Processing FPS", f"{stats['avg_fps']:.1f}")
        with ov_cols[3]:
            st.metric("Players Tracked", f"{stats['unique_tracks']}")

        st.markdown("---")

        # ── Per-player stats table ──
        st.subheader("Player Statistics")

        table_data = []
        for tid in selected_ids:
            tid_key = tid if tid in analytics["players"] else int(tid)
            if tid_key not in analytics["players"]:
                continue
            ps = analytics["players"][tid_key]
            table_data.append({
                "Player": f"P{tid}",
                "Team": ps["team"],
                "Distance (m)": ps["total_distance_m"],
                "Avg Speed (km/h)": ps["avg_speed_kmh"],
                "Court Coverage (%)": ps["court_coverage_pct"],
                "Tracking Time (s)": ps["tracking_duration_s"],
                "Samples": ps["total_samples"],
            })

        if table_data:
            st.dataframe(
                table_data,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No player stats to display.")

        st.markdown("---")

        # ── Team colors for Altair charts ──
        TEAM_A_COLOR = "#4a9eff"
        TEAM_B_COLOR = "#ff5a5a"

        # ── Distance comparison bar chart (Altair) ──
        st.subheader("Distance Covered Comparison")
        if table_data:
            df_dist = pd.DataFrame(table_data)
            df_dist["Team Label"] = "Team " + df_dist["Team"]

            bars = alt.Chart(df_dist).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("Player:N", sort=df_dist["Player"].tolist(),
                         axis=alt.Axis(labelFontSize=12)),
                y=alt.Y("Distance (m):Q", axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
                color=alt.Color("Team Label:N",
                                scale=alt.Scale(domain=["Team A", "Team B"],
                                                range=[TEAM_A_COLOR, TEAM_B_COLOR]),
                                legend=alt.Legend(title=None, orient="top-right",
                                                  labelFontSize=10)),
                tooltip=["Player", "Team", "Distance (m)"],
            )
            text = alt.Chart(df_dist).mark_text(
                dy=-8, fontSize=12, fontWeight="bold", color="white"
            ).encode(
                x=alt.X("Player:N", sort=df_dist["Player"].tolist()),
                y=alt.Y("Distance (m):Q"),
                text=alt.Text("Distance (m):Q", format=".0f"),
            )
            st.altair_chart(
                (bars + text).properties(height=350).configure_view(strokeWidth=0),
                use_container_width=True,
            )

        st.markdown("---")

        # ── Zone distribution (Altair donut charts) ──
        st.subheader("Court Zone Distribution")
        st.markdown("Percentage of time each player spent in different court zones.")

        zone_palette = ["#2ecc71", "#f1c40f", "#e74c3c"]
        zone_order = ["Back Court", "Mid Court", "Net Zone"]

        if selected_ids:
            zone_cols = st.columns(4)
            for i, tid in enumerate(selected_ids[:4]):
                tid_key = tid if tid in analytics["players"] else int(tid)
                if tid_key not in analytics["players"]:
                    continue
                ps = analytics["players"][tid_key]
                zones = ps["zone_distribution"]
                team_label = ps["team"]
                tc = TEAM_A_COLOR if team_label == "A" else TEAM_B_COLOR

                with zone_cols[i]:
                    st.markdown(
                        f'<p style="text-align:center;color:{tc};font-weight:bold;'
                        f'margin-bottom:0;font-size:14px">P{tid} (Team {team_label})</p>',
                        unsafe_allow_html=True,
                    )
                    df_zone = pd.DataFrame([
                        {"Zone": z, "Pct": zones.get(z, 0)} for z in zone_order
                    ])
                    donut = alt.Chart(df_zone).mark_arc(
                        innerRadius=45, outerRadius=85, stroke="#0e1117", strokeWidth=2
                    ).encode(
                        theta=alt.Theta("Pct:Q", stack=True),
                        color=alt.Color("Zone:N",
                                        scale=alt.Scale(domain=zone_order, range=zone_palette),
                                        legend=None),
                        tooltip=["Zone", alt.Tooltip("Pct:Q", format=".1f", title="%")],
                    )
                    pct_text = alt.Chart(df_zone).transform_calculate(
                        label="datum.Pct > 5 ? format(datum.Pct, '.0f') + '%' : ''"
                    ).mark_text(
                        radius=68, fontSize=12, fontWeight="bold", color="white"
                    ).encode(
                        theta=alt.Theta("Pct:Q", stack=True),
                        text="label:N",
                    )
                    st.altair_chart(
                        (donut + pct_text).properties(height=200).configure_view(strokeWidth=0),
                        use_container_width=True,
                    )

            # HTML legend — small, clean, always visible
            legend_html = '<div style="text-align:center;margin-top:4px">'
            for z, c in zip(zone_order, zone_palette):
                legend_html += (
                    f'<span style="display:inline-block;margin:0 12px">'
                    f'<span style="display:inline-block;width:10px;height:10px;'
                    f'background:{c};border-radius:2px;margin-right:5px"></span>'
                    f'<span style="color:#ccc;font-size:12px">{z}</span></span>'
                )
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("---")

        # ── Side preference (Altair grouped bar) ──
        st.subheader("Left / Right Court Preference")

        if selected_ids:
            side_rows = []
            for tid in selected_ids:
                tid_key = tid if tid in analytics["players"] else int(tid)
                if tid_key not in analytics["players"]:
                    continue
                ps = analytics["players"][tid_key]
                side_rows.append({"Player": f"P{tid}", "Side": "Left",
                                  "Time (%)": ps["side_preference"]["left_pct"]})
                side_rows.append({"Player": f"P{tid}", "Side": "Right",
                                  "Time (%)": ps["side_preference"]["right_pct"]})

            if side_rows:
                df_side = pd.DataFrame(side_rows)
                player_order = [f"P{tid}" for tid in selected_ids]

                bars_side = alt.Chart(df_side).mark_bar(
                    cornerRadiusTopLeft=3, cornerRadiusTopRight=3
                ).encode(
                    x=alt.X("Player:N", sort=player_order,
                             axis=alt.Axis(labelFontSize=12)),
                    xOffset="Side:N",
                    y=alt.Y("Time (%):Q",
                             axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
                    color=alt.Color("Side:N",
                                    scale=alt.Scale(domain=["Left", "Right"],
                                                    range=["#3498db", "#e67e22"]),
                                    legend=alt.Legend(title=None, orient="top-right",
                                                      labelFontSize=10)),
                    tooltip=["Player", "Side", alt.Tooltip("Time (%):Q", format=".1f")],
                )
                text_side = alt.Chart(df_side).mark_text(
                    dy=-8, fontSize=10, color="white"
                ).encode(
                    x=alt.X("Player:N", sort=player_order),
                    xOffset="Side:N",
                    y=alt.Y("Time (%):Q"),
                    text=alt.Text("Time (%):Q", format=".0f"),
                )
                st.altair_chart(
                    (bars_side + text_side).properties(height=350).configure_view(strokeWidth=0),
                    use_container_width=True,
                )

        st.markdown("---")

        # ── Team comparison ──
        st.subheader("Team Comparison")
        team_stats = analytics.get("teams", {})

        if team_stats:
            tc1, tc2 = st.columns(2)
            for team_name, col in [("A", tc1), ("B", tc2)]:
                ts = team_stats.get(team_name, {})
                with col:
                    tc = TEAM_A_COLOR if team_name == "A" else TEAM_B_COLOR
                    st.markdown(
                        f'<h3 style="color:{tc}; margin-bottom: 0.5rem;">Team {team_name}</h3>',
                        unsafe_allow_html=True,
                    )
                    if ts:
                        st.metric("Total Distance", f"{ts['total_distance_m']:.0f} m")
                        st.metric("Avg Court Coverage", f"{ts['avg_coverage_pct']:.0f}%")
                        st.metric("Players", f"{ts['player_count']}")
                    else:
                        st.info("No data for this team.")
        else:
            st.info("Team comparison data not available.")


if __name__ == "__main__":
    main()
