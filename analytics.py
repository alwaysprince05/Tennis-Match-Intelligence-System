"""
analytics.py — Tennis Match Intelligence System
Developer/Creator: tubakhxn

Generates broadcast-quality tactical heatmaps and the match dashboard PNG,
including court coverage maps, movement efficiency, aggression/defensive scores,
activity analysis, and match intensity visualisation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image
import logging

log = logging.getLogger("TMIS.Analytics")

# ─────────────────────────── Colour palette (matplotlib RGB 0-1) ─────────────
PLAYER_COLORS_MPL = {
    0: {"mpl": (0.0,  1.0,  1.0),  "name": "Cyan"},
    1: {"mpl": (1.0,  0.0,  1.0),  "name": "Magenta"},
    2: {"mpl": (0.0,  1.0,  0.0),  "name": "Green"},
    3: {"mpl": (1.0,  0.65, 0.0),  "name": "Orange"},
    4: {"mpl": (1.0,  1.0,  0.0),  "name": "Yellow"},
    5: {"mpl": (0.55, 0.0,  1.0),  "name": "Purple"},
}

COURT_W, COURT_H = 1344, 576

# ─────────────────────────── Custom colormaps ────────────────────────────────
def _neon_cmap(base_rgb):
    """Build a transparent → neon colourmap for a given player colour."""
    r, g, b = base_rgb
    colors   = [
        (r*0.0, g*0.0, b*0.0, 0.0),
        (r*0.4, g*0.4, b*0.4, 0.4),
        (r,     g,     b,     0.7),
        (1.0,   1.0,   1.0,   0.9),
    ]
    return LinearSegmentedColormap.from_list("neon", colors)


_INTENSITY_CMAP = LinearSegmentedColormap.from_list(
    "intensity",
    [(0,0,0,0), (0,0.5,1,0.5), (0,1,1,0.7), (1,1,0,0.85), (1,0,0,1)]
)

# ─────────────────────────── Court drawing helpers ───────────────────────────
def _draw_court_lines(ax, width=COURT_W, height=COURT_H, lw=1.5, color="white", alpha=0.6):
    """Draw a top-down tennis court diagram onto a matplotlib axis."""
    kw = dict(color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    W, H = width, height

    # Court fill
    ax.add_patch(mpatches.Rectangle((0, 0), W, H,
                                    facecolor=(0.18, 0.38, 0.22), zorder=0))

    # Outer boundary
    ax.add_patch(mpatches.Rectangle((0, 0), W, H,
                                    edgecolor=color, facecolor="none",
                                    lw=lw+1, alpha=alpha, zorder=1))

    # Net
    ax.plot([W/2, W/2], [0, H], **kw, lw=lw+2, zorder=2)

    # Singles sidelines (inner)
    margin_y = H * 0.13
    ax.plot([0, W], [margin_y, margin_y],         **kw, zorder=1)
    ax.plot([0, W], [H-margin_y, H-margin_y],     **kw, zorder=1)

    # Service boxes
    service_x_left  = W * 0.25
    service_x_right = W * 0.75
    ax.plot([service_x_left,  service_x_left],  [margin_y, H-margin_y], **kw, zorder=1)
    ax.plot([service_x_right, service_x_right], [margin_y, H-margin_y], **kw, zorder=1)

    # Centre service line
    ax.plot([W/2, W/2], [margin_y, H-margin_y], **kw, lw=lw-0.5, zorder=1)

    # Baselines
    ax.plot([0, 0], [0, H], **kw, lw=lw+1, zorder=1)
    ax.plot([W, W], [0, H], **kw, lw=lw+1, zorder=1)


# ─────────────────────────── Heatmap generator ───────────────────────────────
class TennisAnalytics:

    def generate_heatmap(self, player_stats: dict, output_path: str):
        """
        Renders a professional multi-panel court heatmap PNG.
        """
        n_players = len(player_stats)
        if n_players == 0:
            log.warning("No player data — skipping heatmap.")
            return

        cols = min(n_players, 3)
        rows = (n_players + cols - 1) // cols + 1   # +1 for combined row

        fig = plt.figure(figsize=(cols * 7, rows * 4.5), facecolor="#080C10")
        fig.suptitle(
            "COURT HEATMAP  ·  Tennis Match Intelligence System  ·  tubakhxn",
            fontsize=14, color="white", fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(rows, cols, figure=fig,
                               hspace=0.45, wspace=0.25,
                               left=0.04, right=0.96,
                               top=0.93, bottom=0.04)

        # ── Individual player heatmaps ──
        axes_list = []
        for idx, (tid, st) in enumerate(player_stats.items()):
            r, c  = divmod(idx, cols)
            ax    = fig.add_subplot(gs[r, c])
            ci    = idx % len(PLAYER_COLORS_MPL)
            cinfo = PLAYER_COLORS_MPL[ci]
            cmap  = _neon_cmap(cinfo["mpl"])

            hm = st["heatmap"].copy()
            if hm.max() > 0:
                hm = gaussian_filter(hm, sigma=18)
                hm /= hm.max()

            _draw_court_lines(ax, zorder=0)
            ax.imshow(hm, origin="lower", extent=[0, COURT_W, 0, COURT_H],
                      cmap=cmap, alpha=0.88, aspect="auto", zorder=3)

            # Coverage %
            coverage = float(np.mean(hm > 0.05) * 100)
            speeds   = list(st["speeds"])
            avg_spd  = float(np.mean(speeds)) if speeds else 0.0
            dist     = st["dist"]
            agg      = min(100, int(avg_spd * 3.5))

            ax.set_xlim(0, COURT_W)
            ax.set_ylim(0, COURT_H)
            ax.set_aspect("auto")
            ax.set_facecolor("#080C10")
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(cinfo["mpl"])
                spine.set_linewidth(1.5)

            ax.set_title(
                f"Player {tid+1}  ({cinfo['name']})\n"
                f"Dist: {dist:.1f}m  │  Avg: {avg_spd:.1f} km/h  │  "
                f"Coverage: {coverage:.1f}%  │  Aggression: {agg}",
                fontsize=9, color=cinfo["mpl"], pad=5
            )
            axes_list.append(ax)

        # ── Combined heatmap ──
        combined_r = rows - 1
        ax_comb = fig.add_subplot(gs[combined_r, :])
        combined = np.zeros((COURT_H, COURT_W), dtype=np.float32)
        for st in player_stats.values():
            hm = st["heatmap"].copy()
            if hm.max() > 0:
                combined += gaussian_filter(hm / hm.max(), sigma=18)

        if combined.max() > 0:
            combined /= combined.max()

        _draw_court_lines(ax_comb, zorder=0)
        ax_comb.imshow(combined, origin="lower", extent=[0, COURT_W, 0, COURT_H],
                       cmap=_INTENSITY_CMAP, alpha=0.88, aspect="auto", zorder=3)

        ax_comb.set_xlim(0, COURT_W)
        ax_comb.set_ylim(0, COURT_H)
        ax_comb.set_aspect("auto")
        ax_comb.set_facecolor("#080C10")
        ax_comb.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in ax_comb.spines.values():
            spine.set_edgecolor("#00CFFF")
            spine.set_linewidth(2)
        ax_comb.set_title("COMBINED COURT COVERAGE  —  All Players",
                          fontsize=10, color="#00CFFF", pad=6)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=_INTENSITY_CMAP,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_comb, orientation="vertical",
                          fraction=0.015, pad=0.01)
        cb.set_label("Occupancy", color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#080C10")
        plt.close(fig)
        log.info(f"✅  Heatmap saved → {output_path}")

    # ─────────────────────────── Dashboard generator ─────────────────────────
    def generate_dashboard(self, player_stats: dict, total_frames: int,
                           fps: float, output_path: str):
        """
        Renders a professional broadcast-style match analytics dashboard PNG.
        """
        fig = plt.figure(figsize=(20, 12), facecolor="#080C10")
        fig.suptitle(
            "TENNIS MATCH INTELLIGENCE SYSTEM  ·  tubakhxn",
            fontsize=16, color="#00CFFF", fontweight="bold", y=0.97,
            fontfamily="monospace"
        )

        gs = gridspec.GridSpec(3, 4, figure=fig,
                               hspace=0.55, wspace=0.38,
                               left=0.06, right=0.97,
                               top=0.92, bottom=0.07)

        # ── 1. Speed timeline ──
        ax_spd = fig.add_subplot(gs[0, :2])
        self._plot_speed_timeline(ax_spd, player_stats, fps)

        # ── 2. Distance bar chart ──
        ax_dist = fig.add_subplot(gs[0, 2])
        self._plot_distance_bars(ax_dist, player_stats)

        # ── 3. Aggression / Defensive radar ──
        ax_radar = fig.add_subplot(gs[0, 3], projection="polar")
        self._plot_tactical_radar(ax_radar, player_stats)

        # ── 4. Court occupancy (mini heatmap) ──
        ax_occ = fig.add_subplot(gs[1, :2])
        self._plot_court_occupancy(ax_occ, player_stats)

        # ── 5. Speed distribution ──
        ax_violin = fig.add_subplot(gs[1, 2])
        self._plot_speed_violin(ax_violin, player_stats)

        # ── 6. Activity score ──
        ax_act = fig.add_subplot(gs[1, 3])
        self._plot_activity_scores(ax_act, player_stats)

        # ── 7. Match intensity timeline ──
        ax_int = fig.add_subplot(gs[2, :2])
        self._plot_intensity_timeline(ax_int, player_stats, fps)

        # ── 8. Tactical scorecard ──
        ax_tac = fig.add_subplot(gs[2, 2:])
        self._plot_tactical_scorecard(ax_tac, player_stats, total_frames, fps)

        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#080C10")
        plt.close(fig)
        log.info(f"✅  Dashboard saved → {output_path}")

    # ─────────────────────── Dashboard sub-plots ─────────────────────────────

    def _style_ax(self, ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#0D1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E2A38")
        ax.tick_params(colors="#8899AA", labelsize=7)
        ax.set_title(title, color="#00CFFF", fontsize=9, pad=6, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, color="#8899AA", fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, color="#8899AA", fontsize=7)
        ax.grid(True, color="#1E2A38", lw=0.5, alpha=0.7)

    def _plot_speed_timeline(self, ax, player_stats, fps):
        self._style_ax(ax, "SPEED TIMELINE (km/h)", "Time (s)", "Speed (km/h)")
        has_data = False
        for idx, (tid, st) in enumerate(player_stats.items()):
            ci    = idx % len(PLAYER_COLORS_MPL)
            col   = PLAYER_COLORS_MPL[ci]["mpl"]
            speeds = list(st["speeds"])
            if not speeds:
                continue
            has_data = True
            t_axis = np.arange(len(speeds)) / max(fps, 1)
            # Smooth
            smooth = np.convolve(speeds, np.ones(5)/5, mode="same")
            ax.plot(t_axis, smooth, color=col, lw=1.5,
                    label=f"P{tid+1} ({PLAYER_COLORS_MPL[ci]['name']})", alpha=0.9)
            ax.fill_between(t_axis, smooth, alpha=0.08, color=col)
        if has_data:
            ax.legend(loc="upper right", fontsize=7, facecolor="#0D1117",
                      labelcolor="white", edgecolor="#1E2A38")

    def _plot_distance_bars(self, ax, player_stats):
        self._style_ax(ax, "DISTANCE COVERED (m)")
        labels, dists, colors = [], [], []
        for idx, (tid, st) in enumerate(player_stats.items()):
            ci = idx % len(PLAYER_COLORS_MPL)
            labels.append(f"P{tid+1}")
            dists.append(st["dist"])
            colors.append(PLAYER_COLORS_MPL[ci]["mpl"])
        if not labels:
            return
        bars = ax.bar(labels, dists, color=colors, edgecolor="#1E2A38",
                      linewidth=0.8, alpha=0.85)
        for bar, d in zip(bars, dists):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{d:.1f}m", ha="center", va="bottom", color="white",
                    fontsize=7)
        ax.set_ylabel("Metres", color="#8899AA", fontsize=7)

    def _plot_tactical_radar(self, ax, player_stats):
        categories = ["Speed", "Dist", "Coverage", "Aggression", "Stamina"]
        N = len(categories)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        ax.set_facecolor("#0D1117")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories,
                          fontsize=7, color="#8899AA")
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=5, color="#556677")
        ax.grid(color="#1E2A38", lw=0.5)
        ax.spines["polar"].set_edgecolor("#1E2A38")
        ax.set_title("TACTICAL RADAR", color="#00CFFF", fontsize=9,
                     pad=14, fontweight="bold")

        for idx, (tid, st) in enumerate(player_stats.items()):
            ci     = idx % len(PLAYER_COLORS_MPL)
            col    = PLAYER_COLORS_MPL[ci]["mpl"]
            speeds = list(st["speeds"])
            avg_spd = float(np.mean(speeds)) if speeds else 0
            hm      = st["heatmap"]
            cov     = float(np.mean(hm > 0.01) * 100) if hm.max() > 0 else 0
            agg     = min(100, avg_spd * 3.5)
            stamina = min(100, st["frames"] * 0.3)
            dist_n  = min(100, st["dist"] * 2)

            values  = [min(100, avg_spd*3), dist_n, cov, agg, stamina]
            values += values[:1]
            ax.plot(angles, values, color=col, lw=1.5, alpha=0.9)
            ax.fill(angles, values, color=col, alpha=0.12)

    def _plot_court_occupancy(self, ax, player_stats):
        self._style_ax(ax, "COURT OCCUPANCY DENSITY")
        combined = np.zeros((COURT_H, COURT_W), dtype=np.float32)
        for st in player_stats.values():
            hm = st["heatmap"].copy()
            if hm.max() > 0:
                from scipy.ndimage import gaussian_filter
                combined += gaussian_filter(hm / hm.max(), sigma=20)
        if combined.max() > 0:
            combined /= combined.max()

        _draw_court_lines(ax, lw=1.2, alpha=0.5)
        ax.imshow(combined, origin="lower", extent=[0, COURT_W, 0, COURT_H],
                  cmap=_INTENSITY_CMAP, alpha=0.85, aspect="auto")
        ax.set_xlim(0, COURT_W)
        ax.set_ylim(0, COURT_H)
        ax.set_aspect("auto")
        ax.tick_params(labelleft=False, labelbottom=False)

    def _plot_speed_violin(self, ax, player_stats):
        self._style_ax(ax, "SPEED DISTRIBUTION (km/h)")
        data, labels, colors = [], [], []
        for idx, (tid, st) in enumerate(player_stats.items()):
            speeds = list(st["speeds"])
            if len(speeds) < 3:
                continue
            ci = idx % len(PLAYER_COLORS_MPL)
            data.append(speeds)
            labels.append(f"P{tid+1}")
            colors.append(PLAYER_COLORS_MPL[ci]["mpl"])

        if not data:
            return

        parts = ax.violinplot(data, showmeans=True, showmedians=False)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("white")
        parts["cmeans"].set_linewidth(1.5)
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_color("#334455")

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Speed (km/h)", color="#8899AA", fontsize=7)

    def _plot_activity_scores(self, ax, player_stats):
        self._style_ax(ax, "ACTIVITY SCORES")
        labels, scores, colors = [], [], []
        for idx, (tid, st) in enumerate(player_stats.items()):
            ci     = idx % len(PLAYER_COLORS_MPL)
            speeds = list(st["speeds"])
            spd    = float(np.mean(speeds)) if speeds else 0
            score  = min(100, spd * 3 + st["frames"] * 0.05)
            labels.append(f"P{tid+1}\n({PLAYER_COLORS_MPL[ci]['name']})")
            scores.append(score)
            colors.append(PLAYER_COLORS_MPL[ci]["mpl"])

        if not labels:
            return

        y_pos = np.arange(len(labels))
        bars  = ax.barh(y_pos, scores, color=colors, alpha=0.8,
                        edgecolor="#1E2A38")
        for bar, sc in zip(bars, scores):
            ax.text(min(sc + 1, 98), bar.get_y() + bar.get_height()/2,
                    f"{sc:.0f}", va="center", color="white", fontsize=7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Score", color="#8899AA", fontsize=7)

    def _plot_intensity_timeline(self, ax, player_stats, fps):
        self._style_ax(ax, "MATCH INTENSITY TIMELINE", "Time (s)", "Intensity")
        all_speeds = None
        for st in player_stats.values():
            sp = np.array(list(st["speeds"]))
            if len(sp) == 0:
                continue
            all_speeds = sp if all_speeds is None else \
                         (all_speeds[:min(len(all_speeds), len(sp))] +
                          sp[:min(len(all_speeds), len(sp))])
        if all_speeds is None:
            return

        intensity = np.convolve(all_speeds, np.ones(10)/10, mode="same")
        t_axis    = np.arange(len(intensity)) / max(fps, 1)
        ax.plot(t_axis, intensity, color="#00CFFF", lw=1.5, alpha=0.9)
        ax.fill_between(t_axis, intensity, alpha=0.15, color="#00CFFF")

        # High intensity zones
        hi_thresh = float(np.percentile(intensity, 75))
        hi_mask   = intensity >= hi_thresh
        ax.fill_between(t_axis, intensity, where=hi_mask,
                        alpha=0.35, color="#FF6600", label="High intensity")
        ax.axhline(hi_thresh, color="#FF6600", lw=0.8, linestyle="--", alpha=0.6)
        ax.legend(fontsize=7, facecolor="#0D1117", labelcolor="white",
                  edgecolor="#1E2A38")

    def _plot_tactical_scorecard(self, ax, player_stats, total_frames, fps):
        ax.set_facecolor("#0D1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E2A38")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("TACTICAL SCORECARD", color="#00CFFF", fontsize=9,
                     pad=6, fontweight="bold")

        duration = total_frames / max(fps, 1)
        m, s     = divmod(int(duration), 60)

        headers = ["PLAYER", "AVG SPD", "TOP SPD", "DIST", "COVERAGE",
                   "AGGRESSION", "DEFENSE", "ACTIVITY"]
        col_x   = np.linspace(0.02, 0.98, len(headers))

        # Header row
        for hx, h in zip(col_x, headers):
            ax.text(hx, 0.92, h, transform=ax.transAxes,
                    color="#556677", fontsize=6.5, ha="center", va="center",
                    fontweight="bold", fontfamily="monospace")
        ax.axhline(0.88, color="#1E2A38", lw=0.8,
                   transform=ax.get_xaxis_transform())

        # Player rows
        for ridx, (tid, st) in enumerate(player_stats.items()):
            ci     = ridx % len(PLAYER_COLORS_MPL)
            cinfo  = PLAYER_COLORS_MPL[ci]
            col    = cinfo["mpl"]
            speeds = list(st["speeds"])
            avg_sp = float(np.mean(speeds)) if speeds else 0
            top_sp = float(np.max(speeds)) if speeds else 0
            dist   = st["dist"]
            hm     = st["heatmap"]
            cov    = float(np.mean(hm > 0.01) * 100) if hm.max() > 0 else 0
            agg    = min(100, int(avg_sp * 3.5))
            deff   = max(0, 100 - agg)
            act    = min(100, int(avg_sp * 3 + st["frames"] * 0.05))

            y_pos  = 0.80 - ridx * 0.14
            bg_col = "#0F1820" if ridx % 2 == 0 else "#0D1117"
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, y_pos - 0.05), 0.98, 0.12,
                boxstyle="round,pad=0.01",
                facecolor=bg_col, edgecolor=col, lw=0.8,
                transform=ax.transAxes
            ))

            row_vals = [
                f"P{tid+1} ({cinfo['name']})",
                f"{avg_sp:.1f} km/h",
                f"{top_sp:.1f} km/h",
                f"{dist:.1f} m",
                f"{cov:.1f}%",
                f"{agg}",
                f"{deff}",
                f"{act}",
            ]
            for hx, val in zip(col_x, row_vals):
                ax.text(hx, y_pos + 0.02, val, transform=ax.transAxes,
                        color=col if hx == col_x[0] else "white",
                        fontsize=7, ha="center", va="center",
                        fontfamily="monospace")

        # Footer
        ax.text(0.5, 0.03,
                f"Match Duration: {m:02d}:{s:02d}  │  "
                f"Total Frames: {total_frames}  │  FPS: {fps:.1f}  │  "
                f"System: tubakhxn · Tennis Match Intelligence",
                transform=ax.transAxes, color="#445566", fontsize=6.5,
                ha="center", va="center", fontfamily="monospace")