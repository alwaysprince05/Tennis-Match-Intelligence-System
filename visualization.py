"""
visualization.py — Tennis Match Intelligence System
Developer/Creator: tubakhxn

Broadcast-quality neon visualization: segmentation masks, glow trails,
motion vectors, heatmap overlays, tactical zones, and the HUD dashboard.
"""

import cv2
import numpy as np
from collections import deque

# ─────────────────────────── Colour palette ──────────────────────────────────
PLAYER_COLORS = {
    0: {"neon": (0, 255, 255),   "name": "Cyan",    "dark": (0, 120, 120)},
    1: {"neon": (255, 0, 255),   "name": "Magenta", "dark": (120, 0, 120)},
    2: {"neon": (0, 255, 0),     "name": "Green",   "dark": (0, 120, 0)},
    3: {"neon": (0, 165, 255),   "name": "Orange",  "dark": (0, 80, 130)},
    4: {"neon": (255, 255, 0),   "name": "Yellow",  "dark": (120, 120, 0)},
    5: {"neon": (180, 0, 255),   "name": "Purple",  "dark": (90, 0, 120)},
}

TRAIL_LENGTH = 45


# ─────────────────────────── Helper drawing utilities ────────────────────────
def _blend(frame, overlay, alpha):
    """In-place weighted blend."""
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_glow_rect(frame, pt1, pt2, color, thickness=2, layers=4):
    """Draw a rectangle with softening outer glow rings."""
    for i in range(layers, 0, -1):
        alpha = 0.06 * i
        expand = i * 3
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (pt1[0]-expand, pt1[1]-expand),
                      (pt2[0]+expand, pt2[1]+expand),
                      color, thickness + i)
        _blend(frame, overlay, alpha)
    cv2.rectangle(frame, pt1, pt2, color, thickness)


def _draw_glow_circle(frame, center, radius, color, layers=4):
    for i in range(layers, 0, -1):
        alpha = 0.07 * i
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + i * 2, color, 2)
        _blend(frame, overlay, alpha)
    cv2.circle(frame, center, radius, color, 2)


def _put_label(frame, text, pos, color, font_scale=0.5, thickness=1, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = pos
    if bg:
        cv2.rectangle(frame, (x-4, y-th-6), (x+tw+4, y+4), (10, 10, 10), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


# ─────────────────────────── Segmentation mask ───────────────────────────────
def _draw_segmentation_mask(frame, x1, y1, x2, y2, color, alpha=0.38):
    """Fills player bounding box with a neon semi-transparent mask."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()
    # Full body rectangle mask
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    # Gradient fade at top (head region brighter)
    head_h = max(1, (y2 - y1) // 4)
    cv2.rectangle(overlay, (x1, y1), (x2, y1 + head_h),
                  tuple(min(255, c + 60) for c in color), -1)
    _blend(frame, overlay, alpha)


# ─────────────────────────── Motion trail ────────────────────────────────────
def _draw_trail(frame, history, color):
    """Fading neon motion trail following player centroid."""
    pts = list(history)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        frac  = i / len(pts)
        alpha = frac * 0.85
        thick = max(1, int(frac * 4))
        col   = tuple(int(c * frac) for c in color)
        overlay = frame.copy()
        cv2.line(overlay, pts[i-1], pts[i], col, thick, cv2.LINE_AA)
        _blend(frame, overlay, alpha)
    # Bright tip
    cv2.circle(frame, pts[-1], 4, color, -1)


# ─────────────────────────── Motion vector ───────────────────────────────────
def _draw_velocity_vector(frame, history, color):
    if len(history) < 5:
        return
    pts = list(history)
    dx  = pts[-1][0] - pts[-5][0]
    dy  = pts[-1][1] - pts[-5][1]
    spd = np.sqrt(dx**2 + dy**2)
    if spd < 2:
        return
    scale = min(3.5, spd / 8)
    ex = int(pts[-1][0] + dx * scale)
    ey = int(pts[-1][1] + dy * scale)
    overlay = frame.copy()
    cv2.arrowedLine(overlay, pts[-1], (ex, ey), color, 2, cv2.LINE_AA,
                    tipLength=0.35)
    _blend(frame, overlay, 0.7)


# ─────────────────────────── Tactical zone overlay ───────────────────────────
def _draw_tactical_zone(frame, x1, y1, x2, y2, color):
    """Draws a subtle tactical zone circle around the player."""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    r  = int(max((x2 - x1), (y2 - y1)) * 0.9)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r, color, 1)
    cv2.circle(overlay, (cx, cy), int(r * 1.4),
               tuple(c // 2 for c in color), 1)
    _blend(frame, overlay, 0.25)


# ─────────────────────────── Player HUD card ─────────────────────────────────
def _draw_player_card(frame, x1, y1, x2, y2, tid, ci, stats):
    color = PLAYER_COLORS[ci]["neon"]
    cname = PLAYER_COLORS[ci]["name"]
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

    speeds = stats["speeds"]
    spd    = np.mean(list(speeds)[-5:]) if speeds else 0.0
    dist   = stats["dist"]
    frames = stats["frames"]

    # Aggression / tactical score
    agg_score  = min(100, int(spd * 3.5))
    def_score  = min(100, max(0, 100 - agg_score))

    card_x = x1i
    card_y = max(0, y1i - 90)

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + 140, card_y + 82),
                  (5, 8, 12), -1)
    _blend(frame, overlay, 0.72)

    # Coloured top bar
    cv2.rectangle(frame, (card_x, card_y), (card_x + 140, card_y + 6), color, -1)

    # ID line
    _put_label(frame, f"P{tid+1} · {cname}",
               (card_x + 5, card_y + 20), color, 0.45, 1, bg=False)

    # Stats
    _put_label(frame, f"SPD  {spd:5.1f} km/h",
               (card_x + 5, card_y + 38), (200, 200, 200), 0.38, 1, bg=False)
    _put_label(frame, f"DIST {dist:5.1f} m",
               (card_x + 5, card_y + 53), (200, 200, 200), 0.38, 1, bg=False)
    _put_label(frame, f"AGG  {agg_score:3d}",
               (card_x + 5, card_y + 68), (100, 220, 255), 0.38, 1, bg=False)
    _put_label(frame, f"DEF  {def_score:3d}",
               (card_x + 75, card_y + 68), (100, 255, 160), 0.38, 1, bg=False)

    # Speed bar
    bar_w = int(np.clip(spd / 30 * 130, 0, 130))
    cv2.rectangle(frame, (card_x + 5, card_y + 75), (card_x + 5 + bar_w, card_y + 80),
                  color, -1)
    cv2.rectangle(frame, (card_x + 5, card_y + 75), (card_x + 135, card_y + 80),
                  (60, 60, 60), 1)


# ─────────────────────────── Global HUD overlay ──────────────────────────────
def _draw_global_hud(frame, active_players, frame_idx, fps, player_stats, tracker):
    H, W = frame.shape[:2]

    # ── Top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 36), (5, 8, 12), -1)
    _blend(frame, overlay, 0.78)
    cv2.line(frame, (0, 36), (W, 36), (0, 200, 255), 1)

    # Title
    _put_label(frame, "TENNIS MATCH INTELLIGENCE", (14, 25),
               (0, 220, 255), 0.6, 1, bg=False)

    # Frame counter & time
    elapsed = frame_idx / max(fps, 1)
    m, s    = divmod(int(elapsed), 60)
    time_s  = f"{m:02d}:{s:02d}"
    _put_label(frame, f"FRAME {frame_idx:05d}   {time_s}",
               (W - 220, 25), (160, 160, 160), 0.5, 1, bg=False)

    # ── Bottom status bar ──
    bar_h = 34
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, H - bar_h), (W, H), (5, 8, 12), -1)
    _blend(frame, overlay2, 0.78)
    cv2.line(frame, (0, H - bar_h), (W, H - bar_h), (0, 200, 255), 1)

    # Active players
    _put_label(frame, f"ACTIVE PLAYERS: {active_players}",
               (14, H - 12), (0, 220, 255), 0.45, 1, bg=False)

    # Per-player speed pills
    x_cursor = 220
    for tid, st in player_stats.items():
        ci    = tracker.color_idx(tid)
        color = PLAYER_COLORS[ci]["neon"]
        spd   = np.mean(list(st["speeds"])[-3:]) if st["speeds"] else 0
        pill  = f"P{tid+1}: {spd:.0f}km/h"
        _put_label(frame, pill, (x_cursor, H - 12), color, 0.4, 1, bg=False)
        x_cursor += 110
        if x_cursor > W - 200:
            break

    # Intensity meter
    all_speeds = []
    for st in player_stats.values():
        all_speeds.extend(list(st["speeds"])[-5:])
    intensity = min(100, int(np.mean(all_speeds) * 3)) if all_speeds else 0
    ibar_w    = int(intensity / 100 * 140)
    ix        = W - 175
    iy        = H - 24
    _put_label(frame, "INTENSITY", (ix, iy - 1), (160, 160, 160), 0.36, 1, bg=False)
    cv2.rectangle(frame, (ix, iy + 2), (ix + 140, iy + 10), (40, 40, 40), -1)
    bar_col = (0, 255, 100) if intensity < 60 else (0, 165, 255) if intensity < 80 else (0, 80, 255)
    cv2.rectangle(frame, (ix, iy + 2), (ix + ibar_w, iy + 10), bar_col, -1)

    # ── tubakhxn watermark ──
    _put_label(frame, "tubakhxn", (W - 80, 25), (80, 80, 80), 0.38, 1, bg=False)


# ─────────────────────────── Court line extraction (demo) ────────────────────
def _detect_court_lines(frame):
    """Lightweight court line detection via Canny + HoughLines."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=60, maxLineGap=20)
    return lines


# ─────────────────────────── Main Visualizer class ───────────────────────────
class TennisVisualizer:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height

    def render_frame(self, frame, tracks, tracker, player_stats,
                     frame_idx: int, fps: float) -> np.ndarray:
        """
        Full render pipeline for a single frame.
        Returns the annotated frame (H×W×3 uint8).
        """
        out = frame.copy()

        # ── Court line overlay (subtle cyan) ──
        lines = _detect_court_lines(frame)
        if lines is not None:
            for ln in lines[:30]:
                x1l, y1l, x2l, y2l = ln[0]
                loverlay = out.copy()
                cv2.line(loverlay, (x1l, y1l), (x2l, y2l), (0, 255, 200), 1)
                _blend(out, loverlay, 0.18)

        # ── Per-player rendering ──
        for t in tracks:
            x1, y1, x2, y2, tid = t
            tid = int(tid)
            ci  = tracker.color_idx(tid)
            col = PLAYER_COLORS[ci]["neon"]
            st  = player_stats.get(tid, {})
            hist = tracker.get_history(tid)

            # 1. Segmentation mask
            _draw_segmentation_mask(out, x1, y1, x2, y2, col)

            # 2. Glow bounding box
            _draw_glow_rect(out, (int(x1), int(y1)), (int(x2), int(y2)), col)

            # 3. Head glow circle
            cx, cy = int((x1+x2)/2), int(y1)
            _draw_glow_circle(out, (cx, cy - 10), 12, col)

            # 4. Tactical zone
            _draw_tactical_zone(out, x1, y1, x2, y2, col)

            # 5. Motion trail
            if hist:
                _draw_trail(out, hist, col)

            # 6. Velocity vector
            if hist:
                _draw_velocity_vector(out, hist, col)

            # 7. Player card HUD
            if st:
                _draw_player_card(out, x1, y1, x2, y2, tid, ci, st)

        # ── Global HUD ──
        _draw_global_hud(out, len(tracks), frame_idx, fps, player_stats, tracker)

        return out