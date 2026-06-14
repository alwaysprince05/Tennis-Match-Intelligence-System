"""
Tennis Match Intelligence System
Developer/Creator: tubakhxn
A professional tennis analytics platform with computer vision,
player tracking, tactical analysis, and broadcast-quality visualization.
"""

import subprocess
import sys
import os
import logging
import time
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────── Auto-install dependencies ───────────────────────
REQUIRED_PACKAGES = [
    "opencv-python",
    "numpy",
    "scipy",
    "matplotlib",
    "ultralytics",
    "Pillow",
    "tqdm",
    "lap",
    "filterpy",
    "scikit-learn",
]

def install_packages():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║       Tennis Match Intelligence System — tubakhxn            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    print("🔧  Checking and installing dependencies...\n")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_").split("==")[0])
        except ImportError:
            print(f"   Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    print("✅  All dependencies satisfied.\n")

install_packages()

# ─────────────────────────── Core imports ────────────────────────────────────
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
from PIL import Image
import torch

# ─────────────────────────── Logging setup ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TMIS")

# ─────────────────────────── Constants ───────────────────────────────────────
PLAYER_COLORS = {
    0: {"neon": (0, 255, 255),   "name": "Cyan",    "hex": "#00FFFF"},   # Player 1
    1: {"neon": (255, 0, 255),   "name": "Magenta", "hex": "#FF00FF"},   # Player 2
    2: {"neon": (0, 255, 0),     "name": "Green",   "hex": "#00FF00"},   # Player 3
    3: {"neon": (0, 165, 255),   "name": "Orange",  "hex": "#FFA500"},   # Player 4
    4: {"neon": (255, 255, 0),   "name": "Yellow",  "hex": "#FFFF00"},   # Player 5
    5: {"neon": (128, 0, 255),   "name": "Purple",  "hex": "#8000FF"},   # Player 6
}

TRAIL_LENGTH       = 45
HEATMAP_RADIUS     = 35
FPS_DEFAULT        = 30
OUTPUT_VIDEO       = "output.mp4"
OUTPUT_HEATMAP     = "heatmap.png"
OUTPUT_DASHBOARD   = "dashboard.png"
COURT_W, COURT_H   = 1344, 576   # Synthetic court canvas size
DEMO_FRAMES        = 300          # frames to generate if no video supplied

# ─────────────────────────── Byte-style simple tracker ───────────────────────
class SimpleTracker:
    """
    Lightweight IoU-based tracker that provides persistent IDs.
    Falls back gracefully when lap/filterpy are unavailable.
    """
    def __init__(self, max_lost=30, iou_thresh=0.25):
        self.max_lost  = max_lost
        self.iou_thresh = iou_thresh
        self.tracks    = {}           # id → {box, lost, color_idx, history}
        self._next_id  = 0
        self._color_map = {}

    # ---------- public API ----------
    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, conf]
        returns:    list of [x1, y1, x2, y2, track_id]
        """
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        dets  = np.array(detections)
        boxes = dets[:, :4]

        if not self.tracks:
            results = []
            for b in boxes:
                tid = self._new_track(b)
                results.append([*b, tid])
            return results

        track_ids  = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[t]["box"] for t in track_ids])

        iou_matrix = self._iou_matrix(track_boxes, boxes)

        matched_t, matched_d = set(), set()
        pairs = []
        while True:
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[idx] < self.iou_thresh:
                break
            pairs.append(idx)
            iou_matrix[idx[0], :] = -1
            iou_matrix[:, idx[1]] = -1
            matched_t.add(idx[0])
            matched_d.add(idx[1])

        results = []
        for ti, di in pairs:
            tid = track_ids[ti]
            b   = boxes[di]
            cx  = int((b[0] + b[2]) / 2)
            cy  = int((b[1] + b[3]) / 2)
            self.tracks[tid]["box"]  = b
            self.tracks[tid]["lost"] = 0
            self.tracks[tid]["history"].append((cx, cy))
            results.append([*b, tid])

        for di in range(len(boxes)):
            if di not in matched_d:
                tid = self._new_track(boxes[di])
                results.append([*boxes[di], tid])

        for ti, tid in enumerate(track_ids):
            if ti not in matched_t:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return results

    def get_history(self, tid):
        return list(self.tracks.get(tid, {}).get("history", []))

    def color_idx(self, tid):
        return self._color_map.get(tid, 0)

    # ---------- helpers ----------
    def _new_track(self, box):
        tid = self._next_id
        self._next_id += 1
        ci  = tid % len(PLAYER_COLORS)
        self._color_map[tid] = ci
        cx  = int((box[0] + box[2]) / 2)
        cy  = int((box[1] + box[3]) / 2)
        self.tracks[tid] = {
            "box":     box,
            "lost":    0,
            "history": deque([(cx, cy)], maxlen=TRAIL_LENGTH),
        }
        return tid

    @staticmethod
    def _iou_matrix(a, b):
        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ix1 = np.maximum(ax1[:, None], bx1[None, :])
        iy1 = np.maximum(ay1[:, None], by1[None, :])
        ix2 = np.minimum(ax2[:, None], bx2[None, :])
        iy2 = np.minimum(ay2[:, None], by2[None, :])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        aa    = (ax2 - ax1) * (ay2 - ay1)
        ab    = (bx2 - bx1) * (by2 - by1)
        union = aa[:, None] + ab[None, :] - inter
        return np.where(union > 0, inter / union, 0)


# ─────────────────────────── Synthetic demo frame generator ──────────────────
class DemoVideoGenerator:
    """
    Generates a realistic synthetic tennis match scene when no video is supplied.
    Includes court, players, and motion for full pipeline testing.
    """
    def __init__(self, width=1280, height=720, n_frames=DEMO_FRAMES):
        self.W, self.H = width, height
        self.n_frames  = n_frames
        self.t         = 0
        self._init_players()

    def _init_players(self):
        W, H = self.W, self.H
        self.players = [
            {
                "x": W * 0.3, "y": H * 0.65,
                "vx": 1.5,    "vy": 0.8,
                "range_x": (W * 0.1, W * 0.45),
                "range_y": (H * 0.55, H * 0.85),
                "phase": 0.0,
            },
            {
                "x": W * 0.7, "y": H * 0.35,
                "vx": -1.3,   "vy": -0.9,
                "range_x": (W * 0.55, W * 0.9),
                "range_y": (H * 0.15, H * 0.45),
                "phase": np.pi,
            },
        ]

    def _draw_court(self, frame):
        W, H = self.W, self.H
        # Dark teal court surface
        cv2.rectangle(frame, (int(W*0.05), int(H*0.1)), (int(W*0.95), int(H*0.9)),
                      (80, 110, 60), -1)
        # Court outline
        cv2.rectangle(frame, (int(W*0.05), int(H*0.1)), (int(W*0.95), int(H*0.9)),
                      (220, 220, 200), 2)
        # Net
        cv2.line(frame, (int(W*0.5), int(H*0.1)), (int(W*0.5), int(H*0.9)),
                 (240, 240, 240), 3)
        # Service boxes
        for yf in [0.1, 0.5, 0.9]:
            cv2.line(frame, (int(W*0.05), int(H*yf)), (int(W*0.95), int(H*yf)),
                     (200, 200, 180), 1)
        cv2.line(frame, (int(W*0.5), int(H*0.1)), (int(W*0.5), int(H*0.5)),
                 (200, 200, 180), 1)
        cv2.line(frame, (int(W*0.5), int(H*0.5)), (int(W*0.5), int(H*0.9)),
                 (200, 200, 180), 1)
        # Baseline labels
        for xf in [0.05, 0.95]:
            cv2.line(frame, (int(W*xf), int(H*0.1)), (int(W*xf), int(H*0.9)),
                     (220, 220, 200), 2)
        return frame

    def _draw_player(self, frame, px, py, color_idx):
        ci  = color_idx % len(PLAYER_COLORS)
        col = PLAYER_COLORS[ci]["neon"]
        pw, ph = 30, 60
        x1, y1 = int(px - pw/2), int(py - ph)
        x2, y2 = int(px + pw/2), int(py)
        # Semi-transparent body mask
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        # Head
        cv2.circle(frame, (int(px), int(py - ph - 12)), 12, col, -1)
        # Glow outline
        for g in [3, 6, 9]:
            alpha = 0.15
            glow = frame.copy()
            cv2.rectangle(glow, (x1-g, y1-g), (x2+g, y2+g), col, 2)
            cv2.addWeighted(glow, alpha, frame, 1-alpha, 0, frame)
        # Bright outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        return (x1, y1, x2, y2)

    def next_frame(self):
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # Background
        frame[:] = (25, 30, 20)
        frame = self._draw_court(frame)

        boxes = []
        for i, p in enumerate(self.players):
            # Sinusoidal movement
            p["x"] += np.sin(self.t * 0.07 + p["phase"]) * 2.0
            p["y"] += np.cos(self.t * 0.05 + p["phase"]) * 1.5
            # Clamp
            p["x"] = np.clip(p["x"], p["range_x"][0], p["range_x"][1])
            p["y"] = np.clip(p["y"], p["range_y"][0], p["range_y"][1])
            box = self._draw_player(frame, p["x"], p["y"], i)
            boxes.append([*box, 0.95])  # fake conf

        # Timestamp watermark
        cv2.putText(frame, f"DEMO FRAME {self.t+1:04d}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        self.t += 1
        return frame, boxes

    def __len__(self):
        return self.n_frames


# ─────────────────────────── Main pipeline ───────────────────────────────────
class TennisIntelligenceSystem:
    def __init__(self, source: str | None = None):
        self.source  = source
        self.use_demo = source is None or not Path(source).exists()

        log.info("Initialising Tennis Match Intelligence System — tubakhxn")
        self._init_model()
        self._init_tracker()

        # Analytics state
        self.player_stats     = defaultdict(lambda: {
            "positions": deque(maxlen=TRAIL_LENGTH),
            "speeds":    deque(maxlen=60),
            "dist":      0.0,
            "heatmap":   np.zeros((COURT_H, COURT_W), dtype=np.float32),
            "frames":    0,
        })
        self.frame_count = 0
        self.fps         = FPS_DEFAULT

    # ---------- model ----------
    def _init_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {device.upper()}")
        try:
            from ultralytics import YOLO
            log.info("Loading YOLOv8n model (person class)...")
            self.model  = YOLO("yolov8n.pt")
            self.device = device
            self.use_yolo = True
            log.info("✅  YOLO model loaded.")
        except Exception as e:
            log.warning(f"YOLO unavailable ({e}). Using demo bounding boxes.")
            self.model    = None
            self.use_yolo = False
            self.device   = "cpu"

    def _init_tracker(self):
        self.tracker = SimpleTracker(max_lost=40, iou_thresh=0.20)
        log.info("✅  Tracker initialised.")

    # ---------- detection ----------
    def _detect(self, frame, demo_boxes=None):
        if self.use_yolo and self.model is not None:
            results = self.model(frame, classes=[0], conf=0.35, verbose=False)[0]
            boxes   = []
            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0])
                boxes.append([x1, y1, x2, y2, conf])
            return boxes
        else:
            return demo_boxes or []

    # ---------- main run ----------
    def run(self):
        from visualization import TennisVisualizer
        from analytics      import TennisAnalytics

        # ── Open source ──
        if self.use_demo:
            log.info("No video supplied — generating synthetic demo match.")
            gen      = DemoVideoGenerator()
            n_frames = len(gen)
            width, height = gen.W, gen.H
        else:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                log.error(f"Cannot open video: {self.source}")
                return
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_src  = cap.get(cv2.CAP_PROP_FPS)
            if fps_src > 0:
                self.fps = fps_src

        log.info(f"Video: {width}×{height}  {n_frames} frames  {self.fps:.1f} FPS")

        # ── Video writer ──
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, self.fps, (width, height))

        visualizer = TennisVisualizer(width, height)
        analytics  = TennisAnalytics()

        log.info(f"Processing {n_frames} frames → {OUTPUT_VIDEO}")
        start = time.time()

        for fi in tqdm(range(n_frames), desc="🎾 Analysing", unit="frame",
                       bar_format="{l_bar}{bar:35}{r_bar}"):
            # ── Read frame ──
            if self.use_demo:
                frame, demo_boxes = gen.next_frame()
                detections = self._detect(frame, demo_boxes)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self._detect(frame)

            # ── Track ──
            tracks = self.tracker.update(detections)

            # ── Update stats ──
            for t in tracks:
                x1, y1, x2, y2, tid = t
                tid = int(tid)
                cx  = (x1 + x2) / 2
                cy  = (y1 + y2) / 2
                pst = self.player_stats[tid]

                # Speed
                if pst["positions"]:
                    prev = pst["positions"][-1]
                    dx   = (cx - prev[0]) / self.fps
                    dy   = (cy - prev[1]) / self.fps
                    px_per_m = height / 23.77  # standard court length
                    speed_mps = np.sqrt(dx**2 + dy**2) / px_per_m
                    speed_kmh = speed_mps * 3.6
                    pst["speeds"].append(speed_kmh)
                    pst["dist"] += speed_mps / self.fps

                pst["positions"].append((cx, cy))
                pst["frames"] += 1

                # Heatmap — map to court canvas
                hx = int(np.clip(cx / width  * COURT_W, 0, COURT_W - 1))
                hy = int(np.clip(cy / height * COURT_H, 0, COURT_H - 1))
                r  = HEATMAP_RADIUS
                y0, y1h = max(0, hy-r), min(COURT_H, hy+r)
                x0, x1h = max(0, hx-r), min(COURT_W, hx+r)
                Y, X = np.ogrid[y0-hy:y1h-hy, x0-hx:x1h-hx]
                mask = X*X + Y*Y <= r*r
                pst["heatmap"][y0:y1h, x0:x1h][mask] += 1.0

            # ── Render ──
            rendered = visualizer.render_frame(
                frame, tracks, self.tracker, self.player_stats,
                self.frame_count, self.fps
            )
            writer.write(rendered)
            self.frame_count += 1

        writer.release()
        if not self.use_demo:
            cap.release()

        elapsed = time.time() - start
        log.info(f"✅  Video written → {OUTPUT_VIDEO}  ({elapsed:.1f}s)")

        # ── Post-process outputs ──
        analytics.generate_heatmap(self.player_stats, OUTPUT_HEATMAP)
        analytics.generate_dashboard(self.player_stats, self.frame_count,
                                     self.fps, OUTPUT_DASHBOARD)

        self._print_summary()

    def _print_summary(self):
        print("\n" + "═"*62)
        print("  MATCH INTELLIGENCE SUMMARY — tubakhxn")
        print("═"*62)
        for tid, st in self.player_stats.items():
            ci    = self.tracker.color_idx(tid)
            cname = PLAYER_COLORS[ci]["name"]
            spd   = np.mean(st["speeds"]) if st["speeds"] else 0
            print(f"  Player {tid+1} ({cname:8s}) │ "
                  f"Dist: {st['dist']:6.1f}m │ "
                  f"Avg Speed: {spd:5.1f} km/h")
        print("═"*62)
        print(f"  Output video  : {OUTPUT_VIDEO}")
        print(f"  Heatmap       : {OUTPUT_HEATMAP}")
        print(f"  Dashboard     : {OUTPUT_DASHBOARD}")
        print("═"*62 + "\n")


# ─────────────────────────── Entry point ─────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tennis Match Intelligence System")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video (omit for synthetic demo)")
    args = parser.parse_args()

    system = TennisIntelligenceSystem(source=args.video)
    system.run()