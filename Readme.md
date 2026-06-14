# Tennis Match Intelligence System

## Dev / Creator — Prince Maurya

> A professional-grade AI + Computer Vision platform that transforms raw tennis match footage into real-time tactical intelligence — built with the visual language of modern sports broadcasting.

---

## What It Does

| Feature | Description |
|---|---|
| Player Detection | YOLOv8 detects every person on court every frame |
| Multi-Object Tracking | ByteTrack-style IoU tracker assigns persistent IDs |
| Neon Segmentation | Each player gets a unique neon colour mask + glow |
| Motion Trails | Fading trajectory lines follow every player's path |
| Speed Estimation | Per-player speed in km/h derived from pixel displacement |
| Court Heatmaps | Gaussian court occupancy maps per player + combined |
| Tactical Scores | Aggression, Defensive, and Activity scores per player |
| Match Dashboard | Full broadcast-style analytics dashboard PNG |
| Output Video | Annotated `output.mp4` with all overlays rendered |

---

## File Structure

```
tennis_intelligence/
├── main.py           ← Entry point. Auto-installs deps, runs full pipeline
├── visualization.py  ← Neon rendering, segmentation, trails, HUD overlays
└── analytics.py      ← Heatmaps, radar charts, dashboards, scorecards
```

> **Only three files. No additional modules required.**

---

## How to Run

### 1. Clone / Fork the project

```bash
# Clone via HTTPS
git clone https://github.com/alwaysprince05/Tennis-Match-Intelligence-System.git

# Or fork it on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/tennis-match-intelligence.git

cd tennis-match-intelligence
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Run

```bash
# Demo mode — generates a synthetic match automatically (no video needed)
python main.py

# With your own match footage
python main.py --video path/to/match.mp4
```

> All missing Python packages are **auto-installed on first run**. No manual `pip install` needed.

### 4. Outputs

| File | Description |
|---|---|
| `output.mp4` | Annotated match video with neon overlays |
| `heatmap.png` | Per-player + combined court heatmaps |
| `dashboard.png` | Full analytics dashboard with charts and scorecard |

---

## Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.9 + |
| RAM | 4 GB |
| GPU | Optional (CUDA auto-detected, CPU fallback included) |
| OS | Windows / macOS / Linux |

Auto-installed dependencies: `opencv-python`, `ultralytics`, `numpy`, `scipy`, `matplotlib`, `Pillow`, `tqdm`, `lap`, `filterpy`, `scikit-learn`

---

## Player Colour Scheme

| Player | Colour | Hex |
|---|---|---|
| Player 1 | Neon Cyan | `#00FFFF` |
| Player 2 | Neon Magenta | `#FF00FF` |
| Player 3 | Neon Green | `#00FF00` |
| Player 4 | Neon Orange | `#FFA500` |
| Player 5 | Neon Yellow | `#FFFF00` |
| Player 6 | Neon Purple | `#8000FF` |

---

## Tech Stack

| Technology | Role |
|---|---|
| [Python](https://www.python.org/) | Core language |
| [OpenCV](https://opencv.org/) | Frame I/O, rendering, computer vision |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Real-time person detection |
| [NumPy](https://numpy.org/) | Numerical computation |
| [SciPy](https://scipy.org/) | Gaussian smoothing for heatmaps |
| [Matplotlib](https://matplotlib.org/) | Dashboard and heatmap generation |
| [Pillow](https://python-pillow.org/) | Image I/O |
| [tqdm](https://github.com/tqdm/tqdm) | Progress bar |

---

## Relevant Wikipedia Links

### Tennis & Sport Science
- [Tennis](https://en.wikipedia.org/wiki/Tennis) — Rules, court dimensions, scoring
- [Tennis court](https://en.wikipedia.org/wiki/Tennis_court) — Court types and dimensions used in speed calibration
- [Tennis tactics](https://en.wikipedia.org/wiki/Tennis_tactics) — Tactical concepts powering the aggression and defense scores
- [Rally (tennis)](https://en.wikipedia.org/wiki/Rally_(tennis)) — What the rally activity analysis measures
- [Sports analytics](https://en.wikipedia.org/wiki/Sports_analytics) — The broader field this system belongs to

### Computer Vision & AI
- [Computer vision](https://en.wikipedia.org/wiki/Computer_vision) — Foundation of the detection pipeline
- [Object detection](https://en.wikipedia.org/wiki/Object_detection) — YOLO's core task
- [You Only Look Once (YOLO)](https://en.wikipedia.org/wiki/You_Only_Look_Once) — The detection model used
- [Multiple object tracking](https://en.wikipedia.org/wiki/Multiple_object_tracking) — How persistent player IDs work
- [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) — Underpins motion and speed estimation
- [Heat map](https://en.wikipedia.org/wiki/Heat_map) — The court occupancy visualisation
- [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) — Architecture behind YOLO
- [Image segmentation](https://en.wikipedia.org/wiki/Image_segmentation) — What the neon player masks implement

### Algorithms Used
- [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) — Used internally by ByteTrack-style tracking
- [Jaccard index / IoU](https://en.wikipedia.org/wiki/Jaccard_index) — IoU metric driving the tracker's matching step
- [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) — Applied to smooth court heatmaps
- [K-nearest neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) — Used in player re-identification

---

## How to Fork & Contribute

1. **Fork** this repository on GitHub (top-right "Fork" button)
2. **Clone** your fork locally (see step 1 under How to Run)
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-improvement
   ```
4. **Make your changes**, then commit:
   ```bash
   git add .
   git commit -m "feat: describe your change"
   ```
5. **Push** and open a Pull Request:
   ```bash
   git push origin feature/my-improvement
   ```

### Ideas for contributions
- Add ball detection and trajectory tracking
- Integrate serve speed radar
- Add shot classification (forehand / backhand / serve)
- Export stats to CSV or JSON for further analysis
- Add live webcam / RTSP stream support

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

<p align="center">Built by <strong>Prince Maurya</strong></p>