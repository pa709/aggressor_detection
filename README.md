# Aggressor Detection

Binary aggressor / non-aggressor classifier from surveillance video.
Two independent pipelines are provided — a **pose-based** approach (root level)
and an **optical flow** approach (`optical_flow/` subfolder).

---

## Folder Structure

```
aggressor_detection/
│
├── README.md
├── .gitignore
│
│  Pose-based pipeline ─────────────────────────────────────────────────────────
├── main.py                      # Pipeline entry point (CLI)
├── bounding_boxes.py            # YOLOv8 + BoT-SORT tracker with CLAHE, multi-scale,
│                                #   gap interpolation and optional Real-ESRGAN SR
├── module1_data_loader.py       # Dataset discovery, CSV cross-reference, XML parsing
├── module2_augmenter.py         # Per-frame augmentation (flip, brightness, blur, HSV)
├── module3_pose_estimator.py    # YOLOv8-pose keypoint extraction per bounding box
├── module4_classifier.py        # 126-d feature engineering + LR classifier training
├── module5_validation.py        # Evaluation, reporting, prediction overlay video
├── ntu_cctv_stage1.ipynb        # Exploratory notebook — Stage 1 fight detection
│
├── data/
│   ├── data.csv                 # Fight frame ranges per video (frame_start, frame_end)
│   ├── video_info.csv           # Trimmed clip frame counts (optional, speeds up loading)
│   └── video_data/              # ← gitignored: one subfolder per video
│       └── <video_name>/
│           ├── <video_name>.mp4
│           └── annotations.xml
│
├── models/                      # ← gitignored: saved classifier .pkl files
├── cache/                       # ← gitignored: pose estimation pickle caches
├── outputs/                     # ← gitignored: prediction overlay videos
│
│  Optical flow pipeline ───────────────────────────────────────────────────────
└── optical_flow/
    ├── README.md                # Detailed optical flow module documentation
    ├── requirements.txt
    │
    ├── features/
    │   ├── extract_flow_features.py               # Single-track (N, 30, 11) features
    │   ├── extract_flow_features_j.py             # Single-track variant — annotator batch J
    │   ├── extract_flow_features_p.py             # Single-track variant — annotator batch P
    │   ├── extract_pairwise_flow_features_dual.py # Pairwise (N, 30, 15) — Format A & B
    │   ├── extract_pairwise_flow_features_j.py    # Pairwise variant — annotator batch J
    │   └── extract_pairwise_flow_features_p.py    # Pairwise variant — annotator batch P
    │
    ├── tracking/
    │   ├── track_clip.py        # YOLOv8 + BoT-SORT batch tracker; produces CVAT-ready XML
    │   ├── track_F138.py        # Single-clip tracker (debugging / one-off use)
    │   └── fix_cvat_xmls.py     # Post-process CVAT XMLs (duplicate terminators, tag fixes)
    │
    ├── training/
    │   ├── train_tcn.py          # Leave-one-annotator-out CV on single-track NPZ files
    │   └── train_tcn_pairwise.py # Leave-one-annotator-out CV on pairwise NPZ files
    │
    └── visualization/
        ├── visualize_predictions.py          # Render TCN predictions on video (single-track)
        └── visualize_predictions_pairwise.py # Render TCN predictions on video (pairwise)
```

---

## Approach 1 — Pose-based Pipeline

Uses YOLOv8-pose to extract 17 COCO keypoints per person per frame, builds a
126-dimensional feature vector, and trains a Logistic Regression classifier.
Evaluation uses Leave-One-Video-Out cross-validation.

### Pipeline Overview

```
Raw videos + annotations.xml
        │
        ▼
[bounding_boxes.py]
    YOLOv8 + BoT-SORT tracking with CLAHE contrast enhancement,
    multi-scale inference, gap interpolation, optional Real-ESRGAN SR.
    Produces *_tracked.mp4 + *_annotations.xml for CVAT upload.
        │
        ▼
[module1_data_loader.py]
    Discovers data/video_data/, cross-references data.csv for fight
    frame ranges, parses CVAT XML annotations (3-way coordinate system
    auto-detection: CLEAN / FULL_VIDEO / TIGHT_XML).
        │
        ▼
[module2_augmenter.py]
    Per-frame augmentation applied at read time (training set only):
    horizontal flip, brightness/contrast jitter, HSV shift, Gaussian
    blur, additive noise.
        │
        ▼
[module3_pose_estimator.py]
    Crops each annotated bounding box, runs YOLOv8-pose, maps 17 COCO
    keypoints back to full-frame coordinates. Results cached to cache/.
        │
        ▼
[module4_classifier.py]
    Builds 126-d feature vector per person:
      [0:51]    Keypoint confidence (mean/std/max × 17 joints)
      [51:55]   Bounding box area and aspect ratio statistics
      [55:106]  Joint velocity (mean/std/max × 17 joints)
      [106:118] Body geometry (arm/leg extension, torso lean, upper-body area)
      [118:120] Social approach velocity toward nearest person
      [120:126] Intra-video percentile rank features (6-d)
    Classifier: StandardScaler → degree-2 PolynomialFeatures → L2 LogisticRegression
        │
        ▼
[module5_validation.py]
    Per-video prediction with threshold + per-video top-1 strategies.
    Accuracy, precision, recall, F1, confusion matrix, per-track breakdown.
    Optional prediction overlay video output.
```

### Quick Start

```bash
# Full run (uses pose cache if it exists)
python main.py

# Force re-run pose estimation
python main.py --skip-pose-cache

# Leave-One-Video-Out cross-validation
python main.py --loocv

# Save prediction overlay videos
python main.py --save-videos

# Subsample every 5th frame (faster)
python main.py --frame-step 5

# Set number of test videos
python main.py --n-test 3
```

### Data Layout

```
data/
├── data.csv            # columns: name, frame_start, frame_end
├── video_info.csv      # columns: name, frames  (optional)
└── video_data/
    └── F_9_1_2_0_0/
        ├── F_9_1_2_0_0.mp4
        └── annotations.xml
```

### Tracking (bounding_boxes.py)

```bash
# Basic — CLAHE + multi-scale + gap fill
python bounding_boxes.py \
    --video_dir /path/to/videos \
    --output_dir ./tracked_output

# With Real-ESRGAN 2× upscale
python bounding_boxes.py \
    --video_dir /path/to/videos \
    --output_dir ./tracked_output \
    --sr --sr_scale 2

# With Stage-1 fight frame labels
python bounding_boxes.py \
    --video_dir /path/to/videos \
    --output_dir ./tracked_output \
    --stage1_dir ./stage1_labels
```

---

## Approach 2 — Optical Flow Pipeline

Uses Farneback dense optical flow to build per-person motion feature sequences
and trains a Temporal Convolutional Network (TCN) classifier.
See `optical_flow/README.md` for full documentation.

### Pipeline Overview

```
Raw videos
        │
        ▼
[optical_flow/tracking/track_clip.py]
    YOLOv8 + BoT-SORT; produces *_clean.mp4 + *_cvat.xml for CVAT upload.
        │
        ▼
[optical_flow/tracking/fix_cvat_xmls.py]
    Fixes duplicate terminator boxes and malformed tags in CVAT exports.
        │
        ▼
[optical_flow/features/extract_flow_features.py]          ← single-track NPZ
[optical_flow/features/extract_pairwise_flow_features_dual.py] ← pairwise NPZ
    Farneback optical flow → (N, 30, 11) or (N, 30, 15) feature arrays.
        │
        ▼
[optical_flow/training/train_tcn.py]
[optical_flow/training/train_tcn_pairwise.py]
    TCN: Input(B,11,30) → TCNBlock×3 → GlobalAvgPool → Linear → logit
    Leave-one-annotator-out cross-validation.
        │
        ▼
[optical_flow/visualization/visualize_predictions.py]
[optical_flow/visualization/visualize_predictions_pairwise.py]
    Renders coloured bounding boxes on video.
```

### Quick Start

```bash
# 1 — Track clips
python optical_flow/tracking/track_clip.py \
    --batch \
    --video_dir /path/to/videos \
    --output_dir /path/to/outputs

# 2 — Fix CVAT XMLs
python optical_flow/tracking/fix_cvat_xmls.py /path/to/tracked_clips

# 3 — Extract features
python optical_flow/features/extract_flow_features.py \
    --videos_dir data/videos \
    --ann_dir    data/annotations \
    --output     data/outputs/optical_flow_features_annotator1.npz

# 4 — Train
python optical_flow/training/train_tcn.py \
    --data_dir data/outputs \
    --epochs 200

# 5 — Visualise
python optical_flow/visualization/visualize_predictions.py \
    --npz annotator1:data/outputs/optical_flow_features_annotator1.npz \
    --xml_dir    data/annotations \
    --videos_dir data/videos \
    --out_dir    data/outputs/viz
```

### Annotation Format Support

| Format | Track label | Role storage |
|--------|-------------|--------------|
| **A** | `"aggressor"` / `"non-aggressor"` on `<track>` | Track-level |
| **B** | `"person"` on `<track>` | Per-box `<attribute name="role">` |

`extract_pairwise_flow_features_dual.py` and the visualisation scripts
auto-detect the format per file.

### Feature Vector Layout

**Single-track `(N, 30, 11)`**

| Index | Feature |
|-------|---------|
| 0 | `mean_magnitude` |
| 1 | `peak_magnitude` |
| 2–9 | `dir_hist` (8 bins, 45° each, normalised) |
| 10 | `temporal_derivative` (frame-to-frame Δ mean_mag) |

**Pairwise `(N, 30, 15)`**

| Index | Feature |
|-------|---------|
| 0–10 | Own 11-feature vector |
| 11 | `delta_mean_mag` = own[0] − other[0] |
| 12 | `delta_peak_mag` = own[1] − other[1] |
| 13 | `delta_temporal_deriv` = own[10] − other[10] |
| 14 | `motion_leader` = 1 if own[0] > other[0] else 0 |

---

## Requirements

**Pose pipeline (root level):**
```bash
pip install ultralytics opencv-python numpy pandas torch scikit-learn joblib scipy
# Optional — Real-ESRGAN super-resolution:
pip install basicsr facexlib gfpgan realesrgan
```

**Optical flow pipeline:**
```bash
pip install -r optical_flow/requirements.txt
```

---

## Labels (colour key in visualisation)

| Class | Colour |
|-------|--------|
| Aggressor | 🔴 `#ED2939` |
| Non-aggressor | 🟡 `#FFFF00` |
