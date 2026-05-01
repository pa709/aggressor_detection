#!/usr/bin/env python3
"""
bounding_boxes.py
-----------------
Upgraded pipeline for low-quality surveillance video annotation.

Improvements over v1:
  1. CLAHE contrast enhancement  per frame (LAB L-channel)
  2. Optional Real-ESRGAN super-resolution upscale (--sr flag)
  3. Multi-scale inference  (native + 1.5x crop mosaic) merged with WBF
  4. BoT-SORT with tuned track_buffer / low detection threshold
  5. Post-hoc gap interpolation  for missing bounding boxes (≤ MAX_GAP frames)
  6. Background-subtraction ROI hint  (MOG2, optional mask overlay)

Usage:
    # Basic (CLAHE + multi-scale + gap fill):
    python3 bounding_boxes.py \
        --video_dir /path/to/videos \
        --output_dir ./tracked_output

    # With Real-ESRGAN 2x upscale (slower, best quality):
    python3 bounding_boxes.py \
        --video_dir /path/to/videos \
        --output_dir ./tracked_output \
        --sr --sr_scale 2

    # Show MOG2 background-subtraction mask overlay:
    python3 bounding_boxes.py \
        --video_dir /path/to/videos \
        --output_dir ./tracked_output \
        --show_bg_mask

    # Stage-1 (frame 0/1) + full-video tracking (consistent track IDs) + CVAT 1.1 .xml
    # on violence frames only:
    python3 bounding_boxes.py --video_dir ./videos --output_dir ./out \
        --stage1_dir ./stage1_dir
    #   stage1_dir: one of per video
    #     {stem}.json  —  [0,0,0,1,1,1,0, ...]  (same as consolidate_ubi_gt --export_stage1_dir)
    #     {stem}.csv   —  Stage-1 side-by-side eval CSV (e.g. fight_0002_stage2_..._thr_0.70.csv
    #                      with columns frame_idx, created_fight_label, ...); use
    #     --stage1_label_column created_fight_label  (default)
    #   Or: {stem}_*.csv  (first match) if the file has a long suffix.
    #   Or: {id}_stage2_side_by_side_labels_thr_0.70.csv  — if the only file matching
    #       *_stage2_side_by_side_labels_thr_*.csv in the folder, it is used for any
    #       video stem (e.g. Fighting002_x264.mp4 + fight_0002_....csv). Multiple such
    #       CSVs require --stage1_index to map each video stem to a file.

Requirements:
    pip install ultralytics lapx opencv-python numpy
    pip install basicsr facexlib gfpgan          # only needed with --sr
    pip install realesrgan                        # only needed with --sr
"""

import argparse
import csv
import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  GROUND TRUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_fps(vid_name: str, gt_fps: float, video_dir: Path,
                tolerance: float = 0.1) -> tuple:
    """
    Return (fps_to_use, source) where source is 'gt' or 'opencv'.

    OpenCV fps is used as authoritative when it differs from GT fps by more
    than `tolerance`. Handles two known failure modes in the NTU dataset:
      - fight_0358-0376: OpenCV reads 29.0 vs GT 29.97  (delta ~0.97)
      - fight_0246:      OpenCV reads 20.0 vs GT 23.976 (delta ~3.98)
    Falls back to GT fps if file is not found in video_dir.
    """
    for ext in (".mp4", ".webm"):
        p = video_dir / f"{vid_name}{ext}"
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            cv_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if cv_fps > 0 and abs(cv_fps - gt_fps) >= tolerance:
                return cv_fps, "opencv"
            return gt_fps, "gt"
    return gt_fps, "gt"   # file not in this dir — GT is best guess


def _groundtruth_file_kind(gt_path: Path) -> str:
    """
    Return 'json' (NTU-style groundtruth.json) or 'ubi_csv' (consolidate_ubi_gt.py output).
    Extension can be wrong (e.g. CSV saved as .json); content is sniffed.
    """
    with open(gt_path, encoding="utf-8-sig") as f:
        head = f.read(4096)
    if not head.strip():
        raise ValueError(f"Ground-truth file is empty: {gt_path}")
    first = head.lstrip().splitlines()[0] if head else ""
    fl = first.lstrip()
    if fl.startswith("{") or fl.startswith("["):
        return "json"
    low = first.lower()
    if "video_name" in low and "segment_idx" in low:
        return "ubi_csv"
    raise ValueError(
        f"Unrecognized ground-truth format in {gt_path}: "
        "expected JSON object/array or UBI CSV (header with video_name, segment_idx, ...)."
    )


def load_groundtruth_ubi_csv(gt_path: Path, video_dir: Path = None,
                             fps_tolerance: float = 0.1) -> dict:
    """
    Load consolidate_ubi_gt.py CSV: one row per fight segment.
    Builds per-video fight frame sets from start_frame..end_frame (inclusive).
    """
    by_frames: dict = defaultdict(set)
    gt_fps_map: dict = {}

    with open(gt_path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        fields = set(reader.fieldnames or ())
        need = {"video_name", "fps", "segment_idx", "start_frame", "end_frame"}
        if not need <= fields:
            raise ValueError(
                f"UBI ground-truth CSV missing columns {sorted(need - fields)}; "
                f"found {sorted(fields)}"
            )
        for row in reader:
            vid = (row.get("video_name") or "").strip()
            if not vid:
                continue
            try:
                seg_idx = int(str(row.get("segment_idx", "")).strip())
            except ValueError:
                continue
            if seg_idx < 0:
                continue
            sf_s = (row.get("start_frame") or "").strip()
            ef_s = (row.get("end_frame") or "").strip()
            if not sf_s or not ef_s:
                continue
            start_f = int(sf_s)
            end_f = int(ef_s)
            by_frames[vid].update(range(start_f, end_f + 1))
            if vid not in gt_fps_map:
                try:
                    gt_fps_map[vid] = float(row["fps"])
                except (KeyError, ValueError):
                    gt_fps_map[vid] = 30.0

    gt = {}
    mismatches = []

    for vid_name, fight_frames in by_frames.items():
        gt_fps = gt_fps_map.get(vid_name, 30.0)
        if video_dir is not None:
            fps, src = resolve_fps(vid_name, gt_fps, video_dir, fps_tolerance)
            if src == "opencv":
                mismatches.append((vid_name, gt_fps, fps))
        else:
            fps, src = gt_fps, "gt"

        ff = set(fight_frames)
        if src == "opencv" and abs(fps - gt_fps) >= fps_tolerance:
            ff = {int(round(f / gt_fps * fps)) for f in fight_frames}

        gt[vid_name] = {"fps": fps, "fps_source": src, "fight_frames": ff}

    if mismatches:
        print(f"  [fps] {len(mismatches)} video(s) using OpenCV fps over GT fps:")
        for name, gt_f, cv_f in mismatches:
            print(f"    {name}  GT={gt_f:.3f} -> CV={cv_f:.3f}  "
                  f"delta={abs(cv_f - gt_f):.3f}")

    return gt


def load_groundtruth(gt_path: Path, video_dir: Path = None,
                     fps_tolerance: float = 0.1) -> dict:
    """
    Load ground truth from NTU-style groundtruth.json or UBI consolidated CSV
    (consolidate_ubi_gt.py). Returns a lookup keyed by video stem:
    { 'F_215_...': {'fps': 30.0, 'fps_source': 'gt',
                    'fight_frames': set([45,46,...])}, ... }

    JSON: segments are [start_sec, end_sec] per Fight annotation.
    CSV: rows give start_frame/end_frame per segment (see consolidate_ubi_gt.py).

    Frame indices use the most accurate fps available when video_dir is set
    (OpenCV vs GT mismatch handling); see resolve_fps.
    """
    kind = _groundtruth_file_kind(gt_path)
    if kind == "ubi_csv":
        return load_groundtruth_ubi_csv(gt_path, video_dir, fps_tolerance)

    with open(gt_path, encoding="utf-8-sig") as f:
        raw = json.load(f)

    db = raw.get("database", {})
    gt = {}
    mismatches = []

    for vid_name, info in db.items():
        gt_fps = info.get("frame_rate", 30.0) or 30.0

        if video_dir is not None:
            fps, src = resolve_fps(vid_name, gt_fps, video_dir, fps_tolerance)
            if src == "opencv":
                mismatches.append((vid_name, gt_fps, fps))
        else:
            fps, src = gt_fps, "gt"

        fight_frames: set = set()
        for ann in info.get("annotations", []):
            if ann.get("label") != "Fight":
                continue
            start_sec, end_sec = ann["segment"]
            start_f = max(0, int(np.floor(start_sec * fps)))
            end_f   = int(np.ceil(end_sec * fps))
            fight_frames.update(range(start_f, end_f + 1))

        gt[vid_name] = {"fps": fps, "fps_source": src,
                        "fight_frames": fight_frames}

    if mismatches:
        print(f"  [fps] {len(mismatches)} video(s) using OpenCV fps over GT fps:")
        for name, gt_f, cv_f in mismatches:
            print(f"    {name}  GT={gt_f:.3f} -> CV={cv_f:.3f}  "
                  f"delta={abs(cv_f - gt_f):.3f}")

    return gt


# ═══════════════════════════════════════════════════════════════════════════════
# 0b.  STAGE 1 (frame-level 0/1) + CVAT 1.1 export
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_csv_fieldnames(row: dict) -> dict:
    return {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}


def load_stage1_violence_csv(
    path: Path,
    *,
    label_column: str = "created_fight_label",
) -> list:
    """
    Load Stage-1 per-frame 0/1 from a side-by-side export CSV, e.g. sample
    *stage2*side_by_side*labels*_thr_*.csv with columns:
      frame_idx, gt_fight_label, created_fight_label, raw_created_fight_label,
      fight_prob, smoothed_fight_prob, label_match

    For full-video YOLO + CVAT, use created_fight_label (thresholded binary
    used as Stage-2 box mask). For raw unsmoothed binary, set --stage1_label_column
    raw_created_fight_label.
    """
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"Stage-1 CSV is empty: {path}")
        rows = [_normalize_csv_fieldnames(r) for r in reader if any(r.values())]
    if not rows:
        return []

    def find_col(allowed: list[str], sample: dict) -> str:
        lower_map = {k.lower(): k for k in sample.keys()}
        for name in allowed:
            if name in sample:
                return name
            ln = name.lower()
            if ln in lower_map:
                return lower_map[ln]
        raise ValueError(
            f"Stage-1 CSV {path} has no label column. Tried {allowed!r}. "
            f"Columns: {list(sample.keys())}"
        )

    col_y = find_col(
        [label_column, "created_fight_label", "raw_created_fight_label", "violence", "y", "label"],
        rows[0],
    )
    # Optional frame index: build dense [0..max] (sparse / missing -> 0)
    frame_key = None
    for candidate in ("frame_idx", "frame", "index"):
        if candidate in rows[0] or candidate.lower() in {k.lower() for k in rows[0]}:
            # resolve actual key
            for k in rows[0].keys():
                if k.lower() == candidate.lower():
                    frame_key = k
                    break
            if frame_key:
                break

    if frame_key:
        by_f: dict[int, int] = {}
        for r in rows:
            try:
                fi = int(float(r[frame_key]))
            except (TypeError, ValueError):
                continue
            try:
                by_f[fi] = int(float(r[col_y])) & 1
            except (TypeError, ValueError) as e:
                raise ValueError(f"Bad label in {path} row frame={fi}: {r!r}") from e
        if not by_f:
            return []
        max_f = max(by_f.keys())
        out = [0] * (max_f + 1)
        for i, v in by_f.items():
            if 0 <= i <= max_f:
                out[i] = v
        print(
            f"  [stage1] {path.name}: {len(out)} frame label(s) from column "
            f"'{col_y}' (CSV, indexed by {frame_key})"
        )
        return out

    out = []
    for r in rows:
        try:
            out.append(int(float(r[col_y])) & 1)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bad label row in {path}: {r!r}") from e
    print(
        f"  [stage1] {path.name}: {len(out)} frame label(s) from column '{col_y}' (CSV, sequential rows)"
    )
    return out


def load_stage1_violence(
    path: Path,
    *,
    label_column: str = "created_fight_label",
) -> list:
    """
    Load Stage-1 per-frame list: [0,0,0,1,1,1,0] (1 = violence).

    - .json: bare array, or { "frames" | "violence" | "labels" | "y": [...] }.
    - .csv: side-by-side export with frame_idx + label column (see load_stage1_violence_csv).
    """
    if path.suffix.lower() == ".csv":
        return load_stage1_violence_csv(path, label_column=label_column)

    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [int(x) for x in raw]
    for key in ("frames", "violence", "labels", "y"):
        if key in raw and isinstance(raw[key], list):
            return [int(x) for x in raw[key]]
    raise ValueError(
        f"Stage-1 file must be a JSON array or an object with 'frames' / 'violence' list: {path}"
    )


def load_stage1_index(
    path: Path,
    *,
    label_column: str = "created_fight_label",
) -> dict:
    """
    Load JSON: { 'VideoStem': [0,0,1,1,0] | path/to/stage1.json | path/to/stage1.csv, ... }.
    """
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Stage-1 index must be a JSON object mapping video name -> list or file path: {path}"
        )
    out: dict = {}
    for k, v in raw.items():
        key = str(k).strip()
        if isinstance(v, list):
            out[key] = [int(x) for x in v]
        elif isinstance(v, str):
            ref = Path(v)
            if not ref.is_file():
                ref = path.parent / v
            if not ref.is_file():
                raise ValueError(
                    f"Stage-1 index entry {key!r}: not a list and not a file: {v}"
                )
            if ref.suffix.lower() not in (".json", ".csv"):
                raise ValueError(
                    f"Stage-1 index entry {key!r}: file must be .json or .csv: {ref}"
                )
            out[key] = load_stage1_violence(ref, label_column=label_column)
        else:
            raise ValueError(
                f"Stage-1 index entry {key!r} must be a list or a path string, got {type(v)}"
            )
    return out


def resolve_stage1_path(
    stage1_dir: Path,
    stem: str,
) -> Path | None:
    """
    Find Stage-1 file for a video stem:

    1) {stem}.json / {stem}.csv
    2) {stem}_*.csv  (e.g. fight_0002_…csv for fight_0002.mp4)
    3) Canonical export name (Stage-1 side-by-side eval), when video stem
       does not match the clip id in the filename:
         *_stage2_side_by_side_labels_thr_*.csv
       If exactly one such file exists in stage1_dir, use it. If several,
       prefer one whose name starts with ``{stem}_``; otherwise return None
       and use --stage1_index to map {stem!r} to a file.
    """
    for name in (f"{stem}.json", f"{stem}.csv"):
        p = stage1_dir / name
        if p.is_file():
            return p
    matches = sorted(stage1_dir.glob(f"{stem}_*.csv"))
    if matches:
        return matches[0]

    side_by = sorted(
        stage1_dir.glob("*_stage2_side_by_side_labels_thr_*.csv")
    )
    if not side_by:
        return None
    with_stem = [p for p in side_by if p.name.startswith(f"{stem}_")]
    if len(with_stem) == 1:
        return with_stem[0]
    if len(with_stem) > 1:
        with_stem.sort(key=lambda p: p.name)
        return with_stem[0]
    if len(side_by) == 1:
        p = side_by[0]
        print(
            f"  [stage1] no {stem}_*.csv; using sole export in folder: {p.name}"
        )
        return p
    print(
        f"  [stage1] multiple *_stage2_side_by_side_labels_thr_*.csv in "
        f"{stage1_dir} — map {stem!r} in --stage1_index to the correct file.",
        file=sys.stderr,
    )
    return None


def align_stage1_to_video(n_frames: int, stage1: list) -> list:
    """
    Return a length-n_frames list of 0/1, padding with 0 or truncating to match
    the video (OpenCV frame count).
    """
    s = [int(x) for x in stage1]
    if len(s) < n_frames:
        old = len(s)
        s = s + [0] * (n_frames - len(s))
        print(
            f"  [stage1] padded {n_frames - old} trailing zeros (list had {old} entries, video {n_frames} frames)"
        )
    elif len(s) > n_frames:
        print(f"  [stage1] truncated list from {len(s)} to {n_frames} frames to match video")
        s = s[:n_frames]
    return s


def write_cvat_11_video_xml(
    out_path: Path,
    *,
    width: int,
    height: int,
    n_frames: int,
    label: str,
    per_track: dict,  # track_id (int) -> list of (frame, x1, y1, x2, y2)
) -> None:
    """
    Write CVAT for video 1.1 (annotations). Import in CVAT: 'CVAT 1.1' format.
    """
    root = ET.Element("annotations")
    el_ver = ET.SubElement(root, "version")
    el_ver.text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = out_path.stem
    ET.SubElement(task, "size").text = str(n_frames)
    ET.SubElement(task, "mode").text = "annotation"

    labels_el = ET.SubElement(task, "labels")
    lab = ET.SubElement(labels_el, "label")
    ET.SubElement(lab, "name").text = label

    orig = ET.SubElement(task, "original_size")
    ET.SubElement(orig, "width").text = str(width)
    ET.SubElement(orig, "height").text = str(height)

    for tid in sorted(per_track.keys()):
        boxes = per_track[tid]
        if not boxes:
            continue
        tr = ET.SubElement(
            root,
            "track",
            {
                "id": str(tid),
                "label": label,
                "source": "yolo_botsort",
            },
        )
        for frame, x1, y1, x2, y2 in boxes:
            ET.SubElement(
                tr,
                "box",
                {
                    "frame": str(int(frame)),
                    "keyframe": "1",
                    "outside": "0",
                    "occluded": "0",
                    "xtl": f"{float(x1):.2f}",
                    "ytl": f"{float(y1):.2f}",
                    "xbr": f"{float(x2):.2f}",
                    "ybr": f"{float(y2):.2f}",
                    "z_order": "0",
                },
            )

    tree = ET.ElementTree(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)


# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 221,  51), (100, 221,  23),
    ( 24, 201, 114), ( 24, 201, 201), ( 23, 114, 255), (125,  12, 255),
    (255,  12, 240), (255, 120, 120), (198, 255, 120), (120, 255, 198),
    (120, 198, 255), (198, 120, 255), (255, 198, 120), (180, 180, 180),
]

def get_color(track_id: int):
    return PALETTE[int(track_id) % len(PALETTE)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_clahe(frame: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    """
    Apply CLAHE on the L-channel of LAB colour space.
    Improves local contrast without over-brightening or colour shift.
    clip_limit=3.0 is a good starting point; raise to 4-5 for very dark footage.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_grid, tile_grid))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def build_sr_model(scale: int = 2):
    """
    Load Real-ESRGAN upscaler.
    Requires:  pip install realesrgan basicsr
    Model weights are auto-downloaded on first call (~67 MB for x2plus).
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model_name = "RealESRGAN_x2plus" if scale == 2 else "RealESRGAN_x4plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        sr = RealESRGANer(
            scale=scale,
            model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_name}.pth",
            model=model,
            tile=256,          # tile inference to limit VRAM
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        print(f"  Real-ESRGAN {scale}x loaded.")
        return sr
    except ImportError:
        print("[WARN] realesrgan not installed — skipping SR. Run: pip install realesrgan basicsr")
        return None


def apply_sr(sr_model, frame: np.ndarray) -> np.ndarray:
    if sr_model is None:
        return frame
    out, _ = sr_model.enhance(frame, outscale=sr_model.scale)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MULTI-SCALE INFERENCE + WEIGHTED BOX FUSION
# ═══════════════════════════════════════════════════════════════════════════════

def iou(b1, b2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def simple_wbf(boxes_list, scores_list, iou_thr=0.45):
    """
    Lightweight Weighted Box Fusion across multiple scale predictions.
    boxes_list : list of Nx4 arrays  (x1,y1,x2,y2 in pixel coords)
    scores_list: list of N floats
    Returns merged (boxes, scores).
    """
    if not boxes_list:
        return np.empty((0, 4), dtype=float), np.array([])

    all_boxes  = np.concatenate(boxes_list,  axis=0).tolist()
    all_scores = np.concatenate(scores_list, axis=0).tolist()

    order   = sorted(range(len(all_scores)), key=lambda i: -all_scores[i])
    used    = [False] * len(all_boxes)
    merged_boxes, merged_scores = [], []

    for i in order:
        if used[i]:
            continue
        cluster_b = [all_boxes[i]]
        cluster_s = [all_scores[i]]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            if iou(all_boxes[i], all_boxes[j]) >= iou_thr:
                cluster_b.append(all_boxes[j])
                cluster_s.append(all_scores[j])
                used[j] = True
        w = np.array(cluster_s)
        w = w / w.sum()
        fused = (np.array(cluster_b) * w[:, None]).sum(axis=0)
        merged_boxes.append(fused)
        merged_scores.append(float(np.mean(cluster_s)))

    return np.array(merged_boxes), np.array(merged_scores)


def multiscale_detect(model, frame: np.ndarray,
                      conf_thr: float = 0.20,
                      scales: tuple = (1.0, 1.5)):
    """
    Run YOLOv8 at multiple resolutions and fuse results with WBF.
    Returns list of (x1,y1,x2,y2,conf) tuples (pixel coords of original frame).
    """
    H, W = frame.shape[:2]
    all_boxes, all_scores = [], []

    for s in scales:
        if s == 1.0:
            inp = frame
            sx, sy = 1.0, 1.0
        else:
            new_w, new_h = int(W * s), int(H * s)
            inp = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            sx, sy = 1.0 / s, 1.0 / s

        res = model.predict(inp, classes=[0], conf=conf_thr,
                            iou=0.40, verbose=False, agnostic_nms=True)
        b = res[0].boxes
        if b is None or len(b) == 0:
            continue
        xyxy  = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()

        # Scale boxes back to original resolution
        xyxy[:, [0, 2]] *= sx
        xyxy[:, [1, 3]] *= sy
        all_boxes.append(xyxy)
        all_scores.append(confs)

    merged_boxes, merged_scores = simple_wbf(all_boxes, all_scores, iou_thr=0.45)

    detections = []
    for box, sc in zip(merged_boxes, merged_scores):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        detections.append((x1, y1, x2, y2, float(sc)))
    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BOT-SORT TRACKER WRAPPER  (via ultralytics tracker API)
# ═══════════════════════════════════════════════════════════════════════════════

def write_botsort_cfg(path: Path):
    """
    Write a complete, tuned BoT-SORT config YAML.

    Every field that ultralytics BoT-SORT reads via IterableSimpleNamespace
    must be present — any missing key raises AttributeError at runtime.
    Fields confirmed from the official ultralytics botsort.yaml + bot_sort.py source.

    Tuning rationale for low-quality / motion-blur surveillance footage:
      track_high_thresh  0.25   catch blurry low-conf detections in first stage
      track_low_thresh   0.10   recover nearly-lost tracks in second stage
      new_track_thresh   0.30   slightly above low thresh to avoid noise tracks
      track_buffer       90     hold lost tracks ~3 s @ 30 fps (survive blur bursts)
      match_thresh       0.80   keep default; tightening can drop valid re-assoc
      fuse_score         True   fuse detection confidence with IoU cost — stabilises
                                weak detections in blurry frames (REQUIRED field)
      gmc_method         sparseOptFlow   fast GMC for static/slow-pan cameras
                                         use 'orb' or 'sift' for shaky cameras
      proximity_thresh   0.50   spatial gate for appearance matching
      appearance_thresh  0.25   cosine similarity gate for re-ID
      with_reid          False  no external Re-ID model; set True + add
                                reid_weights: osnet_x0_25_market.pt to enable
    """
    cfg = """\
tracker_type: botsort

# Detection thresholds
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.30

# Track lifetime  (90 frames = 3 s @ 30 fps)
track_buffer: 90

# Association
match_thresh: 0.80
fuse_score: True

# Global Motion Compensation
gmc_method: sparseOptFlow

# Re-ID (appearance)
proximity_thresh: 0.50
appearance_thresh: 0.25
with_reid: False
"""
    path.write_text(cfg, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  POST-HOC GAP INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_gaps(raw_tracks: dict, max_gap: int = 15) -> dict:
    """
    For each track, linearly interpolate bounding boxes across missing frames
    where the gap is ≤ max_gap frames.

    raw_tracks: { track_id: { frame_idx: (x1,y1,x2,y2,conf) } }
    Returns the same structure with gap frames filled in.
    """
    filled = {}
    for tid, frame_dict in raw_tracks.items():
        filled[tid] = dict(frame_dict)
        frames = sorted(frame_dict.keys())

        for i in range(len(frames) - 1):
            f_a, f_b = frames[i], frames[i + 1]
            gap = f_b - f_a - 1
            if gap < 1 or gap > max_gap:
                continue

            b_a = np.array(frame_dict[f_a][:4], dtype=float)
            b_b = np.array(frame_dict[f_b][:4], dtype=float)
            avg_conf = (frame_dict[f_a][4] + frame_dict[f_b][4]) / 2.0

            for step in range(1, gap + 1):
                t = step / (gap + 1)
                interp_box = ((1 - t) * b_a + t * b_b).astype(int)
                filled[tid][f_a + step] = (
                    interp_box[0], interp_box[1],
                    interp_box[2], interp_box[3],
                    avg_conf,
                )
    return filled


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_box(frame, x1, y1, x2, y2, track_id, conf, interpolated=False):
    color = get_color(track_id)
    thickness = 1 if interpolated else 2
    style = cv2.LINE_4 if interpolated else cv2.LINE_AA

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, style)

    prefix = "~" if interpolated else ""   # ~ marks interpolated boxes
    label  = f"{prefix}ID:{track_id} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN VIDEO PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(model, tracker_cfg: Path, video_path: Path, output_path: Path,
                  sr_model=None, conf_thr: float = 0.20,
                  scales: tuple = (1.0, 1.5),
                  max_gap: int = 15,
                  show_bg_mask: bool = False,
                  clahe_clip: float = 3.0,
                  fight_frames: set = None,
                  stage1_violence: list = None,
                  xml_path: Path = None,
                  cvat_label: str = "person") -> dict:
    """
    Full pipeline: preprocess → BoT-SORT track → gap interpolation →
    optional CVAT 1.1 XML + annotated video.

    stage1_violence:
        Per-frame 0/1 from Stage 1 (1 = violence). When set (any length; aligned
        to video frame count in-code):
        - YOLO+BoT-SORT run on *every* frame (consistent track IDs over time).
        - Bounding boxes and labels appear only on frames with value 1.
        - A compact CVAT 1.1 for-video .xml is written to xml_path (violence
          frames only) when xml_path is not None.
        - Mutually independent from fight_frames: do not use --gt and Stage 1
          on the same run unless you know what you are doing; Stage 1 takes over.

    fight_frames (legacy, --gt):
        When set and stage1_violence is None:
        - Tracking runs *only* on these frames (faster, but IDs are not
          continuous across the full clip).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path.name}")
        return {}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = width  * (sr_model.scale if sr_model else 1)
    out_h = height * (sr_model.scale if sr_model else 1)

    use_stage1 = stage1_violence is not None
    vlist: list = []
    violence_set: set = set()
    if use_stage1:
        vlist = align_stage1_to_video(total, stage1_violence)
        violence_set = {i for i, v in enumerate(vlist) if int(v) == 1}
        n_v = len(violence_set)
        print(
            f"  Stage-1: {n_v} violence frame(s) of {total} "
            f"(track on all {total} frames for consistent IDs)"
        )

    infer_fight_only = (
        not use_stage1
        and fight_frames is not None
        and len(fight_frames) > 0
    )

    gt_active = (not use_stage1) and fight_frames is not None and len(fight_frames) > 0
    gt_frame_count = len(fight_frames) if gt_active else (len(violence_set) if use_stage1 else total)
    if use_stage1:
        mode_str = f"{total} frames (track all) → boxes on {len(violence_set)} stage1=1"
    elif infer_fight_only:
        mode_str = f"{len(fight_frames)} GT fight frames"
    else:
        mode_str = f"{total} frames"
    print(f"  Pass 1/2 — detect + track ({mode_str} @ {fps:.1f}fps) ...", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # Background subtractor (MOG2) — used as a visual hint / mask overlay
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=50, detectShadows=False)

    # ── Pass 1: collect raw detections with tracking ──────────────────────────
    raw_frames   = []     # preprocessed (+ SR) frames, kept in RAM
    raw_tracks   = defaultdict(dict)
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_pre = apply_clahe(frame, clip_limit=clahe_clip)
        frame_pre = apply_sr(sr_model, frame_pre)

        _ = bg_sub.apply(frame_pre)
        raw_frames.append(frame_pre)

        run_inf = (not infer_fight_only) or (frame_idx in fight_frames)
        if run_inf:
            track_res = model.track(
                frame_pre,
                persist=True,
                classes=[0],
                conf=conf_thr,
                iou=0.40,
                tracker=str(tracker_cfg),
                verbose=False,
            )
            boxes = track_res[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy      = boxes.xyxy.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)
                confs     = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), tid, conf in zip(xyxy, track_ids, confs):
                    raw_tracks[int(tid)][frame_idx] = (x1, y1, x2, y2, float(conf))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    {frame_idx}/{total} frames done", flush=True)

    cap.release()

    # ── Gap interpolation (Pass 1.5) ─────────────────────────────────────────
    print(f"  Interpolating gaps (max_gap={max_gap} frames) ...", flush=True)
    filled_tracks = interpolate_gaps(dict(raw_tracks), max_gap=max_gap)

    # frame_idx -> [(tid,x1,y1,x2,y2,conf,interp)]
    frame_boxes = defaultdict(list)
    for tid, fdict in filled_tracks.items():
        orig_f = set(raw_tracks.get(tid, {}).keys())
        for fidx, (x1, y1, x2, y2, conf) in fdict.items():
            is_interp = fidx not in orig_f
            if use_stage1 and fidx not in violence_set:
                continue
            frame_boxes[fidx].append((tid, x1, y1, x2, y2, conf, is_interp))

    if use_stage1 and xml_path is not None:
        per_export = defaultdict(list)  # tid -> [(f, x1, y1, x2, y2)]
        for tid, fdict in filled_tracks.items():
            for fidx, (x1, y1, x2, y2, _conf) in fdict.items():
                if fidx in violence_set:
                    per_export[tid].append((fidx, x1, y1, x2, y2))
        for tid in per_export:
            per_export[tid].sort(key=lambda t: t[0])
        write_cvat_11_video_xml(
            xml_path,
            width=out_w,
            height=out_h,
            n_frames=len(raw_frames) if raw_frames else total,
            label=cvat_label,
            per_track=dict(per_export),
        )
        print(f"  CVAT 1.1 XML: {xml_path}", flush=True)

    # Pass 2: render — for Stage 1, only draw on violence frames (box list already filtered)
    print(f"  Pass 2/2 — rendering annotated video ...", flush=True)
    for fidx, frame_pre in enumerate(raw_frames):
        out_frame = frame_pre.copy()

        if show_bg_mask:
            fg_mask = bg_sub.apply(frame_pre, learningRate=0)
            tint    = np.zeros_like(out_frame)
            tint[fg_mask > 0] = (0, 80, 0)
            out_frame = cv2.addWeighted(out_frame, 1.0, tint, 0.3, 0)

        if not use_stage1 or fidx in violence_set:
            for tid, x1, y1, x2, y2, conf, is_interp in frame_boxes.get(fidx, []):
                draw_box(out_frame, x1, y1, x2, y2, tid, conf, interpolated=is_interp)

        h, w = out_frame.shape[:2]

        if use_stage1 and fidx in violence_set:
            banner_h = 28
            overlay = out_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 160), -1)
            out_frame = cv2.addWeighted(overlay, 0.55, out_frame, 0.45, 0)
            cv2.putText(out_frame, "STAGE1: violence", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        elif gt_active and fidx in fight_frames:
            banner_h = 28
            overlay = out_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
            out_frame = cv2.addWeighted(overlay, 0.55, out_frame, 0.45, 0)
            cv2.putText(out_frame, "GT: FIGHT", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(out_frame, f"Frame {fidx}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(out_frame)

        if (fidx + 1) % 100 == 0:
            print(f"    rendered {fidx+1}/{len(raw_frames)} frames", flush=True)

    writer.release()

    # Summary (per full track, not only violence; eval scripts may want both)
    track_summary = {}
    for tid, fdict in filled_tracks.items():
        orig_frames = sorted(raw_tracks[tid].keys())
        all_frames  = sorted(fdict.keys())
        track_summary[tid] = {
            "first_frame":         all_frames[0]  if all_frames  else -1,
            "last_frame":          all_frames[-1] if all_frames  else -1,
            "frames_seen":         len(orig_frames),
            "frames_interpolated": len(all_frames) - len(orig_frames),
        }

    return track_summary


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Low-quality surveillance video annotation: CLAHE + multi-scale YOLOv8 + BoT-SORT + gap fill"
    )
    parser.add_argument("--video_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--num_videos",  type=int,   default=10)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--model",       default="yolov8s.pt")
    parser.add_argument("--gt",          default=None,
                        help="Path to ground truth: NTU-style groundtruth.json, or "
                             "UBI consolidated CSV from consolidate_ubi_gt.py (e.g. "
                             "ubi_ground_truth.csv). When set, detection runs only on "
                             "GT fight frames; a red GT:FIGHT banner marks those frames.")
    parser.add_argument(
        "--stage1_index",
        default=None,
        help="JSON file: { 'VideoStem': [0,0,1,1,0], ... } — Stage 1 frame-level "
        "violence (1=violence). When set, YOLO+track runs on *all* frames for "
        "consistent IDs; boxes + CVAT export only on 1-frames. Ignores --gt for those videos.",
    )
    parser.add_argument(
        "--stage1_dir",
        default=None,
        help="Directory of Stage-1 files: {stem}.json / {stem}.csv / {stem}_*.csv, or "
        "a unique *_stage2_side_by_side_labels_thr_*.csv (e.g. "
        "fight_0002_stage2_side_by_side_labels_thr_0.70.csv for mismatched video names). "
        "If several such CSVs exist, set --stage1_index. "
        "Used if --stage1_index has no key for a video, or as the only source.",
    )
    parser.add_argument(
        "--stage1_label_column",
        default="created_fight_label",
        help="For Stage-1 CSV: column to use as 0/1 violence label for Stage-2. "
        "Default created_fight_label (matches *stage2*side_by_side* CSV). "
        "Alternatives: raw_created_fight_label, gt_fight_label.",
    )
    parser.add_argument(
        "--cvat_label",
        default="person",
        help="Object label in generated CVAT 1.1 .xml (default: person).",
    )
    parser.add_argument(
        "--no_cvat_xml",
        action="store_true",
        help="With Stage 1, skip writing {stem}_annotations.xml (video still obeys Stage 1).",
    )

    # Preprocessing
    parser.add_argument("--sr",          action="store_true",  help="Enable Real-ESRGAN super-resolution")
    parser.add_argument("--sr_scale",    type=int, default=2,  choices=[2, 4], help="SR upscale factor")
    parser.add_argument("--clahe_clip",  type=float, default=3.0, help="CLAHE clip limit (higher=more contrast)")
    parser.add_argument("--show_bg_mask",action="store_true",  help="Overlay MOG2 foreground mask")

    # Detection
    parser.add_argument("--conf",        type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--scales",      type=float, nargs="+", default=[1.0, 1.5],
                        help="Inference scales for multi-scale detection (e.g. 1.0 1.5 2.0)")

    # Tracking / gap fill
    parser.add_argument("--max_gap",     type=int, default=15,
                        help="Max frame gap to linearly interpolate bounding boxes")

    args = parser.parse_args()

    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write tuned BoT-SORT config
    tracker_cfg = output_dir / "botsort_tuned.yaml"
    write_botsort_cfg(tracker_cfg)
    print(f"BoT-SORT config written to: {tracker_cfg}")

    def _video_stem_ok(stem: str) -> bool:
        # UBI/NTU: F_... — sample: fight_... — e.g. Fighting002_x264.mp4
        return (
            stem.startswith("F_")
            or stem.startswith("fight_")
            or stem.startswith("Fighting")
        )

    all_videos = sorted([
        p for p in video_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".webm"}
        and _video_stem_ok(p.stem)
    ])

    if not all_videos:
        print(
            f"[ERROR] No F_*, fight_*, or Fighting*.mp4 / .webm files found in {video_dir}",
            file=sys.stderr,
        )
        return

    print(f"Found {len(all_videos)} videos.")
    random.seed(args.seed)
    selected = sorted(random.sample(all_videos, min(args.num_videos, len(all_videos))))
    print(f"Selected {len(selected)} videos (seed={args.seed}):\n" +
          "\n".join(f"  {v.name}" for v in selected) + "\n")

    # Load ground truth
    gt_db = {}
    if args.gt:
        print(f"Loading ground truth: {args.gt}")
        gt_db = load_groundtruth(Path(args.gt), video_dir=video_dir)
        total_fight_videos = sum(1 for v in gt_db.values() if v["fight_frames"])
        print(f"  {len(gt_db)} videos in GT, {total_fight_videos} with Fight annotations.\n")

    stage1_index_map: dict = {}
    if args.stage1_index:
        stage1_index_map = load_stage1_index(
            Path(args.stage1_index), label_column=args.stage1_label_column
        )
        print(f"Stage-1 index: {len(stage1_index_map)} video(s) in {args.stage1_index}\n")
    stage1_dir = Path(args.stage1_dir) if args.stage1_dir else None
    s1col = args.stage1_label_column

    def resolve_stage1_list(stem: str) -> list | None:
        if stem in stage1_index_map:
            return stage1_index_map[stem]
        if stage1_dir is not None:
            p = resolve_stage1_path(stage1_dir, stem)
            if p is not None:
                return load_stage1_violence(p, label_column=s1col)
        return None

    if not (args.stage1_index or args.stage1_dir) and (args.no_cvat_xml):
        print("[WARN] --no_cvat_xml has no effect without Stage-1 input.", file=sys.stderr)

    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded.\n")

    # Optionally load SR model
    sr_model = build_sr_model(args.sr_scale) if args.sr else None

    all_summaries = {}

    for i, video_path in enumerate(selected, 1):
        out_path = output_dir / (video_path.stem + "_tracked.mp4")
        print(f"[{i}/{len(selected)}] {video_path.name} → {out_path.name}")

        vid_stem     = video_path.stem
        gt_entry     = gt_db.get(vid_stem, {})
        fight_frames = gt_entry.get("fight_frames", None)
        s1 = resolve_stage1_list(vid_stem)
        if s1 is not None:
            fight_frames = None  # Stage-1 + full-tube tracking; do not use GT fast path
        elif stage1_index_map or stage1_dir is not None:
            if stage1_dir is not None:
                hint = (
                    f"{stage1_dir / (vid_stem + '.json')}, "
                    f"{stage1_dir / (vid_stem + '.csv')}, {stage1_dir / (vid_stem + '_*.csv')}, "
                    f"a single *_stage2_side_by_side_labels_thr_*.csv in the folder, or --stage1_index"
                )
            else:
                hint = "a key in --stage1_index"
            print(
                f"  [WARN] No Stage-1 for {vid_stem} (expected {hint}); skipping video.",
                file=sys.stderr,
            )
            continue

        if s1 is not None:
            print(f"  Stage-1: {sum(1 for x in s1 if int(x) == 1)} non-zero frame(s) in list (len={len(s1)}) for {vid_stem}")
        elif fight_frames is not None:
            print(f"  GT: {len(fight_frames)} fight frame(s) found for {vid_stem}")
        else:
            print(f"  GT: no entry for {vid_stem} — processing all frames")

        xml_path = None
        if s1 is not None and not args.no_cvat_xml:
            xml_path = output_dir / f"{vid_stem}_annotations.xml"

        summary = process_video(
            model            = model,
            tracker_cfg      = tracker_cfg,
            video_path       = video_path,
            output_path      = out_path,
            sr_model         = sr_model,
            conf_thr         = args.conf,
            scales           = tuple(args.scales),
            max_gap          = args.max_gap,
            show_bg_mask     = args.show_bg_mask,
            clahe_clip       = args.clahe_clip,
            fight_frames     = fight_frames,
            stage1_violence  = s1,
            xml_path         = xml_path,
            cvat_label       = args.cvat_label,
        )

        total_interp = sum(v["frames_interpolated"] for v in summary.values())
        entry = {
            "output":            str(out_path),
            "num_tracks":        len(summary),
            "gt_fight_frames":   len(fight_frames) if fight_frames else "all",
            "fps_used":          gt_entry.get("fps", "unknown"),
            "fps_source":        gt_entry.get("fps_source", "unknown"),
            "tracks":            {str(tid): info for tid, info in summary.items()},
        }
        if s1 is not None:
            entry["stage1"] = {
                "list_length":   len(s1),
                "violence_frames": sum(1 for x in s1 if int(x) == 1),
                "cvat11_xml":    str(xml_path) if xml_path else None,
            }
        all_summaries[video_path.name] = entry
        print(f"  Done. {len(summary)} track(s), {total_interp} interpolated box(es).\n")

    summary_path = output_dir / "tracking_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("── All done ──")
    print(f"Annotated videos : {output_dir}")
    print(f"Tracking summary : {summary_path}")
    print()
    print("Legend on annotated videos:")
    print("  Solid box  (ID:N conf)   — detected box from model")
    print("  Dashed box (~ID:N conf)  — linearly interpolated (missing detection)")


if __name__ == "__main__":
    main()
