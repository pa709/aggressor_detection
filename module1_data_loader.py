"""
Module 1: Data Loader
- Reads data.csv for fight frame ranges
- Discovers video + XML pairs from data/video_data/
- Splits into train/test sets
- Provides VideoAugmenter (used by Module 2) applied at read time
"""

import os
import random
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Optional

import cv2
import numpy as np
import xml.etree.ElementTree as ET


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class TrackAnnotation:
    """One labeled person track inside a video."""
    track_id: int
    label: str                          # "aggressor" | "non_aggressor"
    boxes: dict = field(default_factory=dict)   # frame_idx -> (xtl, ytl, xbr, ybr, outside)


@dataclass
class VideoSample:
    """Everything we know about one video clip."""
    name: str                           # e.g. "F_9_1_2_0_0"
    video_path: Path
    xml_path: Path
    frame_start: int                    # fight segment start (from CSV)
    frame_end: int                      # fight segment end   (from CSV)
    tracks: List[TrackAnnotation] = field(default_factory=list)


# ──────────────────────────────────────────────
# XML parser
# ──────────────────────────────────────────────

AGGRESSOR_VARIANTS = {"aggressor", "agressor", "agresor", "aggresor"}

def is_aggressor_label(label: str) -> bool:
    """Match 'aggressor' tolerating common typos."""
    return label.strip().lower() in AGGRESSOR_VARIANTS


def parse_cvat_xml(
    xml_path: Path,
    filter_start: Optional[int] = None,
    filter_end: Optional[int] = None,
    min_boxes: int = 3,
) -> List[TrackAnnotation]:
    """
    Parse CVAT 1.1 XML and return relevant tracks.

    Two coordinate systems exist in this dataset:

    Group A — Trimmed clips sent to CVAT:
        XML frame indices start at 0, covering [0, clip_length).
        filter_start=None, filter_end=None → accept all boxes.

    Group B — Full original videos sent to CVAT:
        XML frame indices are original-video coordinates.
        filter_start=csv_frame_start, filter_end=csv_frame_end
        → only boxes inside the fight window are kept.

    min_boxes: tracks with fewer valid boxes than this are discarded
               (removes ghost detections from BoT-SORT noise).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = []

    for track_el in root.findall("track"):
        track_id = int(track_el.get("id"))
        label = track_el.get("label")

        boxes = {}
        for box_el in track_el.findall("box"):
            if int(box_el.get("outside", 0)) == 1:
                continue
            frame = int(box_el.get("frame"))

            # Apply range filter for full-video annotations
            if filter_start is not None and frame < filter_start:
                continue
            if filter_end is not None and frame > filter_end:
                continue

            boxes[frame] = (
                float(box_el.get("xtl")),
                float(box_el.get("ytl")),
                float(box_el.get("xbr")),
                float(box_el.get("ybr")),
            )

        if len(boxes) < min_boxes:
            continue

        tracks.append(TrackAnnotation(
            track_id=track_id,
            label="aggressor" if is_aggressor_label(label) else "non_aggressor",
            boxes=boxes,
        ))

    return tracks


# ──────────────────────────────────────────────
# Dataset discovery
# ──────────────────────────────────────────────

def discover_dataset(data_root: str = "data") -> List[VideoSample]:
    """
    Walk data/video_data/<name>/ folders.
    Expects:
        data/video_data/<name>/<name>.mp4  (or .avi / .mov)
        data/video_data/<name>/annotations.xml
    Cross-references data/data.csv for fight frame ranges.
    """
    data_root = Path(data_root)
    csv_path = data_root / "data.csv"
    video_root = data_root / "video_data"

    if not csv_path.exists():
        raise FileNotFoundError(f"data.csv not found at {csv_path}")
    if not video_root.exists():
        raise FileNotFoundError(f"video_data folder not found at {video_root}")

    df = pd.read_csv(csv_path)
    csv_lookup = {
        row["name"]: {
            "csv_frame_start": int(row["frame_start"]),
            "csv_frame_end":   int(row["frame_end"]),
        }
        for _, row in df.iterrows()
    }

    # Load video_info.csv if present (gives actual trimmed-clip frame counts)
    video_info_path = data_root / "video_info.csv"
    video_info = {}
    if video_info_path.exists():
        vi_df = pd.read_csv(video_info_path)
        video_info = {row["name"]: int(row["frames"]) for _, row in vi_df.iterrows()}
        print(f"[DataLoader] Loaded video_info.csv ({len(video_info)} entries)")
    else:
        print("[DataLoader] video_info.csv not found — will use cv2 to get frame counts")

    samples = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for folder in sorted(video_root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name

        # Find video file
        video_file = None
        for ext in video_extensions:
            candidate = folder / f"{name}{ext}"
            if candidate.exists():
                video_file = candidate
                break
        if video_file is None:
            print(f"[WARN] No video found in {folder}, skipping.")
            continue

        # Find XML
        xml_file = folder / "annotations.xml"
        if not xml_file.exists():
            print(f"[WARN] No annotations.xml in {folder}, skipping.")
            continue

        if name not in csv_lookup:
            print(f"[WARN] {name} not in data.csv, skipping.")
            continue

        # ── Get actual frame count ────────────────────────────────────────────
        csv_start = csv_lookup[name]["csv_frame_start"]
        csv_end   = csv_lookup[name]["csv_frame_end"]

        if name in video_info:
            actual_frames = video_info[name]
        else:
            cap_tmp = cv2.VideoCapture(str(video_file))
            actual_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_tmp.release()

        # ── Determine coordinate system (3-way) ───────────────────────────────
        #
        # CLEAN      actual ≈ csv_end - csv_start (±10 frames)
        #            Video is a pre-trimmed fight clip; XML is 0-based.
        #
        # FULL_VIDEO actual >> expected AND xml covers ≥50% of video
        #            Full original video sent to CVAT; XML uses original coords.
        #            Filter XML boxes to [csv_start, csv_end], remap to local.
        #
        # TIGHT_XML  actual >> expected AND xml covers <50% of video
        #            Annotator only labelled the fight segment in a longer video.
        #            Use xml extent as fight window, remap to local coords.

        # Quick scan to get xml global extent (no filtering yet)
        xml_all_frames = []
        try:
            import xml.etree.ElementTree as _ET
            _root = _ET.parse(xml_file).getroot()
            xml_all_frames = [
                int(b.get("frame"))
                for t in _root.findall("track")
                for b in t.findall("box")
                if int(b.get("outside", 0)) == 0
            ]
        except Exception:
            pass

        xml_global_min = min(xml_all_frames) if xml_all_frames else 0
        xml_global_max = max(xml_all_frames) if xml_all_frames else actual_frames - 1
        xml_coverage   = (xml_global_max - xml_global_min) / actual_frames if actual_frames > 0 else 1.0

        expected_clip = csv_end - csv_start
        delta         = abs(actual_frames - expected_clip)

        if delta <= 10:
            filter_start, filter_end = None, None
            remap_offset = 0
            frame_start_local, frame_end_local = 0, actual_frames - 1
            coord_note = "CLEAN"
        elif xml_coverage >= 0.5:
            filter_start, filter_end = csv_start, csv_end
            remap_offset = csv_start
            frame_start_local, frame_end_local = 0, actual_frames - 1
            coord_note = f"FULL_VIDEO csv=[{csv_start},{csv_end}]"
        else:
            filter_start, filter_end = xml_global_min, xml_global_max
            remap_offset = xml_global_min
            frame_start_local = xml_global_min
            frame_end_local   = xml_global_max
            coord_note = f"TIGHT_XML xml=[{xml_global_min},{xml_global_max}]"

        # ── Parse annotations ─────────────────────────────────────────────────
        tracks = parse_cvat_xml(
            xml_file,
            filter_start=filter_start,
            filter_end=filter_end,
            min_boxes=3,
        )

        if not tracks:
            print(f"[WARN] {name}: no tracks after filtering ({coord_note}), skipping.")
            continue

        aggressor_count = sum(1 for t in tracks if t.label == "aggressor")
        non_agg_count   = len(tracks) - aggressor_count

        if aggressor_count == 0:
            print(f"[WARN] {name}: 0 aggressor tracks found — skipping.")
            continue

        print(f"  {name}: {len(tracks)} tracks "
              f"({aggressor_count} aggressor, {non_agg_count} non_aggressor) "
              f"| {coord_note}")

        # Remap box frame indices to local video coords
        if remap_offset != 0:
            for track in tracks:
                track.boxes = {
                    (f - remap_offset): bbox
                    for f, bbox in track.boxes.items()
                    if 0 <= (f - remap_offset) < actual_frames
                }

        samples.append(VideoSample(
            name=name,
            video_path=video_file,
            xml_path=xml_file,
            frame_start=frame_start_local,
            frame_end=frame_end_local,
            tracks=tracks,
        ))

    print(f"[DataLoader] Discovered {len(samples)} valid video samples.")
    return samples


# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────

def split_dataset(
    samples: List[VideoSample],
    n_test: int = 3,
    seed: int = 42,
) -> Tuple[List[VideoSample], List[VideoSample]]:
    """
    Stratified split: keep roughly equal aggressor coverage in both sets.
    Returns (train_samples, test_samples).
    """
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    test_samples = shuffled[:n_test]
    train_samples = shuffled[n_test:]

    print(f"[DataLoader] Split → train: {len(train_samples)}, test: {len(test_samples)}")
    print(f"  Test videos: {[s.name for s in test_samples]}")
    return train_samples, test_samples


# ──────────────────────────────────────────────
# Frame reader  (used by downstream modules)
# ──────────────────────────────────────────────

def read_frames_in_range(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    augmenter=None,         # module2 VideoAugmenter instance or None
    step: int = 1,          # read every N-th frame (set >1 to speed up)
) -> List[Tuple[int, np.ndarray]]:
    """
    Read frames [frame_start, frame_end] from video.
    Applies augmenter if provided (training only).
    Returns list of (frame_idx, bgr_frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frames = []
    idx = frame_start

    while idx <= frame_end:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - frame_start) % step == 0:
            if augmenter is not None:
                frame = augmenter.apply(frame)
            frames.append((idx, frame))
        idx += 1

    cap.release()
    return frames


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    samples = discover_dataset("data")
    train, test = split_dataset(samples, n_test=3)

    # Print summary
    for s in train[:3]:
        aggressor_tracks = [t for t in s.tracks if t.label == "aggressor"]
        print(f"  {s.name} | frames [{s.frame_start}-{s.frame_end}] "
              f"| tracks: {len(s.tracks)} | aggressors: {len(aggressor_tracks)}")