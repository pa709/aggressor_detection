

import argparse
import glob
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Per-file skip overrides  (default = 10)
# Add video stems here when a specific skip is needed.
# ──────────────────────────────────────────────────────────────────────────────

SKIP_OVERRIDES = {
    "F_95_0_0_0_0": 180,   # aggressor label starts at frame 0 but fight is much later
    "F_39_1_0_0_0": 60,    # similar early-label issue
    "F_66_1_2_0_0": 0,     # tracks end at frame 211; skip=0 uses window [180..209] (full coverage)
}


# ──────────────────────────────────────────────────────────────────────────────
# Label normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise_label(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("aggressor", "agressor"):
        return "aggressor"
    if raw in ("non-aggressor", "non_aggressor"):
        return "non-aggressor"
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# XML parsing  (unchanged from reference)
# ──────────────────────────────────────────────────────────────────────────────

def parse_annotations(xml_path: str) -> dict:
    """
    Returns:
        {track_id (int): {
            'label': str,
            'frames': {frame_num: (xtl, ytl, xbr, ybr, outside)}
        }}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = {}
    for track_el in root.findall("track"):
        track_id = int(track_el.get("id"))
        label    = normalise_label(track_el.get("label", ""))
        frames   = {}
        for box in track_el.findall("box"):
            fn      = int(box.get("frame"))
            xtl     = float(box.get("xtl"))
            ytl     = float(box.get("ytl"))
            xbr     = float(box.get("xbr"))
            ybr     = float(box.get("ybr"))
            outside = int(box.get("outside", 0))
            frames[fn] = (xtl, ytl, xbr, ybr, outside)
        tracks[track_id] = {"label": label, "frames": frames}
    return tracks


def first_aggressor_frame(tracks: dict) -> int:
    """First non-outside frame across all aggressor tracks."""
    frames = [
        fn
        for t in tracks.values()
        if t["label"] == "aggressor"
        for fn, box in t["frames"].items()
        if box[4] == 0
    ]
    if not frames:
        raise ValueError("No aggressor frames found in annotation.")
    return min(frames)


# ──────────────────────────────────────────────────────────────────────────────
# Pair planning — find all valid (agg_track, non_agg_track) pairs
# ──────────────────────────────────────────────────────────────────────────────

def plan_pairs(tracks: dict, ws: int, we: int) -> list:
    """
    Returns a list of pair plans:
      {
        'agg_track_id' : str,
        'non_track_id' : str,
        'agg_frame_map': {fn: (xtl, ytl, xbr, ybr)},
        'non_frame_map': {fn: (xtl, ytl, xbr, ybr)},
      }

    Only tracks with full 30/30 non-outside coverage in the window are used.
    For aggressor: each qualifying track is a separate sample.
    For non-aggressor: each qualifying track is a separate sample.
    All combinations within a video form pairs.
    """
    window = set(range(ws, we + 1))

    agg_plans = []   # list of {track_id, frame_map}
    non_plans = []

    for tid, td in tracks.items():
        covered = {fn for fn, box in td["frames"].items()
                   if fn in window and box[4] == 0}
        if len(covered) != 30:
            continue
        frame_map = {fn: td["frames"][fn][:4] for fn in covered}

        if td["label"] == "aggressor":
            agg_plans.append({"track_id": str(tid), "frame_map": frame_map})
        elif td["label"] == "non-aggressor":
            non_plans.append({"track_id": str(tid), "frame_map": frame_map})

    # Cross-product of all aggressor × non-aggressor tracks
    pairs = []
    for agg in agg_plans:
        for non in non_plans:
            pairs.append({
                "agg_track_id" : agg["track_id"],
                "non_track_id" : non["track_id"],
                "agg_frame_map": agg["frame_map"],
                "non_frame_map": non["frame_map"],
            })

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Optical flow feature helpers  (unchanged from reference)
# ──────────────────────────────────────────────────────────────────────────────

def flow_features(flow: np.ndarray, bbox: tuple) -> tuple:
    h, w = flow.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, np.zeros(8, dtype=np.float32)
    crop = flow[y1:y2, x1:x2]
    fx, fy   = crop[..., 0], crop[..., 1]
    mag      = np.sqrt(fx ** 2 + fy ** 2)
    mean_mag = float(mag.mean())
    peak_mag = float(mag.max())
    angles   = np.degrees(np.arctan2(fy, fx)) % 360.0
    hist, _  = np.histogram(angles.ravel(), bins=8, range=(0.0, 360.0))
    total    = hist.sum()
    dir_hist = (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)
    return mean_mag, peak_mag, dir_hist


NULL_FEAT = (0.0, 0.0, np.zeros(8, dtype=np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Video frame loading  (unchanged from reference)
# ──────────────────────────────────────────────────────────────────────────────

def load_gray_frames(video_path: str, frame_indices: list) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gray  = {}
    for fn in sorted(frame_indices):
        if fn < 0 or fn >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret:
            gray[fn] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return gray


# ──────────────────────────────────────────────────────────────────────────────
# Sequence extraction for one track's frame_map → (30, 11)
# ──────────────────────────────────────────────────────────────────────────────

def extract_sequence(frame_map: dict, gray: dict, ws: int, we: int) -> np.ndarray:
    """
    Compute the (30, 11) optical flow feature sequence for one track.
    frame_map: {fn: (xtl, ytl, xbr, ybr)} for all 30 window frames.
    """
    seq       = np.zeros((30, 11), dtype=np.float32)
    prev_mean = None

    for i, ann_frame in enumerate(range(ws, we + 1)):
        prev_fn = ann_frame - 1
        if prev_fn not in gray or ann_frame not in gray or ann_frame not in frame_map:
            feat = NULL_FEAT
        else:
            fflow = cv2.calcOpticalFlowFarneback(
                gray[prev_fn], gray[ann_frame],
                flow=None, pyr_scale=0.5, levels=3,
                winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            feat = flow_features(fflow, frame_map[ann_frame])

        mean_mag, peak_mag, dir_hist = feat
        temporal_deriv  = 0.0 if prev_mean is None else (mean_mag - prev_mean)
        prev_mean       = mean_mag

        seq[i, 0]    = mean_mag
        seq[i, 1]    = peak_mag
        seq[i, 2:10] = dir_hist
        seq[i, 10]   = temporal_deriv

    return seq


# ──────────────────────────────────────────────────────────────────────────────
# Build pairwise (30, 15) feature from two (30, 11) sequences
# ──────────────────────────────────────────────────────────────────────────────

def build_pairwise_features(own_seq: np.ndarray,
                             other_seq: np.ndarray) -> np.ndarray:
    """
    Combine two (30, 11) sequences into one (30, 15) pairwise feature array.

    Columns:
      [0..10]  own 11 features
      [11]     delta_mean_mag      = own[0] - other[0]
      [12]     delta_peak_mag      = own[1] - other[1]
      [13]     delta_temporal_deriv = own[10] - other[10]
      [14]     motion_leader       = 1 if own[0] > other[0] else 0
    """
    out = np.zeros((30, 15), dtype=np.float32)
    out[:, :11] = own_seq
    out[:, 11]  = own_seq[:, 0]  - other_seq[:, 0]    # delta mean_mag
    out[:, 12]  = own_seq[:, 1]  - other_seq[:, 1]    # delta peak_mag
    out[:, 13]  = own_seq[:, 10] - other_seq[:, 10]   # delta temporal_deriv
    out[:, 14]  = (own_seq[:, 0] > other_seq[:, 0]).astype(np.float32)  # motion_leader
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ──────────────────────────────────────────────────────────────────────────────

def process_video(video_path: str, xml_path: str) -> list:
    """
    Returns list of samples, each a dict:
      {
        video_name        : str,
        agg_track_id      : str,
        non_agg_track_id  : str,
        pair_id           : int,   # index within this video's pairs
        label             : int,   # 1 = aggressor, 0 = non-aggressor
        features          : np.ndarray (30, 15)
      }

    Each pair produces TWO samples — one per perspective.
    """
    tracks     = parse_annotations(xml_path)
    video_name = Path(video_path).stem
    fa         = first_aggressor_frame(tracks)
    skip       = SKIP_OVERRIDES.get(video_name, 10)
    ws         = fa + skip
    we         = fa + skip + 29   # inclusive

    pairs = plan_pairs(tracks, ws, we)
    if not pairs:
        return []

    # Load frames: one extra before window for flow computation
    needed = list(range(ws - 1, we + 1))
    gray   = load_gray_frames(video_path, needed)

    results = []
    for pair_id, pair in enumerate(pairs):

        agg_seq = extract_sequence(pair["agg_frame_map"], gray, ws, we)
        non_seq = extract_sequence(pair["non_frame_map"], gray, ws, we)

        # Perspective A: aggressor's view  → label = 1
        results.append({
            "video_name"      : video_name,
            "agg_track_id"    : pair["agg_track_id"],
            "non_agg_track_id": pair["non_track_id"],
            "pair_id"         : pair_id,
            "label"           : 1,
            "features"        : build_pairwise_features(agg_seq, non_seq),
        })

        # Perspective B: non-aggressor's view  → label = 0
        results.append({
            "video_name"      : video_name,
            "agg_track_id"    : pair["agg_track_id"],
            "non_agg_track_id": pair["non_track_id"],
            "pair_id"         : pair_id,
            "label"           : 0,
            "features"        : build_pairwise_features(non_seq, agg_seq),
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract pairwise contrastive optical flow features."
    )
    parser.add_argument("--videos_dir", required=True,
                        help="Folder containing .mp4 video files.")
    parser.add_argument("--ann_dir",    required=True,
                        help="Folder containing *_annotations.xml files.")
    parser.add_argument("--output",     default="optical_flow_pairwise.npz",
                        help="Output .npz file path.")
    args = parser.parse_args()

    xml_files = sorted(glob.glob(os.path.join(args.ann_dir, "*_annotations.xml")))
    if not xml_files:
        print(f"[ERROR] No *_annotations.xml files found in: {args.ann_dir}")
        return

    all_results = []
    for xml_path in xml_files:
        xml_name   = Path(xml_path).name
        video_stem = xml_name.replace("_annotations.xml", "")
        video_path = os.path.join(args.videos_dir, video_stem + ".mp4")

        if not os.path.exists(video_path):
            print(f"[SKIP] Video not found: {video_path}")
            continue

        print(f"Processing {video_stem} ...", end=" ", flush=True)
        try:
            results = process_video(video_path, xml_path)
            if not results:
                print("no valid pairs (skipped)")
                continue
            all_results.extend(results)
            n_pairs = len(results) // 2
            print(f"{n_pairs} pairs -> {len(results)} samples")
        except Exception as exc:
            print(f"ERROR - {exc}")
            traceback.print_exc()

    if not all_results:
        print("[ERROR] No data extracted.")
        return

    n = len(all_results)
    print(f"\nBuilding dataset ({n} samples from {n//2} pairs) ...")

    features          = np.stack([r["features"]           for r in all_results]).astype(np.float32)
    labels            = np.array([r["label"]              for r in all_results], dtype=np.int32)
    video_names       = np.array([r["video_name"]         for r in all_results])
    agg_track_ids     = np.array([r["agg_track_id"]       for r in all_results])
    non_agg_track_ids = np.array([r["non_agg_track_id"]   for r in all_results])
    pair_ids          = np.array([r["pair_id"]            for r in all_results], dtype=np.int32)

    np.savez_compressed(
        args.output,
        features          = features,
        labels            = labels,
        video_names       = video_names,
        agg_track_ids     = agg_track_ids,
        non_agg_track_ids = non_agg_track_ids,
        pair_ids          = pair_ids,
    )

    print(f"\n{'─' * 55}")
    print(f"Saved  ->  {args.output}")
    print(f"Total samples : {n}  ({n//2} pairs x 2 perspectives)")
    print(f"Feature shape : {features.shape}  (N x 30 x 15)")
    print(f"Label dist    : 1(aggressor)={int(labels.sum())}  "
          f"0(non-aggressor)={int((labels==0).sum())}")
    print(f"Videos w/pairs: {len(set(video_names[labels==1]))}")
    print(f"{'─' * 55}")
    print("\nColumn legend (axis 2):")
    print("  [:, :,  0]     - own mean_magnitude")
    print("  [:, :,  1]     - own peak_magnitude")
    print("  [:, :, 2:10]   - own dir_hist (8 bins, normalised)")
    print("  [:, :, 10]     - own temporal_derivative")
    print("  [:, :, 11]     - delta_mean_mag      (own - other)")
    print("  [:, :, 12]     - delta_peak_mag      (own - other)")
    print("  [:, :, 13]     - delta_temporal_deriv (own - other)")
    print("  [:, :, 14]     - motion_leader (1 if own_mean_mag > other)")


if __name__ == "__main__":
    main()
