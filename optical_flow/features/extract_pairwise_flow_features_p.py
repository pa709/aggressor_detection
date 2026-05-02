

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
# ──────────────────────────────────────────────────────────────────────────────

SKIP_OVERRIDES = {
    # Format A  (batch 1)
    "F_95_0_0_0_0"                     : 180,
    "F_39_1_0_0_0"                     :  60,
    # Format B  (_clean batch)
    "F_127_1_2_0_0_0_53_to_1_03_clean" :   0,
    "F_135_1_2_0_0_0_00_to_0_05_clean" : -10,
}


# ──────────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_format(xml_path: str) -> str:
    """
    Returns 'A' if any track has a label other than 'person' (track-level role),
    otherwise returns 'B' (per-box role attribute).
    """
    root = ET.parse(xml_path).getroot()
    for track_el in root.findall("track"):
        label = track_el.get("label", "").strip().lower()
        if label != "person":
            return "A"
    return "B"


# ──────────────────────────────────────────────────────────────────────────────
# Label normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise_role(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("aggressor", "agressor"):
        return "aggressor"
    if raw in ("non-aggressor", "non_aggressor"):
        return "non-aggressor"
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# FORMAT A — parsing, anchor, pair planning
# ──────────────────────────────────────────────────────────────────────────────

def parse_annotations_A(xml_path: str) -> dict:
    """
    Format A: role is the track-level label attribute.
    Returns:
        {track_id: {
            'label' : str,
            'frames': {frame_num: (xtl, ytl, xbr, ybr, outside)}
        }}
    """
    root = ET.parse(xml_path).getroot()
    tracks = {}
    for track_el in root.findall("track"):
        track_id = int(track_el.get("id"))
        label    = normalise_role(track_el.get("label", ""))
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


def first_aggressor_frame_A(tracks: dict) -> int:
    """First non-outside frame across all aggressor tracks (Format A)."""
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


def plan_pairs_A(tracks: dict, ws: int, we: int) -> list:
    """
    Format A: track label is the role for every frame.
    Only tracks with full 30/30 non-outside coverage in window are used.
    Returns list of pair dicts.
    """
    window = set(range(ws, we + 1))
    agg_plans = []
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
# FORMAT B — parsing, anchor, pair planning
# ──────────────────────────────────────────────────────────────────────────────

def parse_annotations_B(xml_path: str) -> dict:
    """
    Format B: role is a per-box <attribute name="role"> element.
    Track label is always 'person' and is ignored.
    Returns:
        {track_id: {
            'frames': {frame_num: (xtl, ytl, xbr, ybr, outside, role)}
        }}
    Also returns whether ANY box has uncertain_clip=true.
    Returns (tracks_dict, is_uncertain).
    """
    root  = ET.parse(xml_path).getroot()
    tracks    = {}
    uncertain = False

    for track_el in root.findall("track"):
        track_id = int(track_el.get("id"))
        frames   = {}
        for box in track_el.findall("box"):
            fn      = int(box.get("frame"))
            outside = int(box.get("outside", 0))
            xtl     = float(box.get("xtl"))
            ytl     = float(box.get("ytl"))
            xbr     = float(box.get("xbr"))
            ybr     = float(box.get("ybr"))

            role     = None
            for attr in box.findall("attribute"):
                name = attr.get("name", "")
                if name == "role":
                    role = normalise_role(attr.text or "")
                elif name == "uncertain_clip" and (attr.text or "").strip().lower() == "true":
                    uncertain = True

            frames[fn] = (xtl, ytl, xbr, ybr, outside, role)
        tracks[track_id] = {"frames": frames}

    return tracks, uncertain


def first_aggressor_frame_B(tracks: dict) -> int:
    """First non-outside frame with role='aggressor' across all tracks (Format B)."""
    frames = [
        fn
        for t in tracks.values()
        for fn, box in t["frames"].items()
        if box[4] == 0 and box[5] == "aggressor"
    ]
    if not frames:
        raise ValueError("No aggressor frames found in annotation.")
    return min(frames)


def plan_pairs_B(tracks: dict, ws: int, we: int) -> list:
    """
    Format B: role is per-box and can vary within a track.
    Aggressor = the single track with the MOST aggressor-labeled
    non-outside frames in the window (must be > 0).
    If two tracks tie for the most aggressor frames, the clip is
    ambiguous and 0 pairs are returned (treated as uncertain).
    All other 30/30-covered tracks with 0 aggressor frames are non-aggressors.
    Returns list of pair dicts.
    """
    window = set(range(ws, we + 1))

    eligible = []   # (tid, agg_count, frame_map)
    for tid, td in tracks.items():
        covered = {fn for fn, box in td["frames"].items()
                   if fn in window and box[4] == 0}
        if len(covered) != 30:
            continue
        frame_map  = {fn: td["frames"][fn][:4] for fn in covered}
        agg_count  = sum(1 for fn in covered
                         if td["frames"][fn][5] == "aggressor")
        eligible.append((tid, agg_count, frame_map))

    if not eligible:
        return []

    # Split into aggressor candidates and non-aggressors
    agg_candidates = [(tid, cnt, fm) for tid, cnt, fm in eligible if cnt > 0]
    non_plans      = [{"track_id": str(tid), "frame_map": fm}
                      for tid, cnt, fm in eligible if cnt == 0]

    if not agg_candidates:
        return []

    # Sort descending by aggressor frame count
    agg_candidates.sort(key=lambda x: -x[1])

    # Tie check: if top two share the same count, clip is ambiguous
    if len(agg_candidates) > 1 and agg_candidates[0][1] == agg_candidates[1][1]:
        print(f"    [WARN] Tie between tracks {agg_candidates[0][0]} and "
              f"{agg_candidates[1][0]} ({agg_candidates[0][1]} agg frames each) "
              f"— skipping clip as ambiguous.")
        return []

    best_tid, best_cnt, best_fm = agg_candidates[0]
    agg_plan = {"track_id": str(best_tid), "frame_map": best_fm}

    pairs = []
    for non in non_plans:
        pairs.append({
            "agg_track_id" : agg_plan["track_id"],
            "non_track_id" : non["track_id"],
            "agg_frame_map": agg_plan["frame_map"],
            "non_frame_map": non["frame_map"],
        })
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Optical flow feature helpers  (unchanged)
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
# Video frame loading  (unchanged)
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
# Sequence extraction for one track → (30, 11)  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def extract_sequence(frame_map: dict, gray: dict, ws: int, we: int) -> np.ndarray:
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
        temporal_deriv = 0.0 if prev_mean is None else (mean_mag - prev_mean)
        prev_mean      = mean_mag

        seq[i, 0]    = mean_mag
        seq[i, 1]    = peak_mag
        seq[i, 2:10] = dir_hist
        seq[i, 10]   = temporal_deriv

    return seq


# ──────────────────────────────────────────────────────────────────────────────
# Build pairwise (30, 15) feature from two (30, 11) sequences  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def build_pairwise_features(own_seq: np.ndarray,
                             other_seq: np.ndarray) -> np.ndarray:
    out = np.zeros((30, 15), dtype=np.float32)
    out[:, :11] = own_seq
    out[:, 11]  = own_seq[:, 0]  - other_seq[:, 0]
    out[:, 12]  = own_seq[:, 1]  - other_seq[:, 1]
    out[:, 13]  = own_seq[:, 10] - other_seq[:, 10]
    out[:, 14]  = (own_seq[:, 0] > other_seq[:, 0]).astype(np.float32)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-video processing  (dispatches on format)
# ──────────────────────────────────────────────────────────────────────────────

def process_video(video_path: str, xml_path: str) -> list:
    """
    Returns list of sample dicts, each with:
        video_name, agg_track_id, non_agg_track_id,
        pair_id, label (1/0), features (30, 15)
    Each pair produces TWO samples — one per perspective.
    """
    fmt        = detect_format(xml_path)
    video_name = Path(video_path).stem

    # ── Format A ──────────────────────────────────────────────
    if fmt == "A":
        tracks = parse_annotations_A(xml_path)
        fa     = first_aggressor_frame_A(tracks)
        skip   = SKIP_OVERRIDES.get(video_name, 10)
        ws     = fa + skip
        we     = ws + 29
        pairs  = plan_pairs_A(tracks, ws, we)

    # ── Format B ──────────────────────────────────────────────
    else:
        tracks, uncertain = parse_annotations_B(xml_path)
        if uncertain:
            print(f"    [SKIP] uncertain_clip=true — excluding clip.")
            return []
        fa    = first_aggressor_frame_B(tracks)
        skip  = SKIP_OVERRIDES.get(video_name, 10)
        ws    = fa + skip
        we    = ws + 29
        pairs = plan_pairs_B(tracks, ws, we)

    if not pairs:
        return []

    # Load frames: one extra before the window start for flow computation
    needed = list(range(ws - 1, we + 1))
    gray   = load_gray_frames(video_path, needed)

    results = []
    for pair_id, pair in enumerate(pairs):
        agg_seq = extract_sequence(pair["agg_frame_map"], gray, ws, we)
        non_seq = extract_sequence(pair["non_frame_map"], gray, ws, we)

        # Perspective A: aggressor's view → label = 1
        results.append({
            "video_name"      : video_name,
            "agg_track_id"    : pair["agg_track_id"],
            "non_agg_track_id": pair["non_track_id"],
            "pair_id"         : pair_id,
            "label"           : 1,
            "features"        : build_pairwise_features(agg_seq, non_seq),
        })

        # Perspective B: non-aggressor's view → label = 0
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
        description="Extract pairwise contrastive optical flow features "
                    "(handles Format A track-level labels and Format B per-box roles)."
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

        fmt = detect_format(xml_path)
        print(f"Processing {video_stem}  [Format {fmt}] ...", end=" ", flush=True)
        try:
            results = process_video(video_path, xml_path)
            if not results:
                print("no valid pairs (skipped)")
                continue
            all_results.extend(results)
            n_pairs = len(results) // 2
            print(f"{n_pairs} pairs -> {len(results)} samples")
        except Exception as exc:
            print(f"ERROR — {exc}")
            traceback.print_exc()

    if not all_results:
        print("[ERROR] No data extracted.")
        return

    n = len(all_results)
    print(f"\nBuilding dataset ({n} samples from {n // 2} pairs) ...")

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

    print(f"\n{'─' * 60}")
    print(f"Saved  ->  {args.output}")
    print(f"Total samples : {n}  ({n // 2} pairs x 2 perspectives)")
    print(f"Feature shape : {features.shape}  (N x 30 x 15)")
    print(f"Label dist    : 1(aggressor)={int(labels.sum())}  "
          f"0(non-aggressor)={int((labels == 0).sum())}")
    print(f"Videos w/pairs: {len(set(video_names[labels == 1]))}")
    print(f"{'─' * 60}")
    print("\nColumn legend (axis 2):")
    print("  [:, :,  0]     - own mean_magnitude")
    print("  [:, :,  1]     - own peak_magnitude")
    print("  [:, :, 2:10]   - own dir_hist (8 bins, normalised)")
    print("  [:, :, 10]     - own temporal_derivative")
    print("  [:, :, 11]     - delta_mean_mag       (own - other)")
    print("  [:, :, 12]     - delta_peak_mag       (own - other)")
    print("  [:, :, 13]     - delta_temporal_deriv (own - other)")
    print("  [:, :, 14]     - motion_leader (1 if own_mean_mag > other)")


if __name__ == "__main__":
    main()
