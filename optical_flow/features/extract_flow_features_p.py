

import argparse
import glob
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


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
# XML parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_annotations(xml_path: str) -> dict:
    """
    Handles two annotation formats automatically.

    Format A (track-level label):
      <track label="aggressor"> ... </track>
      frame entry: (xtl, ytl, xbr, ybr, outside, track_label)

    Format B (per-box role attribute):
      <track label="person">
        <box ...><attribute name="role">aggressor</attribute></box>
      </track>
      frame entry: (xtl, ytl, xbr, ybr, outside, per_box_role)

    Returns:
        {track_id (int): {
            'label': str,   # track-level label (normalised)
            'format': 'A' | 'B',
            'frames': {frame_num: (xtl, ytl, xbr, ybr, outside, effective_role)}
        }}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = {}
    for track_el in root.findall("track"):
        track_id    = int(track_el.get("id"))
        track_label = normalise_label(track_el.get("label", ""))
        fmt         = "B" if track_label == "person" else "A"
        frames      = {}
        for box in track_el.findall("box"):
            fn      = int(box.get("frame"))
            xtl     = float(box.get("xtl"))
            ytl     = float(box.get("ytl"))
            xbr     = float(box.get("xbr"))
            ybr     = float(box.get("ybr"))
            outside = int(box.get("outside", 0))
            if fmt == "A":
                role = track_label
            else:
                role = None
                for attr in box.findall("attribute"):
                    if attr.get("name") == "role":
                        role = normalise_label(attr.text or "")
            frames[fn] = (xtl, ytl, xbr, ybr, outside, role)
        tracks[track_id] = {"label": track_label, "format": fmt, "frames": frames}
    return tracks


def first_aggressor_frame(tracks: dict) -> int:
    """
    First non-outside frame where role == 'aggressor'.
    Works for both Format A (track label) and Format B (per-box role).
    """
    frames = [
        fn
        for t in tracks.values()
        for fn, box in t["frames"].items()
        if box[4] == 0 and box[5] == "aggressor"
    ]
    if not frames:
        raise ValueError("No aggressor frames found in annotation.")
    return min(frames)


# ──────────────────────────────────────────────────────────────────────────────
# Sequence planning — decide which tracks form each sample
# ──────────────────────────────────────────────────────────────────────────────

def plan_sequences(tracks: dict, ws: int, we: int) -> list:
    """
    Returns a list of sample plans:
      {
        'label'   : str,
        'track_id': str,
        'frame_map': {fn: (xtl, ytl, xbr, ybr)}
      }

    Format A aggressor logic:
      - Tracks with role='aggressor' for 30/30 window frames -> 1 sample each
      - Non-overlapping partial tracks merged if combined = 30/30 -> 1 sample

    Format B aggressor logic:
      - Pick the track with the MOST aggressor-labeled frames in the window
        (this is the aggressor person regardless of per-frame role transitions)
      - Use all frames from that track that are in the window (forward-fill gaps)
      - Label the whole sequence 'aggressor'

    Non-aggressor logic (both formats):
      - Only tracks where all 30 window frames have role='non-aggressor'
    """
    window = set(range(ws, we + 1))
    samples = []

    # Detect format from first track
    fmt = next(iter(tracks.values()))["format"] if tracks else "A"

    if fmt == "A":
        # ── Format A Aggressor ──
        agg_partial = {}
        for tid, td in tracks.items():
            covered = {fn for fn, box in td["frames"].items()
                       if fn in window and box[4] == 0 and box[5] == "aggressor"}
            if len(covered) == 0:
                continue
            if len(covered) == 30:
                frame_map = {fn: td["frames"][fn][:4] for fn in covered}
                samples.append({"label": "aggressor", "track_id": str(tid),
                                 "frame_map": frame_map})
            else:
                agg_partial[tid] = covered

        # Merge non-overlapping partial aggressor tracks
        if agg_partial:
            used = set()
            tids = sorted(agg_partial.keys())
            for tid in tids:
                if tid in used:
                    continue
                merged_frames = set(agg_partial[tid])
                merged_tids   = [tid]
                used.add(tid)
                for tid2 in tids:
                    if tid2 in used:
                        continue
                    if not merged_frames & agg_partial[tid2]:
                        merged_frames |= agg_partial[tid2]
                        merged_tids.append(tid2)
                        used.add(tid2)
                if len(merged_frames) == 30:
                    frame_map = {}
                    for fn in window:
                        for t in merged_tids:
                            td = tracks[t]
                            if fn in td["frames"] and td["frames"][fn][4] == 0:
                                frame_map[fn] = td["frames"][fn][:4]
                                break
                    samples.append({
                        "label"    : "aggressor",
                        "track_id" : "+".join(str(t) for t in sorted(merged_tids)),
                        "frame_map": frame_map,
                    })

    else:
        # ── Format B Aggressor ──
        # Pick track with most aggressor-labeled frames in window
        best_tid, best_count, best_frames = None, 0, {}
        for tid, td in tracks.items():
            win_boxes = {fn: box for fn, box in td["frames"].items()
                         if fn in window and box[4] == 0}
            n_agg = sum(1 for box in win_boxes.values() if box[5] == "aggressor")
            if n_agg > best_count:
                best_count  = n_agg
                best_tid    = tid
                best_frames = win_boxes

        if best_tid is not None and best_count > 0:
            # Build frame_map for all 30 window frames; forward-fill missing ones
            frame_map  = {}
            last_bbox  = None
            for fn in sorted(window):
                if fn in best_frames:
                    last_bbox = best_frames[fn][:4]
                    frame_map[fn] = last_bbox
                elif last_bbox is not None:
                    frame_map[fn] = last_bbox   # forward-fill
            if len(frame_map) == 30:
                samples.append({"label": "aggressor", "track_id": str(best_tid),
                                 "frame_map": frame_map})

    # ── Non-aggressor: all 30 window frames must have role='non-aggressor' ──
    for tid, td in tracks.items():
        win_roles = {fn: box[5] for fn, box in td["frames"].items()
                     if fn in window and box[4] == 0}
        if len(win_roles) == 30 and all(r == "non-aggressor" for r in win_roles.values()):
            frame_map = {fn: td["frames"][fn][:4] for fn in win_roles}
            samples.append({"label": "non-aggressor", "track_id": str(tid),
                             "frame_map": frame_map})

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Optical flow feature helpers
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
    mag      = np.sqrt(fx**2 + fy**2)
    mean_mag = float(mag.mean())
    peak_mag = float(mag.max())
    angles   = np.degrees(np.arctan2(fy, fx)) % 360.0
    hist, _  = np.histogram(angles.ravel(), bins=8, range=(0.0, 360.0))
    total    = hist.sum()
    dir_hist = (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)
    return mean_mag, peak_mag, dir_hist


NULL_FEAT = (0.0, 0.0, np.zeros(8, dtype=np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Video frame loading
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
# Per-video processing
# ──────────────────────────────────────────────────────────────────────────────

# Per-file skip overrides: video_stem -> frames to skip after first aggressor frame
# Default skip is 10 (first_aggressor_frame + 10 -> + 39)
SKIP_OVERRIDES = {
    "F_95_0_0_0_0"  : 180,   # custom skip: first_agg + 180 -> + 209
    "F_39_1_0_0_0"  : 60,    # custom skip: first_agg + 60  -> + 89
}


def process_video(video_path: str, xml_path: str) -> list:
    """
    Returns list of samples, each:
      {video_name, track_id, label, sequence: np.ndarray (30, 11)}
    """
    tracks     = parse_annotations(xml_path)
    fa         = first_aggressor_frame(tracks)
    video_name = Path(video_path).stem
    skip       = SKIP_OVERRIDES.get(video_name, 10)
    ws         = fa + skip
    we         = fa + skip + 29   # inclusive

    samples = plan_sequences(tracks, ws, we)
    if not samples:
        return []

    # Load one extra frame before window for flow at ws
    needed = list(range(ws - 1, we + 1))
    gray   = load_gray_frames(video_path, needed)

    results    = []

    for sample in samples:
        frame_map = sample["frame_map"]   # fn -> (xtl, ytl, xbr, ybr)
        sequence  = np.zeros((30, 11), dtype=np.float32)
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

            sequence[i, 0]    = mean_mag
            sequence[i, 1]    = peak_mag
            sequence[i, 2:10] = dir_hist
            sequence[i, 10]   = temporal_deriv

        results.append({
            "video_name": video_name,
            "track_id"  : sample["track_id"],
            "label"     : sample["label"],
            "sequence"  : sequence,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--ann_dir",    required=True)
    parser.add_argument("--output",     default="optical_flow_features_v2.npz")
    args = parser.parse_args()

    EXCLUDED = {
        "F_3_1_0_0_0",    # severe imbalance
        "F_44_1_2_0_0",   # severe imbalance
        "F_202_0_0_0_0",  # no aggressor tracks
        "F_45_0_0_0_0",   # severe imbalance
        "F_47_1_2_0_0",   # severe imbalance
        "F_48_1_2_0_0",   # severe imbalance
        "F_53_1_2_0_0",   # severe imbalance
        "F_66_1_2_0_0",   # aggressor ends before window, no swap found
        "F_212_1_1_0_0",  # aggressor gap, no swap found
        "F_213_0_1_0_0",  # aggressor swap ambiguous
    }

    xml_files = sorted(glob.glob(os.path.join(args.ann_dir, "*_annotations.xml")))
    if not xml_files:
        print(f"[ERROR] No *_annotations.xml files found in: {args.ann_dir}")
        return

    all_results = []
    for xml_path in xml_files:
        xml_name   = Path(xml_path).name
        video_stem = xml_name.replace("_annotations.xml", "")
        video_path = os.path.join(args.videos_dir, video_stem + ".mp4")

        if video_stem in EXCLUDED:
            print(f"[EXCLUDED] {video_stem}")
            continue
        if not os.path.exists(video_path):
            print(f"[SKIP] Video not found: {video_path}")
            continue

        print(f"Processing {video_stem} ...", end=" ", flush=True)
        try:
            results = process_video(video_path, xml_path)
            all_results.extend(results)
            agg = sum(1 for r in results if r["label"] == "aggressor")
            non = sum(1 for r in results if r["label"] == "non-aggressor")
            print(f"{len(results)} samples  (agg={agg}, non-agg={non})")
        except Exception as exc:
            print(f"ERROR - {exc}")
            traceback.print_exc()

    if not all_results:
        print("[ERROR] No data extracted.")
        return

    n = len(all_results)
    print(f"\nBuilding dataset ({n} samples) ...")

    features    = np.stack([r["sequence"]   for r in all_results]).astype(np.float32)
    labels      = np.array([r["label"]      for r in all_results])
    video_names = np.array([r["video_name"] for r in all_results])
    track_ids   = np.array([r["track_id"]   for r in all_results])

    np.savez_compressed(
        args.output,
        features    = features,
        labels      = labels,
        video_names = video_names,
        track_ids   = track_ids,
    )

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n{'─' * 50}")
    print(f"Saved  ->  {args.output}")
    print(f"Total samples : {n}")
    print(f"Feature shape : {features.shape}  (N x 30 x 11)")
    print("Label distribution:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  {lbl:<20s}: {cnt}")
    print(f"Videos processed: {len(np.unique(video_names))}")
    print(f"{'─' * 50}")
    print("\nColumn legend (axis 2):")
    print("  [:, :, 0]     - mean_magnitude")
    print("  [:, :, 1]     - peak_magnitude")
    print("  [:, :, 2:10]  - direction histogram (8 bins, normalised)")
    print("  [:, :, 10]    - temporal_derivative of mean_magnitude")


if __name__ == "__main__":
    main()
