"""
Module 5: Validation & Evaluation
─────────────────────────────────────────────────────
Runs the trained LR pipeline on test samples, computes:
  - Per-video aggressor prediction with two strategies:
      1. Threshold: P(aggressor) >= threshold → aggressor
      2. Per-video top-1: highest-probability person is always labelled aggressor
         (if their probability exceeds MIN_TOP1_PROB)
  - Accuracy, precision, recall, F1
  - Confusion matrix
  - Per-video track breakdown
  - Optional: prediction overlay video (bbox coloured by result)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import rankdata

from module1_data_loader import VideoSample, read_frames_in_range
from module3_pose_estimator import FramePose
from module4_classifier import (
    extract_features_one_person,
    ABS_DIM, REL_DIM, TOTAL_DIM,
    BBOX_AREA_IDX, VEL_MEAN_IDX, VEL_MAX_IDX,
    ARM_EXT_L_IDX, UBODY_AREA_IDX, APPROACH_IDX,
)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_sample(
    pipeline: Pipeline,
    pose_results: Dict[int, List[FramePose]],
    frame_width:    int   = 640,
    frame_height:   int   = 360,
    threshold:      float = 0.35,
    per_video_top1: bool  = True,
) -> Dict[int, Dict]:
    """
    Predict role for each track in one video.

    Returns { track_id -> {"label_true", "label_pred", "prob_aggressor",
                           "correct", "by_threshold", "by_top1"} }
    """
    MIN_TOP1_PROB = 0.20

    # Centroid sequences (for social & contact features)
    track_centroids = {
        tid: [np.array([(p.bbox[0]+p.bbox[2])/2, (p.bbox[1]+p.bbox[3])/2])
              for p in poses if p.valid]
        for tid, poses in pose_results.items()
    }

    # Pass 1: absolute features
    tids_ordered = []
    raw: Dict[int, dict] = {}

    for tid, poses in pose_results.items():
        if not poses:
            continue
        tids_ordered.append(tid)
        raw[tid] = {
            "true_label": poses[0].label,
            "feats":      extract_features_one_person(
                              poses, frame_width, frame_height,
                              track_centroids, tid),
        }

    if not raw:
        return {}

    # Pass 2: intra-video rank features (mirrors build_feature_matrix)
    def rank_scalars(v: np.ndarray) -> np.ndarray:
        return np.array([
            v[VEL_MEAN_IDX:VEL_MEAN_IDX+17].mean(),
            v[VEL_MAX_IDX:VEL_MAX_IDX+17].max(),
            v[APPROACH_IDX],
            v[ARM_EXT_L_IDX],
            v[BBOX_AREA_IDX],
            v[UBODY_AREA_IDX],
        ], dtype=np.float32)

    abs_vecs = np.stack([raw[t]["feats"] for t in tids_ordered])  # (N, 132)
    sc       = np.stack([rank_scalars(raw[t]["feats"]) for t in tids_ordered])
    n        = len(tids_ordered)

    if n > 1:
        rel = np.apply_along_axis(
            lambda col: (rankdata(col) - 1) / (n - 1), axis=0, arr=sc
        ).astype(np.float32)
    else:
        rel = np.full((1, REL_DIM), 0.5, dtype=np.float32)

    full_feats = np.concatenate([abs_vecs, rel], axis=1)   # (N, 140)

    # NaN imputation (wrist unreliable) — use column mean within this video
    nan_mask = np.isnan(full_feats)
    if nan_mask.any():
        col_means = np.nanmean(full_feats, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        full_feats[nan_mask] = col_means[np.where(nan_mask)[1]]

    # Pass 3: predict
    probs    = pipeline.predict_proba(full_feats)[:, 1]   # P(aggressor)
    best_idx = int(np.argmax(probs))

    predictions = {}
    for i, tid in enumerate(tids_ordered):
        prob       = float(probs[i])
        by_thresh  = prob >= threshold
        by_top1    = per_video_top1 and (i == best_idx) and (prob >= MIN_TOP1_PROB)
        pred_label = "aggressor" if (by_thresh or by_top1) else "non_aggressor"
        true_label = raw[tid]["true_label"]

        predictions[tid] = {
            "label_true":     true_label,
            "label_pred":     pred_label,
            "prob_aggressor": prob,
            "correct":        pred_label == true_label,
            "by_threshold":   by_thresh,
            "by_top1":        by_top1,
        }

    return predictions


def evaluate_all(
    pipeline: Pipeline,
    all_pose_results: Dict[str, Dict[int, List[FramePose]]],
    frame_width:    int   = 640,
    frame_height:   int   = 360,
    threshold:      float = 0.35,
    per_video_top1: bool  = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run prediction across all test videos.
    Returns (y_true, y_pred, per_video_results_dict).
    """
    y_true_all, y_pred_all = [], []
    per_video = {}

    for video_name, track_dict in all_pose_results.items():
        preds = predict_sample(
            pipeline, track_dict, frame_width, frame_height,
            threshold, per_video_top1,
        )
        per_video[video_name] = preds
        for info in preds.values():
            y_true_all.append(1 if info["label_true"] == "aggressor" else 0)
            y_pred_all.append(1 if info["label_pred"] == "aggressor" else 0)

    return np.array(y_true_all), np.array(y_pred_all), per_video


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_video: Dict,
):
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"\nOverall accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["non_aggressor", "aggressor"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':15s} non_agg   aggressor")
    print(f"  {'non_aggressor':15s}   {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  {'aggressor':15s}   {cm[1,0]:4d}      {cm[1,1]:4d}")

    print(f"\n{'Per-video breakdown:':}")
    print(f"  {'Video':<30s} {'Track':>5s} {'True':>14s} {'Pred':>14s} "
          f"{'P(agg)':>8s} {'OK':>4s}")
    print("  " + "-"*80)
    for vname, preds in per_video.items():
        for tid, info in preds.items():
            ok = "✓" if info["correct"] else "✗"
            print(f"  {vname:<30s} {tid:>5d} "
                  f"{info['label_true']:>14s} {info['label_pred']:>14s} "
                  f"{info['prob_aggressor']:>8.3f} {ok:>4s}")
    print("="*60)


# ── Optional: overlay video ───────────────────────────────────────────────────

def save_prediction_video(
    sample: VideoSample,
    pose_results: Dict[int, List[FramePose]],
    track_predictions: Dict[int, Dict],
    output_path: Path,
    frame_step: int = 1,
):
    """
    Write a result video with bbox coloured by prediction:
      Red    = correctly identified aggressor
      Green  = correctly identified non_aggressor
      Yellow = wrong prediction
    Keypoints drawn as dots.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(sample.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / frame_step, (w, h),
    )

    # Build frame → poses lookup
    frame_to_poses: Dict[int, List[FramePose]] = {}
    for poses in pose_results.values():
        for fp in poses:
            frame_to_poses.setdefault(fp.frame_idx, []).append(fp)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_start)
    idx = sample.frame_start

    while idx <= sample.frame_end:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - sample.frame_start) % frame_step == 0:
            frame = _draw_predictions(frame, frame_to_poses.get(idx, []),
                                      track_predictions)
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"[Validation] Saved prediction video → {output_path}")


def _draw_predictions(
    frame: np.ndarray,
    frame_poses: List[FramePose],
    track_predictions: Dict[int, Dict],
) -> np.ndarray:
    for fp in frame_poses:
        info = track_predictions.get(fp.track_id)
        if not info:
            continue
        correct  = info["correct"]
        pred_lbl = info["label_pred"]
        prob     = info["prob_aggressor"]

        if correct and pred_lbl == "aggressor":
            color = (0, 0, 220)      # red
        elif correct:
            color = (0, 200, 0)      # green
        else:
            color = (0, 200, 220)    # yellow

        xtl, ytl, xbr, ybr = (int(v) for v in fp.bbox)
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
        cv2.putText(frame, f"{pred_lbl} {prob:.2f}", (xtl, ytl - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for kp in fp.keypoints:
            x, y, conf = kp
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    return frame
