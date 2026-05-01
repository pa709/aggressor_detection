"""
main.py — Fight role recognition pipeline
─────────────────────────────────────────────────────
Usage:
    python main.py                       # full run (uses pose cache if exists)
    python main.py --skip-pose-cache     # force re-run pose estimation
    python main.py --save-videos         # write prediction overlay videos
    python main.py --frame-step 5        # subsample frames (faster)
    python main.py --n-test 3            # number of test videos (default 3)
"""

import argparse
import pickle
import time
from pathlib import Path

from module1_data_loader import discover_dataset, split_dataset
from module2_augmenter import VideoAugmenter, AugmentConfig
from module3_pose_estimator import PoseEstimator
from module4_classifier import build_feature_matrix, train_classifier, print_top_features
from module5_validation import evaluate_all, print_evaluation_report, save_prediction_video


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT      = Path("data")
MODEL_PATH     = Path("models/classifier.pkl")
POSE_CACHE_DIR = Path("cache")
OUTPUT_DIR     = Path("outputs")


# ── Pose cache helpers ────────────────────────────────────────────────────────

def _cache_path(split: str) -> Path:
    return POSE_CACHE_DIR / f"poses_{split}.pkl"

def save_pose_cache(poses, split: str):
    POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(split), "wb") as f:
        pickle.dump(poses, f)
    print(f"[Cache] Saved → {_cache_path(split)}")

def load_pose_cache(split: str):
    p = _cache_path(split)
    if p.exists():
        print(f"[Cache] Loading pose results from {p}")
        with open(p, "rb") as f:
            return pickle.load(f)
    return None



# ── Leave-One-Video-Out Cross-Validation ─────────────────────────────────────

def run_loocv(samples, estimator, args):
    """
    Leave-One-Video-Out CV: train on N-1 videos, test on 1, rotate.
    Gives an unbiased estimate of generalisation performance.
    Each video is test set exactly once, regardless of how many tracks it has.
    """
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from module4_classifier import build_feature_matrix, build_classifier
    from module5_validation import predict_sample
    from sklearn.metrics import classification_report, accuracy_score
    import numpy as np

    FIG_DIR = Path("outputs/loocv_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    AGG_C, NONAGG_C = "#E74C3C", "#2980B9"

    def save_video_fig(vname, preds, agg_correct):
        if not preds:
            return None
        tids   = list(preds.keys())
        probs  = [preds[t]["prob_aggressor"] for t in tids]
        roles  = [preds[t]["label_true"]     for t in tids]
        predls = [preds[t]["label_pred"]     for t in tids]
        colors = []
        for role, pred in zip(roles, predls):
            if role == "aggressor" and pred == "aggressor":
                colors.append(AGG_C)
            elif role == "non_aggressor" and pred == "non_aggressor":
                colors.append(NONAGG_C)
            elif role == "aggressor" and pred == "non_aggressor":
                colors.append("#E67E22")   # FN: orange
            else:
                colors.append("#95A5A6")   # FP: grey

        fig, ax = plt.subplots(figsize=(max(4, len(tids) * 0.7 + 1.5), 3.5))
        ax.bar(range(len(tids)), probs, color=colors, edgecolor="white", width=0.6)
        ax.axhline(0.35, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(tids)))
        ax.set_xticklabels([f"T{t}" for t in tids], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(aggressor)")
        status = "Aggressor identified ✓" if agg_correct else "Aggressor missed ✗"
        ax.set_title(f"{vname}  —  {status}", fontsize=9)
        legend_handles = [
            mpatches.Patch(color=AGG_C,     label="True aggressor — correct"),
            mpatches.Patch(color="#E67E22", label="True aggressor — missed"),
            mpatches.Patch(color=NONAGG_C,  label="Non-aggressor — correct"),
            mpatches.Patch(color="#95A5A6", label="Non-aggressor — false positive"),
        ]
        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
        plt.tight_layout()
        suffix = "correct" if agg_correct else "missed"
        fname  = FIG_DIR / f"{vname}_{suffix}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        return fname

    print("\n" + "="*60)
    print("LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION")
    print(f"  {len(samples)} videos, each used as test once")
    print("="*60)

    # Load or build pose cache for all videos
    all_poses = {}
    cache_path = Path("cache/poses_all_loocv.pkl")
    if cache_path.exists() and not args.skip_pose_cache:
        print(f"[LOOCV] Loading pose cache from {cache_path}")
        with open(cache_path, "rb") as f:
            all_poses = pickle.load(f)

    missing = [s for s in samples if s.name not in all_poses]
    if missing:
        from module2_augmenter import VideoAugmenter, AugmentConfig
        aug = VideoAugmenter(
            config=AugmentConfig(p_flip=0.5, p_brightness=0.7, p_blur=0.3),
            seed=args.seed,
        )
        print(f"[LOOCV] Running pose estimation for {len(missing)} videos ...")
        for i, sample in enumerate(missing):
            print(f"  ({i+1}/{len(missing)}) {sample.name}")
            result = estimator.process_sample(sample, augmenter=aug, frame_step=args.frame_step)
            all_poses[sample.name] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_poses, f)
        print(f"[LOOCV] Cache saved → {cache_path}")

    y_true_all, y_pred_all = [], []
    video_results = []
    n_agg_correct = 0
    n_agg_total   = 0

    for i, test_sample in enumerate(samples):
        train_samples = [s for s in samples if s.name != test_sample.name]

        # Build train matrix
        train_poses = {s.name: all_poses[s.name] for s in train_samples
                       if s.name in all_poses}
        X_tr, y_tr, _ = build_feature_matrix(train_poses)

        if y_tr.sum() == 0:
            continue  # no aggressors in training fold, skip

        # Train
        pipeline = build_classifier()
        pipeline.fit(X_tr, y_tr)

        # Predict on test video
        test_poses_single = {test_sample.name: all_poses[test_sample.name]}
        preds = predict_sample(pipeline, all_poses[test_sample.name])

        agg_correct = sum(
            1 for info in preds.values()
            if info["label_true"] == "aggressor" and info["label_pred"] == "aggressor"
        )
        agg_total = sum(1 for info in preds.values() if info["label_true"] == "aggressor")
        n_agg_correct += agg_correct
        n_agg_total   += agg_total

        for info in preds.values():
            y_true_all.append(1 if info["label_true"] == "aggressor" else 0)
            y_pred_all.append(1 if info["label_pred"] == "aggressor" else 0)

        agg_probs = [(t, info["prob_aggressor"]) for t, info in preds.items()
                     if info["label_true"] == "aggressor"]
        status = "✓" if agg_correct > 0 else "✗"
        prob_str = ", ".join(f"T{t}:P={p:.2f}" for t, p in agg_probs)
        print(f"  {status} {test_sample.name:<28s} agg: {prob_str}")

        # Save per-video probability bar chart
        save_video_fig(test_sample.name, preds, agg_correct > 0)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    print(f"\n{'='*60}")
    print(f"LOOCV RESULTS  ({len(samples)} videos)")
    print(f"{'='*60}")
    print(f"Overall accuracy:      {accuracy_score(y_true_all, y_pred_all):.3f}")
    print(f"Aggressor recall:      {n_agg_correct}/{n_agg_total} = {n_agg_correct/max(n_agg_total,1):.3f}")
    print()
    print(classification_report(y_true_all, y_pred_all,
          target_names=["non_aggressor", "aggressor"], zero_division=0))
    print("="*60)
    print()
    print("NOTE: LOOCV gives an UNBIASED estimate of generalisation performance.")
    print("The fixed train/test split result is misleading because it measures")
    print("performance on only 3 videos selected by random seed.")

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(args):
    t0 = time.time()

    # ── Step 1: Discover & split ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 – Data discovery & split")
    print("="*60)
    samples = discover_dataset(str(DATA_ROOT))
    train_samples, test_samples = split_dataset(
        samples, n_test=args.n_test, seed=args.seed,
    )

    # ── Step 2+3: Pose estimation ─────────────────────────────────────
    estimator = PoseEstimator(
        model_name=args.pose_model,
        conf_threshold=args.pose_conf,
    )

    print("\n" + "="*60)
    print("STEP 2+3 – Pose estimation (train set)")
    print("="*60)
    train_poses = None if args.skip_pose_cache else load_pose_cache("train")
    if train_poses is None:
        aug = VideoAugmenter(
            config=AugmentConfig(p_flip=0.5, p_brightness=0.7, p_blur=0.3, p_hsv=0.5),
            seed=args.seed,
        )
        train_poses = estimator.process_all(
            train_samples, augmenter=aug, frame_step=args.frame_step,
        )
        save_pose_cache(train_poses, "train")

    print("\n" + "="*60)
    print("STEP 2+3 – Pose estimation (test set, no augmentation)")
    print("="*60)
    test_poses = None if args.skip_pose_cache else load_pose_cache("test")
    if test_poses is None:
        test_poses = estimator.process_all(
            test_samples, augmenter=None, frame_step=args.frame_step,
        )
        save_pose_cache(test_poses, "test")

    # ── LOOCV (optional) ──────────────────────────────────────────────
    if args.loocv:
        run_loocv(samples, estimator, args)

    # ── Step 4: Feature extraction & classifier training ──────────────
    print("\n" + "="*60)
    print("STEP 4 – Feature extraction & classifier training")
    print("="*60)
    X_train, y_train, _ = build_feature_matrix(train_poses)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipeline = train_classifier(X_train, y_train, save_path=MODEL_PATH)
    print_top_features(pipeline, top_n=10)

    # ── Step 5: Validation ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 – Validation on test set")
    print("="*60)
    y_true, y_pred, per_video = evaluate_all(pipeline, test_poses)
    print_evaluation_report(y_true, y_pred, per_video)

    if args.save_videos:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for sample in test_samples:
            save_prediction_video(
                sample=sample,
                pose_results=test_poses[sample.name],
                track_predictions=per_video.get(sample.name, {}),
                output_path=OUTPUT_DIR / f"{sample.name}_pred.mp4",
            )

    print(f"\n[Main] Pipeline complete in {(time.time()-t0)/60:.1f} minutes.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fight role recognition pipeline")
    p.add_argument("--n-test",          type=int,   default=3)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--frame-step",      type=int,   default=3,
                   help="Process every N-th frame (1=all frames)")
    p.add_argument("--pose-model",      type=str,   default="yolov8m-pose.pt")
    p.add_argument("--pose-conf",       type=float, default=0.3)
    p.add_argument("--skip-pose-cache", action="store_true",
                   help="Force re-run pose estimation even if cache exists")
    p.add_argument("--save-videos",     action="store_true",
                   help="Write prediction overlay MP4s to outputs/")
    p.add_argument("--loocv",           action="store_true",
                   help="Run Leave-One-Video-Out cross-validation on all samples")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())