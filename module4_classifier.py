"""
Module 4: Feature Engineering + Classifier Training
─────────────────────────────────────────────────────
Feature vector per person (total 126 dims):

  [0:51]   Keypoint confidence — mean/std/max per joint (17×3)
  [51:55]  Bounding box — mean/std of area and aspect ratio
  [55:106] Joint velocity — mean/std/max per joint (17×3)
  [106:118] Body geometry — arm/leg extension, torso lean, upper-body area
  [118:120] Social — approach velocity toward nearest person
  [120:128] Temporal — early/late approach and arm extension, initiation index
  + 6 intra-video rank features (appended in build_feature_matrix)
  → TOTAL = 128 absolute + 6 rank = 134

Note on strike features:
  Wrist-speed spike detection was tested but removed. In fight surveillance
  footage, wrist keypoints are frequently occluded (low confidence), causing
  strike features to be zero for both aggressors and non-aggressors. This
  zero-value then gets spuriously correlated with non_aggressor by the LR
  polynomial expansion, hurting recall. The intra-video rank features already
  capture relative motion intensity more robustly.

Classifier: StandardScaler → degree-2 interaction PolynomialFeatures → L2 LogisticRegression
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from module3_pose_estimator import FramePose


# ── Keypoint index constants ──────────────────────────────────────────────────

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
L_ANKLE,    R_ANKLE    = 15, 16
NOSE                   = 0

# Feature-vector dimension constants
ABS_DIM   = 120
REL_DIM   = 6
TOTAL_DIM = ABS_DIM + REL_DIM   # 126

# Index references into the 120-d absolute vector (used for ranking)
BBOX_AREA_IDX  = 51
VEL_MEAN_IDX   = 55
VEL_MAX_IDX    = 89
ARM_EXT_L_IDX  = 106
UBODY_AREA_IDX = 116
APPROACH_IDX   = 118


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kp(kps: np.ndarray, idx: int) -> np.ndarray:
    return kps[idx, :2]

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── Per-person feature extraction ────────────────────────────────────────────

def extract_features_one_person(
    poses: List[FramePose],
    frame_width: int = 640,
    frame_height: int = 360,
    all_track_centroids: Optional[Dict[int, List[np.ndarray]]] = None,
    this_track_id: int = -1,
) -> np.ndarray:
    """
    Build a 120-d feature vector for one person over their fight segment.
    Returns np.ndarray of shape (120,).
    """
    valid_poses = [p for p in poses if p.valid]
    if len(valid_poses) == 0:
        return np.zeros(ABS_DIM, dtype=np.float32)

    frame_area = float(frame_width * frame_height)
    kps_seq    = np.stack([p.keypoints for p in valid_poses])   # (T, 17, 3)
    T          = len(valid_poses)

    # ── 1. Keypoint confidence statistics (51-d) ──────────────────────
    conf_seq  = kps_seq[:, :, 2]
    conf_mean = conf_seq.mean(axis=0)   # (17,)
    conf_std  = conf_seq.std(axis=0)
    conf_max  = conf_seq.max(axis=0)

    # ── 2. Bounding box features (4-d) ────────────────────────────────
    bboxes    = np.array([(p.bbox[2]-p.bbox[0], p.bbox[3]-p.bbox[1])
                           for p in valid_poses])
    bb_areas  = (bboxes[:, 0] * bboxes[:, 1]) / frame_area
    bb_ratios = bboxes[:, 0] / (bboxes[:, 1] + 1e-6)
    bbox_feats = np.array([bb_areas.mean(), bb_areas.std(),
                            bb_ratios.mean(), bb_ratios.std()])

    # ── 3. Joint velocity features (51-d) ─────────────────────────────
    xy_seq = kps_seq[:, :, :2]
    if T > 1:
        vel      = np.linalg.norm(np.diff(xy_seq, axis=0), axis=2)  # (T-1, 17)
        vel_mean = vel.mean(axis=0)
        vel_std  = vel.std(axis=0)
        vel_max  = vel.max(axis=0)
    else:
        vel_mean = np.zeros(17, dtype=np.float32)
        vel_std  = np.zeros(17, dtype=np.float32)
        vel_max  = np.zeros(17, dtype=np.float32)

    # ── 4. Body geometry features (12-d) ──────────────────────────────
    arm_ext_l, arm_ext_r = [], []
    leg_ext_l, leg_ext_r = [], []
    torso_angles          = []
    upper_body_areas      = []

    for kps in kps_seq:
        for side, (sh, el, wr) in [("L", (L_SHOULDER, L_ELBOW, L_WRIST)),
                                    ("R", (R_SHOULDER, R_ELBOW, R_WRIST))]:
            sh_pt, el_pt, wr_pt = _kp(kps, sh), _kp(kps, el), _kp(kps, wr)
            ratio = _dist(wr_pt, sh_pt) / (_dist(el_pt, sh_pt) + 1e-6)
            (arm_ext_l if side == "L" else arm_ext_r).append(ratio)

        for side, (hip, kn, an) in [("L", (L_HIP, L_KNEE, L_ANKLE)),
                                     ("R", (R_HIP, R_KNEE, R_ANKLE))]:
            hip_pt, kn_pt, an_pt = _kp(kps, hip), _kp(kps, kn), _kp(kps, an)
            ratio = _dist(an_pt, hip_pt) / (_dist(kn_pt, hip_pt) + 1e-6)
            (leg_ext_l if side == "L" else leg_ext_r).append(ratio)

        sh_mid  = (_kp(kps, L_SHOULDER) + _kp(kps, R_SHOULDER)) / 2
        hip_mid = (_kp(kps, L_HIP)      + _kp(kps, R_HIP))      / 2
        delta   = sh_mid - hip_mid
        torso_angles.append(float(np.arctan2(delta[0], delta[1] + 1e-6)))

        pts  = np.array([_kp(kps, L_WRIST), _kp(kps, R_WRIST), _kp(kps, NOSE)])
        area = 0.5 * abs(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
        upper_body_areas.append(area / frame_area)

    struct_feats = np.array([
        np.mean(arm_ext_l), np.std(arm_ext_l),
        np.mean(arm_ext_r), np.std(arm_ext_r),
        np.mean(leg_ext_l), np.std(leg_ext_l),
        np.mean(leg_ext_r), np.std(leg_ext_r),
        np.mean(torso_angles), np.std(torso_angles),
        np.mean(upper_body_areas), np.std(upper_body_areas),
    ])

    # ── 5. Social feature — approach velocity (2-d) ───────────────────
    if all_track_centroids is not None and T > 1:
        my_centroids = np.array([
            [(p.bbox[0]+p.bbox[2])/2, (p.bbox[1]+p.bbox[3])/2]
            for p in valid_poses
        ])
        approach_vels = []
        for oid, oc in all_track_centroids.items():
            if oid == this_track_id:
                continue
            ml = min(len(my_centroids), len(oc))
            if ml < 2:
                continue
            dists    = np.linalg.norm(my_centroids[:ml] - np.array(oc[:ml]), axis=1)
            approach = -np.diff(dists)
            approach_vels.extend(approach.tolist())
        social_feats = np.array([
            np.mean(approach_vels) if approach_vels else 0.0,
            np.max(approach_vels)  if approach_vels else 0.0,
        ])
    else:
        social_feats = np.zeros(2, dtype=np.float32)

    # ── Assemble 120-d vector ─────────────────────────────────────────
    feature_vec = np.concatenate([
        conf_mean,     # [0:17]    17-d
        conf_std,      # [17:34]   17-d
        conf_max,      # [34:51]   17-d
        bbox_feats,    # [51:55]   4-d
        vel_mean,      # [55:72]   17-d
        vel_std,       # [72:89]   17-d
        vel_max,       # [89:106]  17-d
        struct_feats,  # [106:118] 12-d
        social_feats,  # [118:120] 2-d
    ]).astype(np.float32)

    assert len(feature_vec) == ABS_DIM,         f"Expected {ABS_DIM}-d, got {len(feature_vec)}"
    return feature_vec


# ── Feature matrix construction ───────────────────────────────────────────────

def build_feature_matrix(
    all_pose_results: Dict[str, Dict[int, List[FramePose]]],
    frame_width: int = 640,
    frame_height: int = 360,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert all pose results to (X, y, sample_ids).
    X shape: (N_samples, 126) = 120 absolute + 6 intra-video rank features.

    Rank features (percentile within the video, 0=lowest, 1=highest):
      vel_mean_rank, vel_max_rank, approach_rank,
      arm_ext_rank, bbox_area_rank, upper_body_area_rank

    Rationale: aggressor is almost always the most active person in the
    video. Absolute values vary with camera distance; percentile rank is
    stable across videos with different resolutions and distances.
    """
    from scipy.stats import rankdata

    def rank_scalars(v: np.ndarray) -> np.ndarray:
        return np.array([
            v[VEL_MEAN_IDX:VEL_MEAN_IDX+17].mean(),
            v[VEL_MAX_IDX:VEL_MAX_IDX+17].max(),
            v[APPROACH_IDX],
            v[ARM_EXT_L_IDX],
            v[BBOX_AREA_IDX],
            v[UBODY_AREA_IDX],
        ], dtype=np.float32)

    X_rows, y_rows, ids = [], [], []

    for video_name, track_dict in all_pose_results.items():

        track_centroids: Dict[int, List[np.ndarray]] = {
            tid: [np.array([(p.bbox[0]+p.bbox[2])/2, (p.bbox[1]+p.bbox[3])/2])
                  for p in poses if p.valid]
            for tid, poses in track_dict.items()
        }

        # Pass 1: absolute features
        video_abs:    Dict[int, np.ndarray] = {}
        video_labels: Dict[int, int]        = {}
        for tid, poses in track_dict.items():
            if not poses:
                continue
            video_labels[tid] = 1 if poses[0].label == "aggressor" else 0
            video_abs[tid]    = extract_features_one_person(
                poses, frame_width, frame_height, track_centroids, tid,
            )

        if not video_abs:
            continue

        # Pass 2: intra-video rank
        tids    = list(video_abs.keys())
        n       = len(tids)
        scalars = np.stack([rank_scalars(video_abs[t]) for t in tids])

        if n > 1:
            rel_ranks = np.apply_along_axis(
                lambda col: (rankdata(col) - 1) / (n - 1),
                axis=0, arr=scalars,
            ).astype(np.float32)
        else:
            rel_ranks = np.full((1, REL_DIM), 0.5, dtype=np.float32)

        # Pass 3: collect
        for i, tid in enumerate(tids):
            full = np.concatenate([video_abs[tid], rel_ranks[i]])
            assert full.shape == (TOTAL_DIM,), \
                f"Dim mismatch: {video_name} track {tid} → {full.shape}"
            X_rows.append(full)
            y_rows.append(video_labels[tid])
            ids.append(f"{video_name}_track{tid}")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, ids


# ── Classifier ────────────────────────────────────────────────────────────────

def build_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("clf",    LogisticRegression(
                       C=0.1,
                       class_weight="balanced",
                       max_iter=2000,
                       solver="lbfgs",
                       random_state=42,
                   )),
    ])


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Optional[Path] = None,
) -> Pipeline:
    print(f"[Classifier] Training on {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} features")
    print(f"  Label distribution: aggressor={y_train.sum()}, "
          f"non_aggressor={(y_train==0).sum()}")

    pipeline = build_classifier()
    pipeline.fit(X_train, y_train)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"[Classifier] Model saved → {save_path}")

    return pipeline


def load_classifier(path: str) -> Pipeline:
    return joblib.load(path)


def print_top_features(pipeline: Pipeline, top_n: int = 10):
    scaler = pipeline.named_steps["scaler"]
    poly   = pipeline.named_steps["poly"]
    clf    = pipeline.named_steps["clf"]
    coef   = clf.coef_[0]
    names  = poly.get_feature_names_out(
        [f"f{i}" for i in range(scaler.n_features_in_)]
    )
    top = np.argsort(np.abs(coef))[-top_n:][::-1]
    print("\n[Feature importance] Top coefficients (positive = aggressor):")
    for i in top:
        print(f"  {names[i]:40s}  coef={coef[i]:+.4f}")
