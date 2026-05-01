"""
Module 3: Pose Estimator
- Uses YOLOv8-pose (GPU accelerated) to extract 17 COCO keypoints
  from each annotated bounding box crop in the fight frame range
- Outputs a structured dict: { video_name -> { track_id -> list of FramePose } }
- FramePose holds frame index, label, bbox, and 17×3 keypoints array
"""

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO

from module1_data_loader import (
    VideoSample, TrackAnnotation, read_frames_in_range
)
from module2_augmenter import VideoAugmenter


# ──────────────────────────────────────────────
# COCO-17 keypoint index reference
# ──────────────────────────────────────────────
# 0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
# 5:l_shoulder  6:r_shoulder  7:l_elbow  8:r_elbow
# 9:l_wrist  10:r_wrist  11:l_hip  12:r_hip
# 13:l_knee  14:r_knee  15:l_ankle  16:r_ankle

KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
N_KP = 17


# ──────────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────────

@dataclass
class FramePose:
    frame_idx: int
    label: str                              # "aggressor" | "non_aggressor"
    track_id: int
    bbox: Tuple[float, float, float, float] # (xtl, ytl, xbr, ybr) in frame coords
    keypoints: np.ndarray                   # shape (17, 3): x, y, confidence
    valid: bool = True                      # False if pose detection failed


# ──────────────────────────────────────────────
# Pose Estimator
# ──────────────────────────────────────────────

class PoseEstimator:
    """
    Wraps YOLOv8-pose inference.
    Crops each person's bounding box from the frame, runs pose estimation,
    then maps keypoints back to full-frame coordinates.
    """

    def __init__(
        self,
        model_name: str = "yolov8m-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.3,
        padding: float = 0.1,           # fraction of bbox size to pad crops
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf_threshold = conf_threshold
        self.padding = padding

        print(f"[PoseEstimator] Loading {model_name} on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)

    # ── Public API ─────────────────────────────────────────────────────

    def process_sample(
        self,
        sample: VideoSample,
        augmenter: Optional[VideoAugmenter] = None,
        frame_step: int = 1,
    ) -> Dict[int, List[FramePose]]:
        """
        Process one VideoSample.
        Returns { track_id -> [FramePose, ...] }
        """
        # Build lookup: frame_idx -> {track_id -> bbox}
        frame_bbox_map: Dict[int, Dict[int, Tuple]] = {}
        track_label_map: Dict[int, str] = {}

        for track in sample.tracks:
            track_label_map[track.track_id] = track.label
            for frame_idx, bbox in track.boxes.items():
                if frame_idx < sample.frame_start or frame_idx > sample.frame_end:
                    continue
                frame_bbox_map.setdefault(frame_idx, {})[track.track_id] = bbox

        # Read frames in fight range
        frames = read_frames_in_range(
            sample.video_path,
            sample.frame_start,
            sample.frame_end,
            augmenter=augmenter,
            step=frame_step,
        )

        results: Dict[int, List[FramePose]] = {
            t.track_id: [] for t in sample.tracks
        }

        for frame_idx, frame in frames:
            if frame_idx not in frame_bbox_map:
                continue

            for track_id, bbox in frame_bbox_map[frame_idx].items():
                # Mirror bbox if frame was flipped by augmenter
                xtl, ytl, xbr, ybr = bbox
                if augmenter is not None and augmenter.last_flipped:
                    xtl, xbr = augmenter.mirror_box(xtl, xbr)

                frame_pose = self._estimate_one(
                    frame, frame_idx, track_id,
                    track_label_map[track_id],
                    (xtl, ytl, xbr, ybr),
                )
                results[track_id].append(frame_pose)

        return results

    def process_all(
        self,
        samples: List[VideoSample],
        augmenter: Optional[VideoAugmenter] = None,
        frame_step: int = 1,
    ) -> Dict[str, Dict[int, List[FramePose]]]:
        """
        Process a list of VideoSamples.
        Returns { video_name -> { track_id -> [FramePose, ...] } }
        """
        all_results = {}
        for i, sample in enumerate(samples):
            print(f"[PoseEstimator] ({i+1}/{len(samples)}) Processing {sample.name} ...")
            all_results[sample.name] = self.process_sample(
                sample, augmenter=augmenter, frame_step=frame_step
            )
        return all_results

    # ── Internal helpers ───────────────────────────────────────────────

    def _estimate_one(
        self,
        frame: np.ndarray,
        frame_idx: int,
        track_id: int,
        label: str,
        bbox: Tuple[float, float, float, float],
    ) -> FramePose:
        """Crop bbox from frame, run pose model, map kpts back to frame coords."""
        h, w = frame.shape[:2]
        xtl, ytl, xbr, ybr = bbox

        # Pad crop for better pose detection at edges
        pad_x = (xbr - xtl) * self.padding
        pad_y = (ybr - ytl) * self.padding
        x1 = max(0, int(xtl - pad_x))
        y1 = max(0, int(ytl - pad_y))
        x2 = min(w, int(xbr + pad_x))
        y2 = min(h, int(ybr + pad_y))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return FramePose(
                frame_idx=frame_idx, label=label, track_id=track_id,
                bbox=bbox, keypoints=np.zeros((N_KP, 3)), valid=False
            )

        results = self.model(
            crop,
            verbose=False,
            conf=self.conf_threshold,
            device=self.device,
        )

        keypoints = self._extract_keypoints(results, x1, y1)

        return FramePose(
            frame_idx=frame_idx,
            label=label,
            track_id=track_id,
            bbox=(xtl, ytl, xbr, ybr),
            keypoints=keypoints,
            valid=(keypoints is not None and keypoints.sum() > 0),
        )

    def _extract_keypoints(
        self,
        results,
        offset_x: int,
        offset_y: int,
    ) -> np.ndarray:
        """
        Extract the highest-confidence person's keypoints from YOLO results,
        translate from crop-local coords back to full-frame coords.
        Returns np.ndarray of shape (17, 3): [x, y, conf] per keypoint.
        """
        kps_out = np.zeros((N_KP, 3), dtype=np.float32)

        for r in results:
            if r.keypoints is None or len(r.keypoints.data) == 0:
                continue

            kps_tensor = r.keypoints.data  # (n_persons, 17, 3)

            # Pick the person detection with highest mean keypoint confidence
            mean_confs = kps_tensor[:, :, 2].mean(dim=1)
            best_idx = mean_confs.argmax().item()
            kps = kps_tensor[best_idx].cpu().numpy()  # (17, 3)

            # Translate crop-local coords → full frame coords
            kps[:, 0] += offset_x
            kps[:, 1] += offset_y
            kps_out = kps
            break   # one person per crop

        return kps_out


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from module1_data_loader import discover_dataset, split_dataset
    from module2_augmenter import VideoAugmenter

    samples = discover_dataset("data")
    train, test = split_dataset(samples, n_test=3)

    estimator = PoseEstimator()
    aug = VideoAugmenter(seed=42)

    # Process just first training sample as smoke test
    result = estimator.process_sample(train[0], augmenter=aug, frame_step=5)
    for track_id, poses in result.items():
        valid = sum(1 for p in poses if p.valid)
        print(f"  Track {track_id} ({poses[0].label if poses else '?'}): "
              f"{len(poses)} frames, {valid} valid poses")
