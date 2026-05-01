"""
Module 2: Video Augmenter
- Applied per-frame at read time (training set only)
- Augmentations: brightness/contrast jitter, horizontal flip,
  Gaussian blur, hue-saturation shift
- All transforms are randomized per call to augmenter.apply()
- Flip is tracked so bounding boxes can be mirrored consistently
"""

import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class AugmentConfig:
    """Tweak these to control augmentation strength."""
    # Probability each augmentation fires
    p_flip:       float = 0.5
    p_brightness: float = 0.7
    p_blur:       float = 0.3
    p_hsv:        float = 0.5
    p_noise:      float = 0.3

    # Magnitude ranges
    brightness_range: Tuple[float, float] = (0.6, 1.4)   # multiplicative
    contrast_range:   Tuple[float, float] = (0.7, 1.3)
    blur_kernel_choices: Tuple[int, ...] = (3, 5)
    hue_shift_range:  Tuple[int, int] = (-15, 15)        # OpenCV hue is 0-179
    sat_scale_range:  Tuple[float, float] = (0.7, 1.3)
    noise_std:        float = 8.0


class VideoAugmenter:
    """
    Stateful per-frame augmenter.  Each call to .apply() independently
    samples all augmentation decisions.

    Usage (training only):
        aug = VideoAugmenter(config, seed=42)
        for frame_idx, frame in read_frames_in_range(..., augmenter=aug):
            ...  # frame already augmented

    To get the flip state after apply() (for mirroring bboxes):
        aug.last_flipped  -> bool
        aug.flip_frame_width -> int
    """

    def __init__(self, config: Optional[AugmentConfig] = None, seed: Optional[int] = None):
        self.config = config or AugmentConfig()
        self.rng = random.Random(seed)
        self.last_flipped: bool = False
        self.flip_frame_width: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply a random combination of augmentations to one BGR frame."""
        cfg = self.config
        h, w = frame.shape[:2]
        self.flip_frame_width = w

        # 1. Horizontal flip
        self.last_flipped = self.rng.random() < cfg.p_flip
        if self.last_flipped:
            frame = cv2.flip(frame, 1)

        # 2. Brightness + Contrast  (operate in float for clean math)
        if self.rng.random() < cfg.p_brightness:
            alpha = self.rng.uniform(*cfg.contrast_range)    # contrast
            beta  = self.rng.uniform(*cfg.brightness_range)  # brightness scale
            frame = np.clip(frame.astype(np.float32) * alpha * beta, 0, 255).astype(np.uint8)

        # 3. Hue / Saturation shift
        if self.rng.random() < cfg.p_hsv:
            frame = self._hsv_jitter(frame)

        # 4. Gaussian blur
        if self.rng.random() < cfg.p_blur:
            k = self.rng.choice(cfg.blur_kernel_choices)
            frame = cv2.GaussianBlur(frame, (k, k), 0)

        # 5. Gaussian noise
        if self.rng.random() < cfg.p_noise:
            noise = np.random.normal(0, cfg.noise_std, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return frame

    def mirror_box(self, xtl: float, xbr: float) -> Tuple[float, float]:
        """
        Mirror a bounding box's x-coordinates if the last frame was flipped.
        Call after apply() when you need the bbox to stay aligned.

        Returns (new_xtl, new_xbr).
        """
        if not self.last_flipped:
            return xtl, xbr
        w = self.flip_frame_width
        new_xtl = w - xbr
        new_xbr = w - xtl
        return new_xtl, new_xbr

    # ── Internal helpers ───────────────────────────────────────────────

    def _hsv_jitter(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.config
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue channel (0-179 in OpenCV)
        hue_delta = self.rng.randint(*cfg.hue_shift_range)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180

        # Saturation channel (0-255)
        sat_scale = self.rng.uniform(*cfg.sat_scale_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────
# Quick visual test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python module2_augmenter.py <video_path>")
        sys.exit(0)

    cap = cv2.VideoCapture(sys.argv[1])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame.")
        sys.exit(1)

    aug = VideoAugmenter()
    variants = [aug.apply(frame.copy()) for _ in range(6)]

    row1 = np.hstack(variants[:3])
    row2 = np.hstack(variants[3:])
    grid = np.vstack([row1, row2])
    grid = cv2.resize(grid, (1280, 480))
    cv2.imshow("Augmentation samples (press any key)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
