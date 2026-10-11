"""English: Backward-compatible inference helpers.
Hindi: Purane inference helpers ke liye backward compatibility layer.

English: The single-stage ``classify_image`` / ``segment_image`` API used by the
first version of the project was inconsistent with training, because it
classified a whole strip with one forward pass per patch and then took a plain
majority vote across patches, including the fifteen clean patches of a strip
that contains one defect. All inference now lives in ``inference.py`` and uses
an explicit weighted aggregation. This module is kept only so existing imports
keep working.
Hindi: Pehle version ka single-stage API training ke saath consistent nahi tha,
kyunki wo har patch ko ek seedha vote deta tha, aur 15 clean patches bhi vote
mein gina jaate the. Ab saara inference ``inference.py`` mein hai.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config import DEVICE, MASK_DIR, SEGMENTATION_THRESHOLD
from inference import (  # noqa: F401  (re-exported for compatibility)
    InferenceResult,
    ModelNotTrainedError,
    PatchPrediction,
    load_classifier,
    load_segmenter,
    predict_file,
    predict_image,
    stage1_scores,
    stage2_distribution,
)
from utils import merged_mask


def segmentation_iou(
    predicted_mask: np.ndarray,
    image_path: Path,
) -> Optional[float]:
    """English: Intersection-over-union against the ground-truth mask.
    Hindi: Ground-truth mask ke against IoU compute karta hai.
    Returns ``None`` when the image has no usable ground-truth mask, which is
    the case for ``0100_025_08`` (no mask) and normal images.
    """
    paths = find_mask(image_path, MASK_DIR)
    if not paths:
        return None

    from utils import read_image

    image = read_image(image_path)
    truth = merged_mask(
        image_path, MASK_DIR, target_shape=image.shape[:2]
    )
    if truth is None:
        return None

    if truth.shape != predicted_mask.shape:
        truth = cv2.resize(
            truth,
            (predicted_mask.shape[1], predicted_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    intersection = np.logical_and(predicted_mask, truth).sum()
    union = np.logical_or(predicted_mask, truth).sum()

    if union == 0:
        return None
    return float(intersection / union)


def segment_image(
    model,
    image: np.ndarray,
    threshold: float = SEGMENTATION_THRESHOLD,
    patch_size: int = 256,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """English: Deprecated thin wrapper around ``inference.segment_image``.
    Hindi: ``inference.segment_image`` ka deprecated thin wrapper.
    """
    from inference import segment_image as _segment_image

    binary_mask, _ = _segment_image(
        model, image, threshold=threshold, only_patches=None, device=device
    )
    return binary_mask
