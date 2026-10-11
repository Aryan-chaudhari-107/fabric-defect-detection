"""English: Two-stage inference pipeline for full fabric strip images.
Hindi: Puri fabric strip images ke liye two-stage inference pipeline.

Aggregation strategy, stated explicitly because it is a modelling decision and
not an implementation detail:

Stage 1 (Normal vs Defective)
    Every 256x256 patch is scored independently. A patch is flagged defective
    when its P(Defective) exceeds ``STAGE1_DEFECT_PROB_THRESHOLD``. The image
    is called defective when at least ``STAGE1_MIN_DEFECTIVE_PATCHES`` patches
    are flagged. The image-level stage-1 confidence is the mean probability
    over the flagged patches, not over all patches, so a single defect in a
    16-patch strip is not diluted to 6%.

Stage 2 (defect type)
    Stage 2 runs only on the flagged patches. Each flagged patch contributes a
    softmax vector. These vectors are averaged, weighted by that patch's stage-1
    defective probability, and the arg-max is the image-level defect type. The
    reported confidence is the mean class probability across the flagged
    patches.

Using the arg-max of a single most-confident patch was rejected: a strip
contains one defect but fifteen clean patches, and those clean patches are the
majority vote, so a single-patch decision would ignore that evidence.

Localization
    U-Net runs on every flagged patch. Each patch mask is pasted back into the
    original image coordinate system and the padded right edge of a 3796 px
    image is cropped away, so the returned mask matches the uploaded image
    exactly.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torchvision import transforms

from config import (
    CLASSIFIER_MODEL_DIR,
    DEVICE,
    PATCH_SIZE,
    PATCH_STRIDE,
    SEGMENTER_MODEL_DIR,
    STAGE1_DEFECT_PROB_THRESHOLD,
    STAGE1_MIN_DEFECTIVE_PATCHES,
    checkpoint_path,
)
from labels import BINARY_DEFECTIVE, DEFECT_TYPE_INDEX_TO_CLASS
from models import CustomCNN, UNet, build_efficientnet
from utils import (
    iter_patch_origins,
    mask_bounding_box,
    overlay_mask,
    pad_to_patch_grid,
    read_image,
    slice_patch,
)


class ModelNotTrainedError(FileNotFoundError):
    """English: Raised when a checkpoint needed for inference is missing.
    Hindi: Jab inference ke liye zaroori checkpoint missing ho to ye error aata hai.
    """


@dataclass
class PatchPrediction:
    """English: Stage-1 result for a single patch.
    Hindi: Ek patch ka stage-1 result.
    """

    x: int
    y: int
    defect_probability: float
    is_defective: bool
    defect_class: Optional[str] = None
    type_confidence: Optional[float] = None
    defect_pixel_ratio: Optional[float] = None


@dataclass
class InferenceResult:
    """English: Full image-level prediction returned by this module.
    Hindi: Ye module jo poora image-level prediction return karta hai.
    """

    is_defective: bool
    defect_class: str
    confidence: float
    stage1_defect_probability: float
    patch_predictions: List[PatchPrediction] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    probability_mask: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    bounding_box: Optional[Dict[str, int]] = None

    def summary(self) -> Dict:
        """English: A JSON-friendly summary for logs and reports.
        Hindi: Logs aur reports ke liye JSON-friendly summary.
        """
        return {
            "is_defective": self.is_defective,
            "defect_class": self.defect_class,
            "confidence": round(float(self.confidence), 6),
            "stage1_defect_probability": round(
                float(self.stage1_defect_probability), 6
            ),
            "defective_patch_count": sum(
                1 for patch in self.patch_predictions if patch.is_defective
            ),
            "total_patches": len(self.patch_predictions),
            "defect_pixel_ratio": (
                round(float(self.mask.mean()), 8) if self.mask is not None else None
            ),
            "bounding_box": self.bounding_box,
        }


def classifier_transform():
    """English: Transform applied to classifier inputs.
    Hindi: Classifier input ke liye transform. Ye training wahi hai, isliye
    inference aur training consistent rahenge.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_classifier(
    stage: str,
    model_name: str,
    best: bool = True,
    device: torch.device = DEVICE,
):
    """English: Load a trained classifier checkpoint for inference.
    Hindi: Classifier model load karke inference ke liye ready karta hai.

    English: The architecture and class list are read from the checkpoint, so
    a mismatch between the training configuration and the inference request
    produces a clear error instead of a silent shape failure deep inside a
    forward pass.
    Hindi: Architecture aur class list checkpoint se padhi jati hai, isliye
    training aur inference ki configuration mismatch ho par saaf error aata hai.
    """
    path = checkpoint_path(stage, model_name, best)
    if not path.exists():
        raise ModelNotTrainedError(
            f"Missing checkpoint: {path}\n"
            f"Train it first:\n  python src/train_classifier.py --stage {stage} "
            f"--model {model_name}\n"
            "Hindi: Pehle model train karein, phir inference karein."
        )

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load classifier checkpoint from {path}: {error}\n"
            "Hindi: Classifier checkpoint load karne mein error aaya. File corrupt ho sakti hai."
        ) from error

    expected_stage = checkpoint.get("stage")
    if expected_stage and expected_stage != stage:
        raise ValueError(
            f"Checkpoint {path.name} was trained for stage '{expected_stage}', "
            f"but stage '{stage}' was requested."
        )

    num_classes = checkpoint["num_classes"]
    if model_name == "cnn":
        model = CustomCNN(num_classes)
    else:
        model = build_efficientnet(num_classes, pretrained=False)

    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint {path.name} does not match the '{model_name}' "
            f"architecture.\n{error}\n"
            "Hindi: Ye checkpoint isi architecture ka nahi hai."
        ) from error

    model.to(device)
    model.eval()
    return model, checkpoint


def load_segmenter(
    best: bool = True,
    device: torch.device = DEVICE,
):
    """English: Load the U-Net checkpoint for inference.
    Hindi: U-Net model load karta hai.
    """
    path = SEGMENTER_MODEL_DIR / f"unet_{'best' if best else 'last'}.pt"
    if not path.exists():
        raise ModelNotTrainedError(
            f"Missing checkpoint: {path}\n"
            "Train it first:\n  python src/train_segmenter.py\n"
            "Hindi: Pehle `python src/train_segmenter.py` chalayein."
        )

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load segmenter checkpoint from {path}: {error}\n"
            "Hindi: Segmenter checkpoint load karne mein error aaya. File corrupt ho sakti hai."
        ) from error

    model = UNet(in_channels=3, out_channels=1)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint {path.name} does not match the UNet architecture.\n{error}\n"
            "Hindi: Ye checkpoint UNet architecture ka nahi hai."
        ) from error
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def stage1_scores(
    model,
    image: np.ndarray,
    device: torch.device = DEVICE,
) -> List[Tuple[int, int, float]]:
    """English: P(Defective) for every patch of a strip image.
    Hindi: Har patch ke liye P(Defective) probability.
    """
    height, width = image.shape[:2]
    padded = pad_to_patch_grid(image, PATCH_SIZE)
    transform = classifier_transform()

    scores: List[Tuple[int, int, float]] = []
    for x, y in iter_patch_origins(width, height, PATCH_SIZE, PATCH_STRIDE):
        patch = slice_patch(padded, x, y, copy=True)
        tensor = transform(patch).unsqueeze(0).to(device)
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        scores.append((x, y, float(probabilities[BINARY_DEFECTIVE].item())))

    return scores


@torch.no_grad()
def stage2_distribution(
    model,
    patches: List[np.ndarray],
    weights: List[float],
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, float]:
    """English: Weighted average stage-2 softmax distribution.
    Hindi: Weighted average stage-2 softmax distribution.
    """
    if not patches:
        return np.zeros(0), 0.0

    transform = classifier_transform()
    stacked = torch.stack(
        [transform(patch) for patch in patches]
    ).to(device)

    probabilities = torch.softmax(model(stacked), dim=1).cpu().numpy()

    weight_array = np.asarray(weights, dtype=np.float64)
    total = weight_array.sum()
    if total <= 0:
        weight_array = np.ones_like(weight_array)
        total = weight_array.sum()

    averaged = (probabilities * weight_array[:, None]).sum(axis=0) / total
    index = int(averaged.argmax())
    confidence = float(averaged[index])
    return averaged, confidence


@torch.no_grad()
def segment_image(
    model,
    image: np.ndarray,
    threshold: float = 0.5,
    only_patches: Optional[List[Tuple[int, int]]] = None,
    min_component_area: int = 16,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """English: Predict a defect mask for a full strip image.
    Hindi: Puri image ka defect mask predict karta hai.

    Returns ``(binary_mask, probability_mask)``, both cropped back to the
    original image size. For the 3796 px wide image the padded right edge is
    removed, so the returned mask aligns exactly with the uploaded image.
    """
    import cv2

    height, width = image.shape[:2]
    padded = pad_to_patch_grid(image, PATCH_SIZE)

    binary_mask = np.zeros(padded.shape[:2], dtype=np.uint8)
    probability_mask = np.zeros(padded.shape[:2], dtype=np.float32)
    transform = classifier_transform()

    wanted = set(only_patches) if only_patches is not None else None

    for x, y in iter_patch_origins(width, height, PATCH_SIZE, PATCH_STRIDE):
        if wanted is not None and (x, y) not in wanted:
            continue

        patch = slice_patch(padded, x, y, copy=True)
        tensor = transform(patch).unsqueeze(0).to(device)

        # English: U-Net returns a full-resolution (1, 1, H, W) probability
        # map, not a scalar. Indexing only the first two dimensions would leave
        # a 256x256 tensor and crash when converted to a float.
        # Hindi: U-Net poore patch ka (1,1,H,W) probability map return karta
        # hai, scalar nahi. Sirf pehle do dimension index karne par 256x256
        # tensor bachta hai jo float mein convert nahi hota.
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        probability = probability.astype(np.float32)

        probability_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = probability
        binary_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = (
            probability > threshold
        ).astype(np.uint8)

    cropped_binary = binary_mask[:height, :width]
    cropped_prob = probability_mask[:height, :width]

    # English: Remove tiny disconnected components that are noise.
    # Hindi: Chhote disconnected noise components ko filter karta hai.
    if min_component_area > 0 and cropped_binary.any():
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cropped_binary, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_component_area:
                cropped_binary[labels == i] = 0

    return cropped_binary, cropped_prob


def predict_image(
    image: np.ndarray,
    stage1_model,
    stage2_model,
    segmenter=None,
    stage1_threshold: float = STAGE1_DEFECT_PROB_THRESHOLD,
    min_defective_patches: int = STAGE1_MIN_DEFECTIVE_PATCHES,
    segmentation_threshold: Optional[float] = None,
    device: torch.device = DEVICE,
) -> InferenceResult:
    """English: Run the full two-stage pipeline on one fabric strip image.
    Hindi: Ek fabric strip image par poora two-stage pipeline chalata hai.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3), got {image.shape}.\n"
            "Hindi: RGB image chahiye, shape (H, W, 3) honi chahiye."
        )

    height, width = image.shape[:2]
    padded = pad_to_patch_grid(image, PATCH_SIZE)

    scores = stage1_scores(stage1_model, image, device=device)
    flagged = [
        (x, y)
        for x, y, probability in scores
        if probability >= stage1_threshold
    ]

    # English: Stage-1 confidence is averaged over the flagged patches only.
    # Averaging over all 16 patches would dilute a single real defect down to
    # roughly 6% and make a confident detection look like noise.
    # Hindi: Stage-1 confidence sirf flagged patches par average hoti hai.
    flagged_probabilities = [
        probability
        for x, y, probability in scores
        if probability >= stage1_threshold
    ]
    stage1_confidence = (
        float(np.mean(flagged_probabilities)) if flagged_probabilities else 0.0
    )

    patch_predictions: List[PatchPrediction] = []
    for x, y, probability in scores:
        patch_predictions.append(
            PatchPrediction(
                x=x,
                y=y,
                defect_probability=probability,
                is_defective=probability >= stage1_threshold,
            )
        )

    is_defective = len(flagged) >= min_defective_patches

    if not is_defective:
        result = InferenceResult(
            is_defective=False,
            defect_class="Normal",
            confidence=1.0 - stage1_confidence,
            stage1_defect_probability=stage1_confidence,
            patch_predictions=patch_predictions,
        )
        return result

    flag_set = set(flagged)
    patches = [
        slice_patch(padded, x, y, copy=True) for x, y in flagged
    ]
    weights = [
        probability
        for x, y, probability in scores
        if (x, y) in flag_set
    ]

    averaged, type_confidence = stage2_distribution(
        stage2_model, patches, weights, device=device
    )

    if averaged.size == 0:
        defect_class, type_confidence = "Unknown", 0.0
    else:
        defect_class = DEFECT_TYPE_INDEX_TO_CLASS[int(averaged.argmax())]

    by_position = {
        (patch.x, patch.y): patch for patch in patch_predictions
    }
    for position, patch in zip(flagged, patches):
        entry = by_position[position]
        entry.defect_class = defect_class
        entry.type_confidence = type_confidence

    result = InferenceResult(
        is_defective=True,
        defect_class=defect_class,
        confidence=type_confidence,
        stage1_defect_probability=stage1_confidence,
        patch_predictions=patch_predictions,
    )

    if segmenter is not None:
        threshold = (
            segmentation_threshold
            if segmentation_threshold is not None
            else 0.5
        )
        binary_mask, probability_mask = segment_image(
            segmenter,
            image,
            threshold=threshold,
            only_patches=flagged,
            device=device,
        )
        result.mask = binary_mask
        result.probability_mask = probability_mask
        result.overlay = overlay_mask(image, binary_mask)
        result.bounding_box = mask_bounding_box(binary_mask)

        for entry in patch_predictions:
            if entry.is_defective:
                region = probability_mask[
                    entry.y:entry.y + PATCH_SIZE, entry.x:entry.x + PATCH_SIZE
                ]
                if region.size:
                    entry.defect_pixel_ratio = float((region > threshold).mean())

    return result


def predict_file(
    path: Path,
    stage1_model,
    stage2_model,
    segmenter=None,
    **kwargs,
) -> InferenceResult:
    """English: Convenience wrapper that reads an image from disk first.
    Hindi: Disk se image padhkar chalane wala convenience wrapper.
    """
    return predict_image(
        read_image(path), stage1_model, stage2_model, segmenter, **kwargs
    )
