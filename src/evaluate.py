"""English: Patch-level, image-level and segmentation evaluation.
Hindi: Patch-level, image-level aur segmentation evaluation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from config import (
    CONFUSION_MATRIX_DIR,
    DEVICE,
    LOCALIZATION_DIR,
    MAX_LOCALIZATION_FIGURES,
    METRICS_DIR,
    NUM_WORKERS,
    PATCH_SIZE,
    SEGMENTER_MODEL_DIR,
    SEGMENTATION_THRESHOLD,
    STAGE1_DEFECT_PROB_THRESHOLD,
    STAGE1_MIN_DEFECTIVE_PATCHES,
    checkpoint_path,
)
from dataset import PatchDataset, SegmentationPatchDataset, load_split
from inference import (
    load_classifier,
    load_segmenter,
    predict_image,
    segment_image,
)
from labels import (
    BINARY_CLASSES,
    DEFECT_TYPE_CLASSES,
    NORMAL_CLASS,
    SINGLE_IMAGE_CLASS_NAMES,
    class_of_image,
)
from models import binary_metrics
from utils import mask_bounding_box, merged_mask, read_image


def boxes_intersect(box1: Optional[Dict[str, int]], box2: Optional[Dict[str, int]]) -> bool:
    """English: Check if two bounding boxes touch or overlap.
    Hindi: Check karta hai ki do bounding boxes aapas mein touch ya overlap karte hain ya nahi.
    """
    if box1 is None or box2 is None:
        return False
    return not (
        box1["x_max"] < box2["x_min"]
        or box1["x_min"] > box2["x_max"]
        or box1["y_max"] < box2["y_min"]
        or box1["y_min"] > box2["y_max"]
    )


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """English: Wilson score interval for a binomial proportion.
    Hindi: Binomial proportion ke liye Wilson score 95% confidence interval.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))
    low = max(0.0, float(center - margin))
    high = min(1.0, float(center + margin))
    return (round(low, 4), round(high, 4))


# English: Coarse groupings for AITEX defect types.
COARSE_GROUPS = {
    "Yarn Defects": ["Broken End", "Broken Yarn", "Broken Pick"],
    "Weave Defects": ["Weft Curling", "Crease", "Weft Crack", "Cut Selvage"],
    "Surface Defects": ["Fuzzyball", "Warp Ball", "Knots", "Contamination", "Nep"],
}


def classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> Dict:
    """English: Compute classification metrics over all classes, with honesty for low-support classes.
    Hindi: Saari classes par classification metrics compute karta hai aur low-support classes ko honestly isolate karta hai.
    """
    all_indices = list(range(len(class_names)))
    present = sorted(set(true_labels.tolist()) | set(predictions.tolist()))

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=all_indices,
        zero_division=0,
    )

    per_class = {}
    supported_indices = []
    for index, name in enumerate(class_names):
        supp = int(support[index])
        is_supported = supp >= 3
        if is_supported:
            supported_indices.append(index)
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": supp,
            "present_in_split": bool(supp > 0),
            "sufficient_data": is_supported,
        }

    matrix = confusion_matrix(true_labels, predictions, labels=all_indices)

    # Honest averages: only over classes with support >= 3
    if supported_indices:
        supp_prec = float(np.mean([precision[i] for i in supported_indices]))
        supp_rec = float(np.mean([recall[i] for i in supported_indices]))
        supp_f1 = float(np.mean([f1[i] for i in supported_indices]))
    else:
        supp_prec, supp_rec, supp_f1 = 0.0, 0.0, 0.0

    result = {
        "samples": int(len(true_labels)),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predictions)),
        "macro_precision_all_classes": float(precision.mean()),
        "macro_recall_all_classes": float(recall.mean()),
        "macro_f1_all_classes": float(f1.mean()),
        "macro_precision_support_ge3": supp_prec,
        "macro_recall_support_ge3": supp_rec,
        "macro_f1_support_ge3": supp_f1,
        "supported_classes_count": len(supported_indices),
        "insufficient_data_classes": [
            class_names[index] for index in all_indices if support[index] < 3
        ],
        "weighted_f1": float(
            np.average(f1, weights=support) if support.sum() else 0.0
        ),
        "present_classes": [class_names[index] for index in present],
        "absent_classes": [
            class_names[index] for index in all_indices if support[index] == 0
        ],
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
    return result


def save_confusion_matrix(matrix: np.ndarray, class_names: List[str], name: str) -> None:
    """English: Save a confusion matrix figure.
    Hindi: Confusion matrix ka graph save karta hai.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(1 + 0.7 * len(class_names),
                                        1 + 0.6 * len(class_names)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Confusion Matrix ({name})")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value == 0:
                continue
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_DIR / f"{name}_confusion_matrix.png", dpi=130)
    plt.close(figure)


@torch.no_grad()
def evaluate_patch_classifier(
    stage: str,
    model,
    dataset: PatchDataset,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """English: Patch-level predictions over a dataset.
    Hindi: Dataset par patch-level predictions.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    labels: List[int] = []
    predictions: List[int] = []
    confidences: List[float] = []

    for images, targets in loader:
        images = images.to(DEVICE)
        probabilities = torch.softmax(model(images), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        predictions.extend(predicted.cpu().numpy())
        labels.extend(targets.numpy())
        confidences.extend(confidence.cpu().numpy())

    return (
        np.array(labels),
        np.array(predictions),
        np.array(confidences),
    )


@torch.no_grad()
def evaluate_segmentation_patches(model, dataset: SegmentationPatchDataset) -> Dict:
    """English: Patch-level segmentation metrics on Dice, IoU, precision, recall.
    Hindi: Patch-level segmentation metrics: Dice, IoU, precision, recall.
    """
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    total_items = 0

    model.eval()
    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        logits = model(images)
        metrics = binary_metrics(logits, masks, SEGMENTATION_THRESHOLD)
        batch_size = images.size(0)
        for key, value in metrics.items():
            totals[key] += value * batch_size
        total_items += batch_size

    divisor = max(total_items, 1)
    return {key: value / divisor for key, value in totals.items()}


def boxes_intersect(box1: Optional[Dict[str, int]], box2: Optional[Dict[str, int]]) -> bool:
    """English: Check if two bounding boxes touch or overlap.
    Hindi: Check karta hai ki do bounding boxes aapas mein touch ya overlap karte hain ya nahi.
    """
    if box1 is None or box2 is None:
        return False
    return not (
        box1["x_max"] < box2["x_min"]
        or box1["x_min"] > box2["x_max"]
        or box1["y_max"] < box2["y_min"]
        or box1["y_min"] > box2["y_max"]
    )


@torch.no_grad()
def evaluate_image_level(
    stage1_model,
    stage2_model,
    image_paths: List[Path],
    stage1_threshold: float,
    min_defective_patches: int,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """English: Image-level binary, per-roll, misclassification and defect-type metrics.
    Hindi: Image-level binary, per-roll, misclassification aur defect-type metrics.
    """
    from labels import fabric_code_of_image

    binary_true: List[int] = []
    binary_pred: List[int] = []
    type_true: List[int] = []
    type_pred: List[int] = []
    rows: List[Dict] = []
    misclassified_rows: List[Dict] = []

    per_roll_records: Dict[str, Dict] = {}

    for path in image_paths:
        truth = class_of_image(path)
        if truth is None:
            continue

        roll = fabric_code_of_image(path) or "unknown"
        if roll not in per_roll_records:
            per_roll_records[roll] = {"true": [], "pred": []}

        image = read_image(path)
        result = predict_image(
            image,
            stage1_model,
            stage2_model,
            segmenter=None,
            stage1_threshold=stage1_threshold,
            min_defective_patches=min_defective_patches,
        )

        is_defective_truth = 0 if truth == NORMAL_CLASS else 1
        binary_true.append(is_defective_truth)
        binary_pred.append(1 if result.is_defective else 0)

        per_roll_records[roll]["true"].append(is_defective_truth)
        per_roll_records[roll]["pred"].append(1 if result.is_defective else 0)

        if is_defective_truth == 1 and result.is_defective:
            from labels import DEFECT_TYPE_TO_INDEX

            type_true.append(DEFECT_TYPE_TO_INDEX.get(truth, -1))
            type_pred.append(
                DEFECT_TYPE_TO_INDEX.get(result.defect_class, -1)
            )

        is_correct = bool(
            (is_defective_truth == 1 and result.is_defective)
            or (is_defective_truth == 0 and not result.is_defective)
        )

        max_patch_prob = (
            max((p.defect_probability for p in result.patch_predictions), default=0.0)
        )
        flagged_count = sum(1 for p in result.patch_predictions if p.is_defective)

        record_entry = {
            "image": path.name,
            "fabric_roll": roll,
            "true_class": truth,
            "predicted": "Defective" if result.is_defective else NORMAL_CLASS,
            "predicted_class": result.defect_class,
            "confidence": round(float(result.confidence), 6),
            "stage1_probability": round(
                float(result.stage1_defect_probability), 6
            ),
            "max_patch_defect_prob": round(float(max_patch_prob), 4),
            "flagged_patches_count": flagged_count,
            "correct": is_correct,
        }
        rows.append(record_entry)

        if not is_correct:
            misclassified_rows.append({
                "image": path.name,
                "fabric_roll": roll,
                "true_class": truth,
                "predicted": record_entry["predicted"],
                "predicted_defect_type": result.defect_class,
                "max_patch_defect_prob": round(float(max_patch_prob), 4),
                "flagged_patches_count": flagged_count,
                "stage1_threshold_used": stage1_threshold,
                "min_patches_required": min_defective_patches,
            })

    binary_metrics_dict = classification_metrics(
        np.array(binary_true), np.array(binary_pred), BINARY_CLASSES
    )
    cm = np.array(binary_metrics_dict["confusion_matrix"])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    recall = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(tn + fp, 1))
    tnr = float(tn / max(tn + fp, 1))

    recall_ci95 = wilson_interval(tp, tp + fn)
    fpr_ci95 = wilson_interval(fp, tn + fp)
    tnr_ci95 = wilson_interval(tn, tn + fp)
    bal_acc_ci95 = [
        round((recall_ci95[0] + tnr_ci95[0]) / 2.0, 4),
        round((recall_ci95[1] + tnr_ci95[1]) / 2.0, 4),
    ]

    binary_metrics_dict["defect_recall"] = recall
    binary_metrics_dict["defect_recall_ci95"] = recall_ci95
    binary_metrics_dict["normal_false_positive_rate"] = fpr
    binary_metrics_dict["normal_fpr_ci95"] = fpr_ci95
    binary_metrics_dict["balanced_accuracy_ci95"] = bal_acc_ci95
    binary_metrics_dict["tp"] = tp
    binary_metrics_dict["fp"] = fp
    binary_metrics_dict["tn"] = tn
    binary_metrics_dict["fn"] = fn

    # Per fabric roll summary
    roll_metrics = {}
    for roll, data in sorted(per_roll_records.items()):
        y_t = np.array(data["true"])
        y_p = np.array(data["pred"])
        r_cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        r_tn, r_fp, r_fn, r_tp = int(r_cm[0, 0]), int(r_cm[0, 1]), int(r_cm[1, 0]), int(r_cm[1, 1])
        roll_metrics[f"roll_{roll}"] = {
            "total_images": len(y_t),
            "defective_images": int((y_t == 1).sum()),
            "normal_images": int((y_t == 0).sum()),
            "tp": r_tp,
            "fp": r_fp,
            "tn": r_tn,
            "fn": r_fn,
            "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_t, y_p)), 4),
            "defect_recall": round(float(r_tp / max(r_tp + r_fn, 1)), 4),
            "normal_fpr": round(float(r_fp / max(r_tn + r_fp, 1)), 4),
        }
    binary_metrics_dict["per_fabric_roll"] = roll_metrics

    type_metrics_dict = None
    if type_true:
        valid = [
            index
            for index, value in enumerate(type_true)
            if value >= 0 and type_pred[index] >= 0
        ]
        if valid:
            type_metrics_dict = classification_metrics(
                np.array([type_true[index] for index in valid]),
                np.array([type_pred[index] for index in valid]),
                DEFECT_TYPE_CLASSES,
            )

    return (
        {"binary": binary_metrics_dict, "defect_type": type_metrics_dict},
        rows,
        misclassified_rows,
    )


def evaluate_localization(
    segmenter,
    image_paths: List[Path],
    stage1_threshold: float,
    min_defective_patches: int,
    stage1_model,
    stage2_model,
) -> Dict:
    """English: Image-level localization metrics against ground-truth masks with box-touch analysis.
    Hindi: Ground-truth masks ke against image-level localization metrics aur box-touch analysis.
    """
    rows: List[Dict] = []
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    scored = 0
    box_touches = 0

    for path in image_paths:
        if class_of_image(path) == NORMAL_CLASS:
            continue

        image = read_image(path)
        height, width = image.shape[:2]
        truth = merged_mask(path, target_shape=(height, width))
        if truth is None:
            continue

        result = predict_image(
            image,
            stage1_model,
            stage2_model,
            segmenter=segmenter,
            stage1_threshold=stage1_threshold,
            min_defective_patches=min_defective_patches,
        )
        if result.mask is None:
            continue

        predicted = result.mask
        intersection = float(np.logical_and(predicted, truth).sum())
        predicted_sum = float(predicted.sum())
        truth_sum = float(truth.sum())

        dice = (2 * intersection + 1.0) / (predicted_sum + truth_sum + 1.0)
        union = predicted_sum + truth_sum - intersection
        iou = (intersection + 1.0) / (union + 1.0)
        precision = (intersection + 1.0) / (predicted_sum + 1.0)
        recall = (intersection + 1.0) / (truth_sum + 1.0)

        gt_box = mask_bounding_box(truth)
        touch = boxes_intersect(result.bounding_box, gt_box)
        if touch:
            box_touches += 1

        for key, value in (
            ("dice", dice),
            ("iou", iou),
            ("precision", precision),
            ("recall", recall),
        ):
            totals[key] += value

        scored += 1
        rows.append(
            {
                "image": path.name,
                "dice": round(dice, 6),
                "iou": round(iou, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "box_touch": bool(touch),
                "bounding_box": result.bounding_box,
                "gt_box": gt_box,
            }
        )

        if scored <= MAX_LOCALIZATION_FIGURES:
            save_localization_figure(image, truth, predicted, path.stem)

    summary = {
        "scored_images": scored,
        "skipped_images": len(
            [p for p in image_paths if merged_mask(p) is None]
        ),
        "box_touch_rate": round(float(box_touches / max(scored, 1)), 4),
    }
    for key, value in totals.items():
        summary[f"mean_{key}"] = round(float(value / scored), 6) if scored else 0.0

    return {"summary": summary, "per_image": rows}


def save_localization_figure(
    image: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    name: str,
) -> None:
    """English: Save a qualitative localization comparison.
    Hindi: Qualitative localization comparison save karta hai.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils import overlay_mask

    figure, axes = plt.subplots(3, 1, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(truth, cmap="gray")
    axes[1].set_title("Ground truth mask")
    axes[2].imshow(overlay_mask(image, predicted))
    axes[2].set_title("Predicted overlay")
    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(LOCALIZATION_DIR / f"{name}.png", dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the two-stage pipeline on the test split"
    )
    parser.add_argument(
        "--stage",
        choices=["binary", "defect_type", "all"],
        default="all",
        help="Which classifier to evaluate. 'all' evaluates both.",
    )
    parser.add_argument("--model", choices=["cnn", "efficientnet"], default="cnn")
    parser.add_argument("--defect-threshold", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--stage1-prob-threshold",
        type=float,
        default=None,
        help="Stage 1 defect probability threshold (overrides tuned thresholds)",
    )
    parser.add_argument(
        "--min-defective-patches",
        type=int,
        default=None,
        help="Minimum defective patches to call an image defective (overrides tuned thresholds)",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip localization metrics even if a U-Net checkpoint exists.",
    )
    args = parser.parse_args()

    # English: Load tuned validation thresholds if available, unless overridden via CLI
    # Hindi: Agar tuned thresholds file available hai toh use karein, jab tak CLI se override na kiya ho
    tuned_file = METRICS_DIR / f"stage1_thresholds_{args.model}.json"
    tuned_data = None
    if tuned_file.exists():
        try:
            with tuned_file.open("r", encoding="utf-8") as f:
                tuned_data = json.load(f)
        except Exception:
            tuned_data = None

    if args.stage1_prob_threshold is not None:
        stage1_prob_threshold = args.stage1_prob_threshold
        prob_source = "CLI override"
    elif tuned_data and "best_stage1_prob_threshold" in tuned_data:
        stage1_prob_threshold = float(tuned_data["best_stage1_prob_threshold"])
        prob_source = f"tuned validation threshold ({tuned_file.name})"
    else:
        stage1_prob_threshold = STAGE1_DEFECT_PROB_THRESHOLD
        prob_source = "config default"

    if args.min_defective_patches is not None:
        min_defective_patches = args.min_defective_patches
        min_p_source = "CLI override"
    elif tuned_data and "best_min_defective_patches" in tuned_data:
        min_defective_patches = int(tuned_data["best_min_defective_patches"])
        min_p_source = f"tuned validation threshold ({tuned_file.name})"
    else:
        min_defective_patches = STAGE1_MIN_DEFECTIVE_PATCHES
        min_p_source = "config default"

    print(
        f"\n[Image-level Decision Thresholds] model={args.model}\n"
        f"  P(defective) threshold: {stage1_prob_threshold:.2f} ({prob_source})\n"
        f"  Min defective patches:  {min_defective_patches} ({min_p_source})"
    )

    splits = load_split()
    test_paths = splits["test"]

    stage1_model, _ = load_classifier("binary", args.model, best=True)
    stage2_model = None
    if args.stage in {"defect_type", "all"}:
        stage2_model, _ = load_classifier("defect_type", args.model, best=True)

    report: Dict = {
        "model": args.model,
        "image_level_thresholds": {
            "stage1_prob_threshold": stage1_prob_threshold,
            "stage1_prob_threshold_source": prob_source,
            "min_defective_patches": min_defective_patches,
            "min_defective_patches_source": min_p_source,
        },
        "stages": {},
    }

    if args.stage in {"binary", "all"}:
        dataset = PatchDataset(
            test_paths, stage="binary", train=False, threshold=args.defect_threshold
        )
        labels, predictions, confidences = evaluate_patch_classifier(
            "binary", stage1_model, dataset, args.batch_size
        )
        metrics = classification_metrics(labels, predictions, BINARY_CLASSES)
        metrics["mean_confidence"] = float(confidences.mean())
        report["stages"]["binary_patch_level"] = metrics
        save_confusion_matrix(
            np.array(metrics["confusion_matrix"]), BINARY_CLASSES, "binary_patch"
        )

        print("=== Binary classifier, patch level ===")
        print(
            classification_report(
                labels,
                predictions,
                labels=list(range(len(BINARY_CLASSES))),
                target_names=BINARY_CLASSES,
                zero_division=0,
            )
        )

    if args.stage in {"defect_type", "all"} and stage2_model is not None:
        dataset = PatchDataset(
            test_paths,
            stage="defect_type",
            train=False,
            threshold=args.defect_threshold,
        )
        labels, predictions, confidences = evaluate_patch_classifier(
            "defect_type", stage2_model, dataset, args.batch_size
        )
        metrics = classification_metrics(labels, predictions, DEFECT_TYPE_CLASSES)
        metrics["mean_confidence"] = float(confidences.mean())
        report["stages"]["defect_type_patch_level"] = metrics
        save_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            DEFECT_TYPE_CLASSES,
            "defect_type_patch",
        )

        print("\n=== Defect type classifier, patch level ===")
        # English: ``labels`` must be passed explicitly. Only 10 of the 12
        # defect types appear in the test split, and letting sklearn infer the
        # label set makes ``target_names`` length mismatch.
        # Hindi: ``labels`` explicitly dena zaroori hai, warna 12 mein se sirf
        # 10 classes test split mein hain aur target_names mismatch ho jata hai.
        print(
            classification_report(
                labels,
                predictions,
                labels=list(range(len(DEFECT_TYPE_CLASSES))),
                target_names=DEFECT_TYPE_CLASSES,
                zero_division=0,
            )
        )
        print("Absent from the test split: " f"{metrics['absent_classes']}")

    # English: Image-level evaluation needs both stages plus the segmenter.
    if stage2_model is not None:
        image_metrics, rows, misclassified_rows = evaluate_image_level(
            stage1_model,
            stage2_model,
            test_paths,
            stage1_prob_threshold,
            min_defective_patches,
        )
        report["stages"]["image_level"] = image_metrics
        report["image_level_predictions"] = rows
        report["misclassified_images"] = misclassified_rows
        save_confusion_matrix(
            np.array(image_metrics["binary"]["confusion_matrix"]),
            BINARY_CLASSES,
            "binary_image_level",
        )

        print("\n=== Image level, binary ===")
        print(
            f"balanced_accuracy={image_metrics['binary']['balanced_accuracy']:.4f}  "
            f"defect_recall={image_metrics['binary']['defect_recall']:.4f}  "
            f"normal_fpr={image_metrics['binary']['normal_false_positive_rate']:.4f}  "
            f"accuracy={image_metrics['binary']['accuracy']:.4f}  "
            f"macro_f1={image_metrics['binary']['macro_f1_all_classes']:.4f}"
        )
        print("Confusion Matrix [TN, FP / FN, TP]:")
        print(np.array(image_metrics["binary"]["confusion_matrix"]))

        if image_metrics["defect_type"] is not None:
            print("\n=== Image level, defect type (defective images only) ===")
            print(
                f"accuracy={image_metrics['defect_type']['accuracy']:.4f}  "
                f"macro_f1={image_metrics['defect_type']['macro_f1_all_classes']:.4f}  "
                f"scored={image_metrics['defect_type']['samples']}"
            )
            print(
                "Absent from image-level scoring: "
                f"{image_metrics['defect_type']['absent_classes']}"
            )

    unet_path = SEGMENTER_MODEL_DIR / "unet_best.pt"
    if not args.skip_segmentation and unet_path.exists() and stage2_model is not None:
        segmenter, _ = load_segmenter(best=True)
        segmentation_dataset = SegmentationPatchDataset(test_paths, train=False)
        patch_metrics = evaluate_segmentation_patches(segmenter, segmentation_dataset)
        report["stages"]["segmentation_patch_level"] = patch_metrics

        print("\n=== Segmentation, patch level ===")
        for key, value in patch_metrics.items():
            print(f"  {key:<10} {value:.4f}")

        localization = evaluate_localization(
            segmenter,
            [p for p in test_paths if class_of_image(p) != NORMAL_CLASS],
            stage1_prob_threshold,
            min_defective_patches,
            stage1_model,
            stage2_model,
        )
        report["stages"]["segmentation_image_level"] = localization["summary"]
        report["localization_per_image"] = localization["per_image"]

        print("\n=== Localization, image level ===")
        print(json.dumps(localization["summary"], indent=2))

    report["dataset_limitations"] = {
        "single_image_classes": sorted(SINGLE_IMAGE_CLASS_NAMES),
        "note": (
            "Knots, Contamination and Weft Crack have exactly one original "
            "image each. They stay entirely in the train split, so they have "
            "no validation or test samples and their per-class scores are "
            "undefined rather than zero."
        ),
    }

    out_path = METRICS_DIR / f"test_{args.model}.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved report to {out_path}")
    print(
        "Reminder: these numbers come from this evaluation run only. "
        "Hindi: Ye numbers sirf is run se hain, koi assume nahi kiya gaya."
    )


if __name__ == "__main__":
    main()
