"""English: Grid-search image-level inference thresholds on the validation set.
Hindi: Validation set par image-level inference thresholds ka grid-search.

Validation images are used to find optimal (STAGE1_DEFECT_PROB_THRESHOLD,
STAGE1_MIN_DEFECTIVE_PATCHES) pair that maximizes image-level balanced accuracy.
Test set is NEVER used for tuning to prevent data leakage.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from config import (
    DEVICE,
    METRICS_DIR,
    STAGE1_DEFECT_PROB_THRESHOLD,
    STAGE1_MIN_DEFECTIVE_PATCHES,
)
from dataset import load_split
from inference import load_classifier, load_segmenter, segment_image, stage1_scores
from labels import NORMAL_CLASS, class_of_image
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


def tune_thresholds(
    model_name: str,
    device: torch.device = DEVICE,
    prob_min: float = 0.30,
    prob_max: float = 0.95,
    prob_step: float = 0.05,
    min_patches_max: int = 4,
) -> Dict:
    """English: Grid-search image-level inference thresholds on the validation set.
    Hindi: Validation set par image-level inference thresholds ka grid-search.
    """
    splits = load_split()
    val_paths = splits["val"]

    model, _ = load_classifier("binary", model_name, best=True, device=device)

    records = []
    print(f"Extracting validation patch scores for {model_name} ({len(val_paths)} images)...")
    for path in val_paths:
        cls_name = class_of_image(path)
        if cls_name is None:
            continue
        is_def = 0 if cls_name == NORMAL_CLASS else 1
        img = read_image(path)
        scores = stage1_scores(model, img, device=device)
        probs = [prob for _, _, prob in scores]
        records.append({
            "path": path,
            "is_defective": is_def,
            "probs": probs,
        })

    y_true = np.array([r["is_defective"] for r in records])
    prob_thresholds = np.round(np.arange(prob_min, prob_max + 1e-5, prob_step), 2).tolist()
    min_patches_range = list(range(1, min_patches_max + 1))

    candidates = []
    for prob_th in prob_thresholds:
        for min_p in min_patches_range:
            y_pred = np.array([
                1 if sum(1 for prob in r["probs"] if prob >= prob_th) >= min_p else 0
                for r in records
            ])
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

            candidates.append({
                "stage1_prob_threshold": float(prob_th),
                "min_defective_patches": int(min_p),
                "balanced_accuracy": round(bal_acc, 4),
                "defect_recall": round(recall, 4),
                "normal_fpr": round(fpr, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "confusion_matrix": [[tn, fp], [fn, tp]],
            })

    candidates.sort(
        key=lambda c: (c["balanced_accuracy"], c["defect_recall"], -c["normal_fpr"]),
        reverse=True,
    )
    best = candidates[0]

    result = {
        "model": model_name,
        "best_stage1_prob_threshold": best["stage1_prob_threshold"],
        "best_min_defective_patches": best["min_defective_patches"],
        "val_balanced_accuracy": best["balanced_accuracy"],
        "val_defect_recall": best["defect_recall"],
        "val_normal_false_positive_rate": best["normal_fpr"],
        "val_confusion_matrix": best["confusion_matrix"],
        "val_images_count": len(records),
        "default_stage1_prob_threshold": STAGE1_DEFECT_PROB_THRESHOLD,
        "default_min_defective_patches": STAGE1_MIN_DEFECTIVE_PATCHES,
        "top_candidates": candidates[:10],
    }

    out_file = METRICS_DIR / f"stage1_thresholds_{model_name}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Validation Tuning Complete for {model_name} ===")
    print(f"Optimal Threshold (tau):     {best['stage1_prob_threshold']}")
    print(f"Min Patches (k):             {best['min_defective_patches']}")
    print(f"Val Balanced Accuracy:       {best['balanced_accuracy']:.4f}")
    print(f"Val Defect Recall:           {best['defect_recall']:.4f}")
    print(f"Val Normal FPR:              {best['normal_fpr']:.4f}")
    print(f"Val Confusion Matrix [TN, FP / FN, TP]:")
    print(np.array(best["confusion_matrix"]))
    print(f"Saved tuned thresholds to: {out_file}")

    return result


def tune_segmentation_thresholds(
    device: torch.device = DEVICE,
    threshold_min: float = 0.10,
    threshold_max: float = 0.90,
    threshold_step: float = 0.10,
    min_component_area: int = 16,
) -> Dict:
    """English: Grid-search U-Net mask threshold and component filtering on validation strips.
    Hindi: Validation strips par U-Net mask threshold aur component filtering ka grid-search.
    """
    from config import MASK_DIR

    splits = load_split()
    val_paths = [p for p in splits["val"] if class_of_image(p) != NORMAL_CLASS]

    segmenter = load_segmenter(best=True, device=device)

    val_data = []
    print(f"Extracting validation segmentation strips ({len(val_paths)} defective images)...")
    for path in val_paths:
        gt_mask = merged_mask(path, MASK_DIR)
        if gt_mask is None or not gt_mask.any():
            continue
        image = read_image(path)
        gt_box = mask_bounding_box(gt_mask)
        val_data.append({
            "path": path,
            "image": image,
            "gt_mask": gt_mask,
            "gt_box": gt_box,
        })

    if not val_data:
        print("No validation images with masks found for segmentation tuning.")
        return {}

    thresholds = np.round(
        np.arange(threshold_min, threshold_max + 1e-5, threshold_step), 2
    ).tolist()

    candidates = []
    for th in thresholds:
        dices = []
        ious = []
        precisions = []
        recalls = []
        box_touches = []

        for item in val_data:
            pred_mask, _ = segment_image(
                segmenter,
                item["image"],
                threshold=float(th),
                min_component_area=min_component_area,
                device=device,
            )
            gt_m = (item["gt_mask"] > 0).astype(np.uint8)
            pred_m = (pred_mask > 0).astype(np.uint8)

            intersection = int((pred_m * gt_m).sum())
            pred_sum = int(pred_m.sum())
            gt_sum = int(gt_m.sum())
            union = pred_sum + gt_sum - intersection

            dice = (2.0 * intersection + 1.0) / (pred_sum + gt_sum + 1.0)
            iou = (intersection + 1.0) / (union + 1.0)
            prec = (intersection + 1.0) / (pred_sum + 1.0)
            rec = (intersection + 1.0) / (gt_sum + 1.0)

            pred_box = mask_bounding_box(pred_m)
            touch = boxes_intersect(pred_box, item["gt_box"])

            dices.append(dice)
            ious.append(iou)
            precisions.append(prec)
            recalls.append(rec)
            box_touches.append(1.0 if touch else 0.0)

        candidates.append({
            "threshold": float(th),
            "min_component_area": min_component_area,
            "val_strip_dice": round(float(np.mean(dices)), 4),
            "val_strip_iou": round(float(np.mean(ious)), 4),
            "val_pixel_precision": round(float(np.mean(precisions)), 4),
            "val_pixel_recall": round(float(np.mean(recalls)), 4),
            "val_box_touch_rate": round(float(np.mean(box_touches)), 4),
        })

    candidates.sort(
        key=lambda c: (c["val_strip_dice"], c["val_box_touch_rate"], c["val_strip_iou"]),
        reverse=True,
    )
    best = candidates[0]

    result = {
        "best_threshold": best["threshold"],
        "best_min_component_area": best["min_component_area"],
        "val_strip_dice": best["val_strip_dice"],
        "val_strip_iou": best["val_strip_iou"],
        "val_pixel_precision": best["val_pixel_precision"],
        "val_pixel_recall": best["val_pixel_recall"],
        "val_box_touch_rate": best["val_box_touch_rate"],
        "val_images_evaluated": len(val_data),
        "candidates": candidates,
    }

    out_file = METRICS_DIR / "segmentation_thresholds.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Validation Segmentation Tuning Complete ===")
    print(f"Optimal Mask Threshold:     {best['threshold']}")
    print(f"Min Component Area:         {best['min_component_area']} px")
    print(f"Val Strip-Level Dice:       {best['val_strip_dice']:.4f}")
    print(f"Val Strip-Level IoU:        {best['val_strip_iou']:.4f}")
    print(f"Val Pixel Precision:        {best['val_pixel_precision']:.4f}")
    print(f"Val Pixel Recall:           {best['val_pixel_recall']:.4f}")
    print(f"Val Box Touch Rate:         {best['val_box_touch_rate']:.4f}")
    print(f"Saved tuned segmentation thresholds to: {out_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Tune Stage-1 image-level and segmentation thresholds on the validation set"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "efficientnet", "segmenter", "all"],
        default="all",
        help="Model to tune thresholds for",
    )
    parser.add_argument("--prob-min", type=float, default=0.30)
    parser.add_argument("--prob-max", type=float, default=0.95)
    parser.add_argument("--prob-step", type=float, default=0.05)
    parser.add_argument("--min-patches-max", type=int, default=4)
    args = parser.parse_args()

    if args.model in ("cnn", "all"):
        tune_thresholds(
            model_name="cnn",
            device=DEVICE,
            prob_min=args.prob_min,
            prob_max=args.prob_max,
            prob_step=args.prob_step,
            min_patches_max=args.min_patches_max,
        )

    if args.model in ("efficientnet", "all"):
        tune_thresholds(
            model_name="efficientnet",
            device=DEVICE,
            prob_min=args.prob_min,
            prob_max=args.prob_max,
            prob_step=args.prob_step,
            min_patches_max=args.min_patches_max,
        )

    if args.model in ("segmenter", "all"):
        tune_segmentation_thresholds(device=DEVICE)


if __name__ == "__main__":
    main()
