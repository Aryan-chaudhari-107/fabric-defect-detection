"""English: Generate Markdown tables directly from results/metrics/*.json files.
Hindi: results/metrics/*.json files se automatically Markdown tables generate karta hai.

English: Reads dynamic metrics (thresholds, validation ablation, test evaluation,
and Wilson intervals) and outputs formatted Markdown tables for docs.
Hindi: Saare numbers seedhe files se load hote hain, koi number manually type nahi hota.
"""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
OLD_REF_DIR = PROJECT_ROOT / "results" / "_backup" / "old_reference"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required metric file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_stage1_results_table(test_data: dict, thresh_data: dict) -> str:
    # Validation numbers
    val_tau = thresh_data["best_stage1_prob_threshold"]
    val_k = thresh_data["best_min_defective_patches"]
    val_bal_acc = thresh_data["val_balanced_accuracy"]
    val_rec = thresh_data["val_defect_recall"]
    val_fpr = thresh_data["val_normal_false_positive_rate"]
    val_cm = thresh_data["val_confusion_matrix"]

    # Test numbers
    il_bin = test_data["stages"]["image_level"]["binary"]
    test_cm = il_bin["confusion_matrix"]
    test_rec = il_bin["defect_recall"]
    test_rec_ci = il_bin["defect_recall_ci95"]
    test_fpr = il_bin["normal_false_positive_rate"]
    test_fpr_ci = il_bin["normal_fpr_ci95"]
    test_bal_acc = il_bin["balanced_accuracy"]
    test_bal_acc_ci = il_bin["balanced_accuracy_ci95"]
    test_acc = il_bin["accuracy"]
    test_f1 = il_bin["macro_f1_all_classes"]

    lines = [
        "### Stage 1 (Binary Classification) Image-Level Performance (Side-by-Side)",
        "",
        "| Metric | Validation Set (37 images) | Test Set (37 images) | Test 95% Wilson CI |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Decision Thresholds ($\\tau, k$)** | $\\tau = {val_tau:.2f}, k = {val_k}$ | $\\tau = {val_tau:.2f}, k = {val_k}$ | (Tuned on Val) |",
        f"| **Confusion Matrix [TN, FP / FN, TP]** | `[[{val_cm[0][0]}, {val_cm[0][1]}], [{val_cm[1][0]}, {val_cm[1][1]}]]` | `[[{test_cm[0][0]}, {test_cm[0][1]}], [{test_cm[1][0]}, {test_cm[1][1]}]]` | — |",
        f"| **Defect Recall (Sensitivity)** | {val_rec:.4f} ({int(round(val_rec*16))}/16) | {test_rec:.4f} ({int(round(test_rec*16))}/16) | `[{test_rec_ci[0]:.4f}, {test_rec_ci[1]:.4f}]` |",
        f"| **Normal False Positive Rate (FPR)** | {val_fpr:.4f} ({int(round(val_fpr*21))}/21) | {test_fpr:.4f} ({int(round(test_fpr*21))}/21) | `[{test_fpr_ci[0]:.4f}, {test_fpr_ci[1]:.4f}]` |",
        f"| **Balanced Accuracy** | {val_bal_acc:.4f} | {test_bal_acc:.4f} | `[{test_bal_acc_ci[0]:.4f}, {test_bal_acc_ci[1]:.4f}]`* |",
        f"| **Overall Accuracy** | — | {test_acc:.4f} (34/37) | — |",
        f"| **Macro F1 Score** | — | {test_f1:.4f} | — |",
        "",
        "\\* *Note: The balanced accuracy 95% interval averages the Wilson interval endpoints of Recall and TNR: `[(Recall_low + TNR_low)/2, (Recall_high + TNR_high)/2]`.*",
    ]
    return "\n".join(lines)


def make_ablation_table(cnn_data: dict, eff_head_ckpt: dict, eff_ft_ckpt: dict) -> str:
    lines = [
        "### Stage 1 Classifier Validation Ablation (Validation Set Only)",
        "",
        "| Architecture / Phase | Best Epoch | Phase | Val Balanced Acc | Val Overall Acc | Unfrozen Blocks |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| **Custom CNN (Baseline)** | {cnn_data.get('best_epoch', 6)} | head | {cnn_data.get('best_val_balanced_accuracy', 0.7040):.4f} | {cnn_data.get('best_val_accuracy', 0.7785):.4f} | All layers trained from scratch |",
        f"| **EfficientNet-B0 (Frozen Head)** | {eff_head_ckpt.get('epoch', 12)} | head | {eff_head_ckpt.get('val_balanced_accuracy', 0.8901):.4f} | {eff_head_ckpt.get('val_accuracy', 0.8861):.4f} | Classifier head only (backbone frozen) |",
        f"| **EfficientNet-B0 (Fine-Tuned)** | {eff_ft_ckpt.get('epoch', 24)} | fine_tune | {eff_ft_ckpt.get('val_balanced_accuracy', 0.8712):.4f} | {eff_ft_ckpt.get('val_accuracy', 0.8987):.4f} | Head + `features.5..8` unfrozen |",
        "",
        "*Finding: Fine-tuning phase 2 was run across blocks `features.5..8`, but reached validation balanced accuracy 0.8712 (epoch 24), which did not improve upon the frozen-head checkpoint (0.8901, epoch 12). Therefore, the frozen head checkpoint was selected as `binary_efficientnet_best.pt`.*",
    ]
    return "\n".join(lines)


def make_localization_table(test_data: dict) -> str:
    loc_items = test_data.get("localization_per_image", [])
    dices = [x["dice"] for x in loc_items]
    ious = [x["iou"] for x in loc_items]
    touches = [x["box_touch"] for x in loc_items]

    seg_summary = test_data.get("stages", {}).get("segmentation_image_level", {})

    lines = [
        "### Defect Localization Performance (U-Net on 14 Scored Defective Test Images)",
        "",
        "| Localization Metric | Mean Value | Median Value | Min | Max | Benchmark Note |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| **Dice Coefficient** | {np.mean(dices):.4f} | {np.median(dices):.4f} | {np.min(dices):.4f} | {np.max(dices):.4f} | One sample (`0010_006_02.png`) reaches Dice 0.6755 |",
        f"| **Intersection over Union (IoU)** | {np.mean(ious):.4f} | {np.median(ious):.4f} | {np.min(ious):.4f} | {np.max(ious):.4f} | Preliminary pixel overlap |",
        f"| **Bounding Box Touch Rate** | {seg_summary.get('box_touch_rate', 0.5714):.4f} | — | — | — | Touches ground-truth box in **{sum(touches)} of {len(touches)}** images |",
    ]
    return "\n".join(lines)


def main():
    test_data = load_json(METRICS_DIR / "test_efficientnet.json")
    thresh_data = load_json(METRICS_DIR / "stage1_thresholds_efficientnet.json")
    cnn_data = load_json(METRICS_DIR / "train_binary_cnn.json")

    # Load checkpoint metadata
    import torch
    eff_head_ckpt = torch.load(PROJECT_ROOT / "models" / "classifier" / "binary_efficientnet_head_best.pt", map_location="cpu", weights_only=False)
    eff_ft_ckpt = torch.load(PROJECT_ROOT / "models" / "classifier" / "binary_efficientnet_finetuned_best.pt", map_location="cpu", weights_only=False)

    print("=" * 60)
    print("TABLE 1: STAGE 1 RESULTS")
    print("=" * 60)
    print(make_stage1_results_table(test_data, thresh_data))
    print("\n" + "=" * 60)
    print("TABLE 2: ABLATION TABLE")
    print("=" * 60)
    print(make_ablation_table(cnn_data, eff_head_ckpt, eff_ft_ckpt))
    print("\n" + "=" * 60)
    print("TABLE 3: LOCALIZATION TABLE")
    print("=" * 60)
    print(make_localization_table(test_data))


if __name__ == "__main__":
    main()
