"""English: Stratified 5-Fold Cross-Validation and Leave-One-Roll-Out (LORO) Evaluation.
Hindi: Stratified 5-Fold Cross-Validation aur Leave-One-Roll-Out (LORO) Evaluation.

STRICT CONSTRAINT: Only uses train + val data. Test data is NEVER touched!
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    METRICS_DIR,
    NUM_WORKERS,
    PATCH_SIZE,
    PATCH_STRIDE,
    RANDOM_SEED,
)
from dataset import PatchDataset, load_split
from inference import stage1_scores
from labels import NORMAL_CLASS, class_of_image, fabric_code_of_image
from models import build_model
from training import make_balanced_sampler, run_epoch, set_seed
from utils import read_image


def tune_fold_thresholds(
    model: torch.nn.Module,
    val_paths: List[Path],
    device: torch.device,
    prob_min: float = 0.30,
    prob_max: float = 0.90,
    prob_step: float = 0.05,
    min_patches_max: int = 4,
) -> Dict:
    """English: Grid-search Stage 1 threshold on the fold validation set.
    Hindi: Fold validation set par Stage 1 threshold ka grid-search.
    """
    records = []
    for p in val_paths:
        cls_name = class_of_image(p)
        if cls_name is None:
            continue
        is_def = 0 if cls_name == NORMAL_CLASS else 1
        img = read_image(p)
        scores = stage1_scores(model, img, device=device)
        probs = [prob for _, _, prob in scores]
        records.append({"is_defective": is_def, "probs": probs})

    y_true = np.array([r["is_defective"] for r in records])
    prob_thresholds = np.round(np.arange(prob_min, prob_max + 1e-5, prob_step), 2).tolist()
    min_patches_range = list(range(1, min_patches_max + 1))

    best_bal_acc = -1.0
    best_candidate = None

    for prob_th in prob_thresholds:
        for min_p in min_patches_range:
            y_pred = np.array([
                1 if sum(1 for prob in r["probs"] if prob >= prob_th) >= min_p else 0
                for r in records
            ])
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_candidate = {
                    "prob_threshold": float(prob_th),
                    "min_patches": int(min_p),
                    "balanced_accuracy": round(bal_acc, 4),
                    "defect_recall": round(recall, 4),
                    "normal_fpr": round(fpr, 4),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                }

    return best_candidate or {
        "prob_threshold": 0.5,
        "min_patches": 1,
        "balanced_accuracy": 0.5,
        "defect_recall": 0.0,
        "normal_fpr": 0.0,
    }


def train_quick_stage1(
    train_paths: List[Path],
    val_paths: List[Path],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> Tuple[torch.nn.Module, Dict]:
    """English: Train a fast Stage 1 EfficientNet model for CV evaluation.
    Hindi: CV evaluation ke liye fast Stage 1 EfficientNet train karta hai.
    """
    train_ds = PatchDataset(train_paths, stage="binary", train=True)
    val_ds = PatchDataset(val_paths, stage="binary", train=False)

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model("efficientnet", num_classes=2, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True, model_name="efficientnet", phase="head")

    # Image-level evaluation on fold validation set
    fold_metrics = tune_fold_thresholds(model, val_paths, device)
    return model, fold_metrics


def run_5fold_cv(train_val_paths: List[Path], device: torch.device, smoke: bool = False) -> Dict:
    """English: Stratified 5-Fold Cross Validation on Train + Val images.
    Hindi: Train + Val images par 5-Fold Stratified Cross Validation.
    """
    n_splits = 2 if smoke else 5
    epochs = 2 if smoke else 6
    print(f"\n--- Running Stratified {n_splits}-Fold Cross-Validation on Train + Val (Test Untouched) ---")
    labels = [1 if class_of_image(p) != NORMAL_CLASS else 0 for p in train_val_paths]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_paths, labels), 1):
        fold_train = [train_val_paths[i] for i in train_idx]
        fold_val = [train_val_paths[i] for i in val_idx]

        print(f"Fold {fold}/{n_splits}: train_images={len(fold_train)}, val_images={len(fold_val)}...")
        _, metrics = train_quick_stage1(fold_train, fold_val, device=device, epochs=epochs, batch_size=4 if smoke else 16)
        metrics["fold"] = fold
        metrics["val_images_count"] = len(fold_val)
        fold_results.append(metrics)
        print(
            f"  Fold {fold} Result: Balanced Acc = {metrics['balanced_accuracy']:.4f}, "
            f"Recall = {metrics['defect_recall']:.4f}, FPR = {metrics['normal_fpr']:.4f} "
            f"(Thresh={metrics['prob_threshold']}, MinP={metrics['min_patches']})"
        )

    bal_accs = [f["balanced_accuracy"] for f in fold_results]
    recalls = [f["defect_recall"] for f in fold_results]
    fprs = [f["normal_fpr"] for f in fold_results]

    summary = {
        "mean_balanced_accuracy": round(float(np.mean(bal_accs)), 4),
        "std_balanced_accuracy": round(float(np.std(bal_accs)), 4),
        "mean_defect_recall": round(float(np.mean(recalls)), 4),
        "std_defect_recall": round(float(np.std(recalls)), 4),
        "mean_normal_fpr": round(float(np.mean(fprs)), 4),
        "std_normal_fpr": round(float(np.std(fprs)), 4),
        "folds": fold_results,
    }
    return summary


def run_leave_one_roll_out(train_val_paths: List[Path], device: torch.device, smoke: bool = False) -> Dict:
    """English: Leave-One-Roll-Out (LORO) Validation across fabric rolls in Train + Val.
    Hindi: Train + Val ke fabric rolls par Leave-One-Roll-Out validation.
    """
    print("\n--- Running Leave-One-Roll-Out (LORO) Validation on Train + Val ---")
    rolls = sorted(set(fabric_code_of_image(p) or "00" for p in train_val_paths))
    roll_results = []
    epochs = 2 if smoke else 6

    for held_out_roll in rolls:
        fold_train = [p for p in train_val_paths if (fabric_code_of_image(p) or "00") != held_out_roll]
        fold_val = [p for p in train_val_paths if (fabric_code_of_image(p) or "00") == held_out_roll]

        # Check if held-out roll has both normal and defective or at least samples
        min_train = 4 if smoke else 10
        if not fold_val or len(fold_train) < min_train:
            continue

        print(f"Held-out Roll '{held_out_roll}': train={len(fold_train)} imgs, val={len(fold_val)} imgs...")
        _, metrics = train_quick_stage1(fold_train, fold_val, device=device, epochs=epochs, batch_size=4 if smoke else 16)
        metrics["held_out_roll"] = held_out_roll
        metrics["val_images_count"] = len(fold_val)
        roll_results.append(metrics)
        print(
            f"  Roll {held_out_roll}: Balanced Acc = {metrics['balanced_accuracy']:.4f}, "
            f"Recall = {metrics['defect_recall']:.4f}, FPR = {metrics['normal_fpr']:.4f}"
        )
        if smoke and len(roll_results) >= 2:
            break

    bal_accs = [r["balanced_accuracy"] for r in roll_results]
    recalls = [r["defect_recall"] for r in roll_results]
    fprs = [r["normal_fpr"] for r in roll_results]

    summary = {
        "mean_balanced_accuracy": round(float(np.mean(bal_accs)), 4) if bal_accs else 0.0,
        "std_balanced_accuracy": round(float(np.std(bal_accs)), 4) if bal_accs else 0.0,
        "mean_defect_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "std_defect_recall": round(float(np.std(recalls)), 4) if recalls else 0.0,
        "mean_normal_fpr": round(float(np.mean(fprs)), 4) if fprs else 0.0,
        "std_normal_fpr": round(float(np.std(fprs)), 4) if fprs else 0.0,
        "rolls_evaluated": roll_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run cross-validation and LORO evaluation.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (16 images, 2 folds, 2 epochs) saved to _smoketest_archive.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    splits = load_split()
    train_val_paths = splits["train"] + splits["val"]

    out_file = METRICS_DIR / "cross_validation.json"
    if args.smoke:
        print("\n" + "=" * 60)
        print(">>> [SMOKE TEST MODE] Running CV with 16 images, 2 folds, 2 epochs.")
        print(">>> Output saved to results/_smoketest_archive/cross_validation.json")
        print("=" * 60 + "\n")
        train_val_paths = train_val_paths[:16]
        archive_dir = METRICS_DIR.parent / "_smoketest_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        out_file = archive_dir / "cross_validation.json"

    print(f"Total Train + Val Images for CV: {len(train_val_paths)} (Test split {len(splits['test'])} untouched)")

    cv_summary = run_5fold_cv(train_val_paths, device=DEVICE, smoke=args.smoke)
    loro_summary = run_leave_one_roll_out(train_val_paths, device=DEVICE, smoke=args.smoke)

    output = {
        "evaluation_scope": "Train + Val splits only (Test set was NOT touched)",
        "train_val_images_count": len(train_val_paths),
        "stratified_5fold_cv": cv_summary,
        "leave_one_roll_out_cv": loro_summary,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved Cross-Validation metrics to: {out_file}")
    print(
        f"CV Balanced Accuracy:   {cv_summary['mean_balanced_accuracy']:.4f} +/- {cv_summary['std_balanced_accuracy']:.4f}"
    )
    print(
        f"LORO Balanced Accuracy: {loro_summary['mean_balanced_accuracy']:.4f} +/- {loro_summary['std_balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
