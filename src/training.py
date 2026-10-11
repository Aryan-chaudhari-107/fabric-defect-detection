"""English: Training loop, sampler construction and checkpointing for classifiers.
Hindi: Classifiers ke liye training loop, sampler aur checkpoint logic.
"""

import datetime
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    BACKUP_MODEL_DIR,
    BINARY_EPOCHS,
    BINARY_LEARNING_RATE,
    CLASSIFIER_MODEL_DIR,
    DEFECT_TYPE_EPOCHS,
    DEFECT_TYPE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_UNFREEZE_FROM_BLOCK,
    LOGS_DIR,
    METRICS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAINING_CURVES_DIR,
    WEIGHT_DECAY,
    checkpoint_path,
)


def safe_save_checkpoint(
    checkpoint_data: dict,
    target_path: Path,
    overwrite: bool = False,
    metric_key: str = "val_balanced_accuracy",
) -> bool:
    """English: Safely save checkpoint with overwrite protection and backup.
    Hindi: Checkpoint ko overwrite protection aur backup ke saath safely save karta hai.

    (a) If target_path is in _smoketest_archive, simply save without backup or refusal.
    (b) Before saving *_best.pt, if it exists, check existing selection metric.
        Refuse to overwrite unless new metric is strictly higher or overwrite=True.
    (c) Before replacing an existing checkpoint, copy the old one to
        models/_backup/<name>_<timestamp>.pt.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if "_smoketest_archive" in str(target_path):
        torch.save(checkpoint_data, target_path)
        return True

    if target_path.exists():
        is_best = target_path.name.endswith("_best.pt")
        if is_best and not overwrite:
            try:
                old_ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
                old_metric = old_ckpt.get(metric_key, old_ckpt.get("val_balanced_accuracy", old_ckpt.get("val_dice", None)))
            except Exception:
                old_metric = None

            new_metric = checkpoint_data.get(metric_key, checkpoint_data.get("val_balanced_accuracy", checkpoint_data.get("val_dice", None)))
            if old_metric is not None and new_metric is not None:
                if new_metric <= old_metric:
                    print(
                        f"[PROTECTION] Refusing to overwrite {target_path.name}: "
                        f"existing metric ({old_metric:.4f}) >= new metric ({new_metric:.4f}). "
                        f"Pass --overwrite to force."
                    )
                    return False

        backup_dir = BACKUP_MODEL_DIR
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{target_path.stem}_{timestamp}{target_path.suffix}"
        try:
            shutil.copy2(target_path, backup_path)
            print(f"[BACKUP] Backed up existing checkpoint {target_path.name} to {backup_path}")
        except Exception as e:
            print(f"[BACKUP WARNING] Could not backup {target_path.name}: {e}")

    torch.save(checkpoint_data, target_path)
    return True
from dataset import PatchDataset
from labels import (
    BINARY_CLASSES,
    DEFECT_TYPE_CLASSES,
    SINGLE_IMAGE_CLASS_NAMES,
)
from models import build_model, set_backbone_trainable, unfreeze_last_blocks

# English: Which label space each stage uses.
# Hindi: Har stage kaunsa label space use karta hai.
STAGE_SPEC = {
    "binary": {
        "class_names": BINARY_CLASSES,
        "epochs": BINARY_EPOCHS,
        "learning_rate": BINARY_LEARNING_RATE,
    },
    "defect_type": {
        "class_names": DEFECT_TYPE_CLASSES,
        "epochs": DEFECT_TYPE_EPOCHS,
        "learning_rate": DEFECT_TYPE_LEARNING_RATE,
    },
}


def set_seed(seed: int = RANDOM_SEED) -> None:
    """English: Seed every random source used during training.
    Hindi: Training mein use hone wale sabhi random sources ko seed karta hai.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_balanced_sampler(dataset: PatchDataset) -> WeightedRandomSampler:
    """English: Build a sampler that equalises class frequency.
    Hindi: Ek aisa sampler banata hai jo class frequency ko barabar karta hai.

    English: For the binary stage, balancing uses ``binary_label`` (Normal vs Defective)
    so that individual defect types do not collectively outweigh Normal samples.
    For ``defect_type`` stage, balancing uses ``class_name`` across defect types.
    A ``WeightedRandomSampler`` is used rather than a weighted loss.
    Hindi: Binary stage mein ``binary_label`` se Normal aur Defective balance karte hain.
    Defect_type stage mein ``class_name`` se har defect type balance karte hain.
    Yahan ``WeightedRandomSampler`` use hota hai, weighted loss nahi.
    """
    if getattr(dataset, "stage", None) == "binary":
        key_fn = lambda record: record["binary_label"]
    else:
        key_fn = lambda record: record["class_name"]

    counts = Counter(key_fn(record) for record in dataset.records)
    weights = torch.tensor(
        [1.0 / max(counts[key_fn(record)], 1) for record in dataset.records],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    if getattr(dataset, "stage", None) == "binary" and len(weights) > 0:
        # English: Sanity check: draw one epoch of indices, compute Defective fraction
        # Hindi: Sanity check: ek epoch ke indices draw karke Defective fraction calculate karo
        sample_indices = torch.multinomial(
            weights,
            len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        ).tolist()
        defective_count = sum(
            1 for idx in sample_indices if dataset.records[idx]["binary_label"] == 1
        )
        defective_fraction = defective_count / max(len(sample_indices), 1)
        print(
            f"Binary sampler sanity check: Defective patch fraction in 1 epoch sample = "
            f"{defective_fraction * 100:.1f}% ({defective_count}/{len(sample_indices)})"
        )
        if defective_fraction < 0.40 or defective_fraction > 0.60:
            logging.warning(
                f"Binary sampler sanity check warning: Defective fraction {defective_fraction:.3f} "
                "is outside expected 40-60% range."
            )
            print(
                f"WARNING: Defective patch fraction {defective_fraction * 100:.1f}% is outside "
                "the expected 40-60% range."
            )

    return sampler


def configure_batchnorm_modes(
    model: nn.Module,
    model_name: str,
    phase: str,
    from_block: int = FINETUNE_UNFREEZE_FROM_BLOCK,
) -> None:
    """English: Configure BatchNorm eval/train modes for frozen/unfrozen blocks.
    Hindi: Frozen blocks ke BatchNorm ko eval mode mein aur unfrozen ko train mode mein rakhta hai.
    """
    if model_name != "efficientnet" or not hasattr(model, "features"):
        return

    if phase == "head":
        # All backbone BatchNorm layers in eval mode
        for module in model.features.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)):
                module.eval()
    elif phase == "fine_tune":
        # Frozen blocks (< from_block) in eval mode; unfrozen blocks (>= from_block) stay in train mode
        blocks = list(model.features.children())
        for pos, block in enumerate(blocks):
            if pos < from_block:
                for module in block.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)):
                        module.eval()
            else:
                for module in block.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)):
                        module.train()


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    stage: str,
    model_name: str,
    finetune: bool,
):
    """English: Create the optimizer for Phase 1.
    Hindi: Phase 1 ke liye optimizer banata hai.

    English: Phase 1 trains only the classifier head of EfficientNet-B0 at the
    base learning rate with the backbone frozen. Phase 2 unfreezes the deepest
    feature blocks (>= FINETUNE_UNFREEZE_FROM_BLOCK) and adds them to the optimizer
    at FINETUNE_LEARNING_RATE during the phase transition.
    Hindi: Phase 1 mein sirf EfficientNet-B0 ka classifier head base learning rate
    par train hota hai aur backbone frozen rehti hai. Phase 2 transition par
    gehre blocks unfreeze karke optimizer mein add hote hain.
    """
    if model_name != "efficientnet" or not finetune:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        return optimizer, None

    # Phase 1: Freeze backbone completely, train only classifier head
    set_backbone_trainable(model, trainable=False)
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("classifier") and parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        head_parameters,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer, None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
    model_name: str = "cnn",
    phase: str = "head",
) -> Dict:
    """English: Run one epoch and return loss, top-1 accuracy, predictions and labels.
    Hindi: Ek epoch chalata hai aur loss, top-1 accuracy, predictions aur labels deta hai.
    """
    if train:
        model.train()
        configure_batchnorm_modes(model, model_name, phase, FINETUNE_UNFREEZE_FROM_BLOCK)
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total_items = 0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_items += images.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if train:
                loss.backward()
                optimizer.step()

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": correct / max(total_items, 1),
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def save_curves(
    history: Dict[str, List[float]],
    name: str,
    curves_dir: Optional[Path] = None,
) -> None:
    """English: Plot loss, accuracy and balanced accuracy curves for the report.
    Hindi: Report ke liye loss, accuracy aur balanced accuracy curves plot karta hai.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = curves_dir if curves_dir is not None else TRAINING_CURVES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    has_bal_acc = "train_bal_acc" in history and "val_bal_acc" in history
    num_cols = 3 if has_bal_acc else 2
    figure, axes = plt.subplots(1, num_cols, figsize=(5.5 * num_cols, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if has_bal_acc:
        axes[2].plot(history["train_bal_acc"], label="train")
        axes[2].plot(history["val_bal_acc"], label="validation")
        axes[2].set_title("Balanced Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    figure.tight_layout()
    figure.savefig(out_dir / f"{name}_curves.png", dpi=120)
    plt.close(figure)


def train(
    stage: str,
    model_name: str,
    train_dataset: PatchDataset,
    val_dataset: PatchDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    finetune: bool,
    finetune_epochs: int,
    patience: int,
    balance: bool,
    threshold: float,
    model_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    curves_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict:
    """English: Full training routine shared by both stages.
    Hindi: Dono stages ke liye shared poora training routine.

    English: Early stopping and checkpoint selection monitor balanced accuracy.
    The checkpoint written to ``*_best.pt`` saves the best validation balanced
    accuracy model and its metrics.
    Hindi: Early stopping aur checkpoint selection validation balanced accuracy
    ko monitor karte hain.
    """
    out_model_dir = model_dir if model_dir is not None else CLASSIFIER_MODEL_DIR
    out_metrics_dir = metrics_dir if metrics_dir is not None else METRICS_DIR
    out_curves_dir = curves_dir if curves_dir is not None else TRAINING_CURVES_DIR
    out_logs_dir = logs_dir if logs_dir is not None else (
        out_metrics_dir.parent / "_smoketest_archive" if "_smoketest_archive" in str(out_metrics_dir) else LOGS_DIR
    )

    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_curves_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = out_logs_dir / f"train_{stage}_{model_name}.log"
    with log_file_path.open("w", encoding="utf-8") as f:
        f.write(f"=== Training Run: stage={stage} model={model_name} device={device} ===\n")

    def log_print(msg: str):
        print(msg)
        with log_file_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    spec = STAGE_SPEC[stage]
    class_names = spec["class_names"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not balance,
        sampler=make_balanced_sampler(train_dataset) if balance else None,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(model_name, len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer, _ = build_optimizer(
        model, learning_rate, stage, model_name, finetune
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    finetune_info = None
    if finetune and model_name == "efficientnet":
        finetune_info = {
            "unfrozen_blocks": [],
            "unfreeze_from_block": FINETUNE_UNFREEZE_FROM_BLOCK,
            "finetune_lr": FINETUNE_LEARNING_RATE,
            "phase2_started": False,
        }

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_bal_acc": [],
        "val_bal_acc": [],
        "phase": [],
    }

    best_val_balanced_acc = -1.0
    best_val_acc = 0.0
    best_val_recall_normal: Optional[float] = None
    best_val_recall_defective: Optional[float] = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    phase1_best_state = None
    phase1_best_val_bal_acc = -1.0
    phase1_best_val_acc = 0.0
    phase1_best_recall_normal = None
    phase1_best_recall_defective = None
    phase1_best_epoch = 0

    phase2_best_state = None
    phase2_best_val_bal_acc = -1.0
    phase2_best_val_acc = 0.0
    phase2_best_recall_normal = None
    phase2_best_recall_defective = None
    phase2_best_epoch = 0

    # ------------------ PHASE 1: Head Only ------------------
    print(f"\n=== Starting Phase 1: Frozen Backbone, Training Classifier Head (max {epochs} epochs) ===")
    for epoch in range(1, epochs + 1):
        phase = "head"
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            model_name=model_name,
            phase=phase,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            model_name=model_name,
            phase=phase,
        )
        scheduler.step()

        train_bal_acc = float(
            balanced_accuracy_score(train_metrics["labels"], train_metrics["predictions"])
        )
        val_bal_acc = float(
            balanced_accuracy_score(val_metrics["labels"], val_metrics["predictions"])
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_bal_acc"].append(train_bal_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["phase"].append(phase)

        val_labels = val_metrics["labels"]
        val_preds = val_metrics["predictions"]
        if stage == "binary":
            norm_mask = (val_labels == 0)
            def_mask = (val_labels == 1)
            cur_recall_normal = (
                float((val_preds[norm_mask] == 0).sum() / norm_mask.sum())
                if norm_mask.sum() > 0
                else 0.0
            )
            cur_recall_defective = (
                float((val_preds[def_mask] == 1).sum() / def_mask.sum())
                if def_mask.sum() > 0
                else 0.0
            )
        else:
            cur_recall_normal = None
            cur_recall_defective = None

        print(
            f"[{phase:<9}] Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"train_bal_acc={train_bal_acc:.4f}  "
            f"val_bal_acc={val_bal_acc:.4f}"
        )

        if val_bal_acc > phase1_best_val_bal_acc:
            phase1_best_val_bal_acc = val_bal_acc
            phase1_best_val_acc = val_metrics["accuracy"]
            phase1_best_recall_normal = cur_recall_normal
            phase1_best_recall_defective = cur_recall_defective
            phase1_best_epoch = epoch
            phase1_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Phase 1 early stopping at epoch {epoch}: validation balanced accuracy did "
                    f"not improve for {patience} epochs."
                )
                break

    # Save Phase 1 best checkpoint
    if phase1_best_state is not None:
        head_ckpt_path = out_model_dir / f"{stage}_{model_name}_head_best.pt"
        head_data = {
            "model_state": phase1_best_state,
            "model_name": model_name,
            "stage": stage,
            "class_names": class_names,
            "num_classes": len(class_names),
            "epoch": phase1_best_epoch,
            "phase": "head",
            "selection_metric": "balanced_accuracy",
            "val_balanced_accuracy": phase1_best_val_bal_acc,
            "val_accuracy": phase1_best_val_acc,
            "val_recall_normal": phase1_best_recall_normal,
            "val_recall_defective": phase1_best_recall_defective,
            "threshold": threshold,
        }
        if safe_save_checkpoint(head_data, head_ckpt_path, overwrite=overwrite):
            log_print(f"Phase 1 best (val_bal_acc={phase1_best_val_bal_acc:.4f}, epoch={phase1_best_epoch}) saved to {head_ckpt_path}")

    # ------------------ PHASE 2: Fine-Tuning ------------------
    if finetune and model_name == "efficientnet":
        print(f"\n=== Starting Phase 2: Unfreezing Blocks >= {FINETUNE_UNFREEZE_FROM_BLOCK} (max {finetune_epochs} epochs) ===")
        # Restore best weights from Phase 1 before fine-tuning
        if phase1_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in phase1_best_state.items()})

        unfrozen = unfreeze_last_blocks(model, FINETUNE_UNFREEZE_FROM_BLOCK)
        head_parameters = [
            p for n, p in model.named_parameters()
            if n.startswith("classifier") and p.requires_grad
        ]
        trainable_backbone = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("classifier")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": head_parameters, "lr": learning_rate},
                {"params": trainable_backbone, "lr": FINETUNE_LEARNING_RATE},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, finetune_epochs),
            eta_min=1e-6,
        )
        if finetune_info is not None:
            finetune_info["unfrozen_blocks"] = unfrozen
            finetune_info["phase2_started"] = True

        epochs_without_improvement = 0
        p1_len = len(history["train_loss"])

        for p2_epoch in range(1, finetune_epochs + 1):
            global_epoch = p1_len + p2_epoch
            phase = "fine_tune"
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
                model_name=model_name,
                phase=phase,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
                model_name=model_name,
                phase=phase,
            )
            scheduler.step()

            train_bal_acc = float(
                balanced_accuracy_score(train_metrics["labels"], train_metrics["predictions"])
            )
            val_bal_acc = float(
                balanced_accuracy_score(val_metrics["labels"], val_metrics["predictions"])
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["train_bal_acc"].append(train_bal_acc)
            history["val_bal_acc"].append(val_bal_acc)
            history["phase"].append(phase)

            val_labels = val_metrics["labels"]
            val_preds = val_metrics["predictions"]
            if stage == "binary":
                norm_mask = (val_labels == 0)
                def_mask = (val_labels == 1)
                cur_recall_normal = (
                    float((val_preds[norm_mask] == 0).sum() / norm_mask.sum())
                    if norm_mask.sum() > 0
                    else 0.0
                )
                cur_recall_defective = (
                    float((val_preds[def_mask] == 1).sum() / def_mask.sum())
                    if def_mask.sum() > 0
                    else 0.0
                )
            else:
                cur_recall_normal = None
                cur_recall_defective = None

            print(
                f"[{phase:<9}] Epoch {global_epoch:>3} (p2:{p2_epoch}/{finetune_epochs})  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"train_acc={train_metrics['accuracy']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"train_bal_acc={train_bal_acc:.4f}  "
                f"val_bal_acc={val_bal_acc:.4f}"
            )

            if val_bal_acc > phase2_best_val_bal_acc:
                phase2_best_val_bal_acc = val_bal_acc
                phase2_best_val_acc = val_metrics["accuracy"]
                phase2_best_recall_normal = cur_recall_normal
                phase2_best_recall_defective = cur_recall_defective
                phase2_best_epoch = global_epoch
                phase2_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    print(
                        f"Phase 2 early stopping at global epoch {global_epoch}: validation balanced accuracy did "
                        f"not improve for {patience} epochs."
                    )
                    break

        if phase2_best_state is not None:
            finetuned_ckpt_path = out_model_dir / f"{stage}_{model_name}_finetuned_best.pt"
            finetuned_data = {
                "model_state": phase2_best_state,
                "model_name": model_name,
                "stage": stage,
                "class_names": class_names,
                "num_classes": len(class_names),
                "epoch": phase2_best_epoch,
                "phase": "fine_tune",
                "selection_metric": "balanced_accuracy",
                "val_balanced_accuracy": phase2_best_val_bal_acc,
                "val_accuracy": phase2_best_val_acc,
                "val_recall_normal": phase2_best_recall_normal,
                "val_recall_defective": phase2_best_recall_defective,
                "threshold": threshold,
            }
            if safe_save_checkpoint(finetuned_data, finetuned_ckpt_path, overwrite=overwrite):
                log_print(f"Phase 2 best (val_bal_acc={phase2_best_val_bal_acc:.4f}, epoch={phase2_best_epoch}) saved to {finetuned_ckpt_path}")

    # ------------------ Select Overall Best Checkpoint ------------------
    if phase2_best_val_bal_acc > phase1_best_val_bal_acc:
        winning_phase = "fine_tune"
        best_val_balanced_acc = phase2_best_val_bal_acc
        best_val_acc = phase2_best_val_acc
        best_val_recall_normal = phase2_best_recall_normal
        best_val_recall_defective = phase2_best_recall_defective
        best_epoch = phase2_best_epoch
        best_state = phase2_best_state
    else:
        winning_phase = "head"
        best_val_balanced_acc = phase1_best_val_bal_acc
        best_val_acc = phase1_best_val_acc
        best_val_recall_normal = phase1_best_recall_normal
        best_val_recall_defective = phase1_best_recall_defective
        best_epoch = phase1_best_epoch
        best_state = phase1_best_state

    # Save final best
    best_ckpt_path = out_model_dir / f"{stage}_{model_name}_best.pt"
    best_data = {
        "model_state": best_state,
        "model_name": model_name,
        "stage": stage,
        "class_names": class_names,
        "num_classes": len(class_names),
        "epoch": best_epoch,
        "phase": winning_phase,
        "selection_metric": "balanced_accuracy",
        "val_balanced_accuracy": best_val_balanced_acc,
        "val_accuracy": best_val_acc,
        "val_recall_normal": best_val_recall_normal,
        "val_recall_defective": best_val_recall_defective,
        "threshold": threshold,
    }
    if safe_save_checkpoint(best_data, best_ckpt_path, overwrite=overwrite):
        log_print(f"Overall Best ({winning_phase} phase) saved to {best_ckpt_path}")

    last_epoch_count = len(history["train_loss"])
    last_ckpt_path = out_model_dir / f"{stage}_{model_name}_last.pt"
    last_data = {
        "model_state": model.state_dict(),
        "model_name": model_name,
        "stage": stage,
        "class_names": class_names,
        "num_classes": len(class_names),
        "epoch": last_epoch_count,
        "phase": history["phase"][-1] if history["phase"] else "head",
        "selection_metric": "balanced_accuracy",
        "val_balanced_accuracy": history["val_bal_acc"][-1] if history["val_bal_acc"] else 0.0,
        "val_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "val_recall_normal": cur_recall_normal,
        "val_recall_defective": cur_recall_defective,
        "threshold": threshold,
    }
    safe_save_checkpoint(last_data, last_ckpt_path, overwrite=True)

    metrics = {
        "stage": stage,
        "model": model_name,
        "class_names": class_names,
        "epochs_requested": epochs + (finetune_epochs if finetune and model_name == "efficientnet" else 0),
        "epochs_run": last_epoch_count,
        "best_epoch": best_epoch,
        "winning_phase": winning_phase,
        "phase1_best_val_balanced_accuracy": phase1_best_val_bal_acc,
        "phase1_best_epoch": phase1_best_epoch,
        "phase2_best_val_balanced_accuracy": phase2_best_val_bal_acc if finetune and model_name == "efficientnet" else None,
        "phase2_best_epoch": phase2_best_epoch if finetune and model_name == "efficientnet" else None,
        "selection_metric": "balanced_accuracy",
        "best_val_balanced_accuracy": best_val_balanced_acc,
        "best_val_accuracy": best_val_acc,
        "val_balanced_accuracy": best_val_balanced_acc,
        "val_recall_normal": best_val_recall_normal,
        "val_recall_defective": best_val_recall_defective,
        "final_val_balanced_accuracy": history["val_bal_acc"][-1] if history["val_bal_acc"] else 0.0,
        "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "early_stopped": stopped_early,
        "balanced_sampler": balance,
        "defect_patch_threshold": threshold,
        "finetune": bool(finetune and model_name == "efficientnet"),
        "unfrozen_blocks": finetune_info["unfrozen_blocks"] if finetune_info else [],
        "finetune_info": finetune_info,
        "train_patch_distribution": train_dataset.distribution(),
        "val_patch_distribution": val_dataset.distribution(),
        "train_patch_sources": train_dataset.source_distribution(),
        "train_binary_labels": train_dataset.binary_label_distribution(),
        "val_binary_labels": val_dataset.binary_label_distribution(),
        "history": history,
    }

    with (out_metrics_dir / f"train_{stage}_{model_name}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    save_curves(history, f"{stage}_{model_name}", curves_dir=out_curves_dir)

    print(
        f"\nOverall Best ({winning_phase} phase): val_balanced_accuracy={best_val_balanced_acc:.4f} "
        f"(accuracy={best_val_acc:.4f}, epoch={best_epoch})"
    )
    if stage == "binary" and best_val_recall_normal is not None and best_val_recall_defective is not None:
        print(
            f"Validation Recall -> Normal: {best_val_recall_normal:.4f}, "
            f"Defective: {best_val_recall_defective:.4f}"
        )

    if stage == "defect_type":
        missing = val_dataset.missing_single_image_classes()
        if missing:
            print(
                "Classes with a single original image and therefore no "
                f"validation samples: {missing}"
            )
            print(
                "Hindi: In one-image classes ka validation sample nahi hai, "
                "isliye unka score meaningful nahi hai."
            )

    print(f"Saved to {best_ckpt_path}")

    return metrics
