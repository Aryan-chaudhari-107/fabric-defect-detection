"""English: Training loop and checkpointing for the U-Net segmenter.
Hindi: U-Net segmenter ke liye training loop aur checkpoint logic.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    EARLY_STOPPING_PATIENCE,
    METRICS_DIR,
    SEGMENTER_MODEL_DIR,
    SEGMENTATION_THRESHOLD,
    TRAINING_CURVES_DIR,
)
from dataset import SegmentationPatchDataset
from models import DiceBCELoss, FocalTverskyBCELoss, UNet, binary_metrics
from training import safe_save_checkpoint


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    """English: Run one segmentation epoch and return loss plus mask metrics.
    Hindi: Ek segmentation epoch chalata hai aur loss ke saath mask metrics deta hai.
    """
    model.train() if train else model.eval()

    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    total_items = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            batch_size = images.size(0)
            totals["loss"] += loss.item() * batch_size
            for key, value in binary_metrics(
                logits, masks, SEGMENTATION_THRESHOLD
            ).items():
                totals[key] += value * batch_size
            total_items += batch_size

            if train:
                loss.backward()
                optimizer.step()

    divisor = max(total_items, 1)
    return {key: value / divisor for key, value in totals.items()}


def save_curves(
    history: Dict[str, List[float]],
    curves_dir: Optional[Path] = None,
) -> None:
    """English: Plot segmentation loss, Dice and IoU curves.
    Hindi: Segmentation ke loss, Dice aur IoU curves plot karta hai.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = curves_dir if curves_dir is not None else TRAINING_CURVES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, key, title in zip(
        axes,
        ("loss", "dice", "iou"),
        ("Loss", "Dice", "IoU"),
    ):
        axis.plot(history[f"train_{key}"], label="train")
        axis.plot(history[f"val_{key}"], label="validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()

    figure.tight_layout()
    figure.savefig(out_dir / "unet_curves.png", dpi=120)
    plt.close(figure)


def train_segmenter(
    train_dataset: SegmentationPatchDataset,
    val_dataset: SegmentationPatchDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    patience: int,
    checkpoint_name: str = "unet",
    model_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    curves_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict:
    """English: Full segmentation training routine.
    Hindi: Poora segmentation training routine.

    English: The best checkpoint is selected on validation Dice rather than
    loss, because Dice is the metric that reflects whether a very small defect
    was actually found.
    Hindi: Best checkpoint validation Dice par select hota hai, loss par nahi,
    kyunki Dice batata hai ki chhota defect sach mein mila ya nahi.
    """
    out_model_dir = model_dir if model_dir is not None else SEGMENTER_MODEL_DIR
    out_metrics_dir = metrics_dir if metrics_dir is not None else METRICS_DIR
    out_curves_dir = curves_dir if curves_dir is not None else TRAINING_CURVES_DIR

    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_curves_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = FocalTverskyBCELoss(alpha=0.7, beta=0.3, bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
    }

    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        for key in ("loss", "dice", "iou"):
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"train_dice={train_metrics['dice']:.4f}  "
            f"val_dice={val_metrics['dice']:.4f}  "
            f"train_iou={train_metrics['iou']:.4f}  "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_data = {
                "model_state": model.state_dict(),
                "threshold": SEGMENTATION_THRESHOLD,
                "epoch": epoch,
                "val_dice": best_dice,
                "val_iou": val_metrics["iou"],
            }
            safe_save_checkpoint(
                best_data,
                out_model_dir / f"{checkpoint_name}_best.pt",
                overwrite=overwrite,
                metric_key="val_dice",
            )
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: validation Dice did not "
                    f"improve for {patience} epochs."
                )
                stopped_early = True
                break

    last_data = {
        "model_state": model.state_dict(),
        "threshold": SEGMENTATION_THRESHOLD,
        "epoch": len(history["train_loss"]),
        "val_dice": history["val_dice"][-1] if history["val_dice"] else 0.0,
    }
    safe_save_checkpoint(
        last_data,
        out_model_dir / f"{checkpoint_name}_last.pt",
        overwrite=True,
        metric_key="val_dice",
    )

    metrics = {
        "epochs_requested": epochs,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "early_stopped": stopped_early,
        "train_patches": len(train_dataset),
        "val_patches": len(val_dataset),
        "history": history,
    }

    with (out_metrics_dir / f"train_{checkpoint_name}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    save_curves(history, curves_dir=out_curves_dir)

    print(f"\nBest validation Dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"Saved to {out_model_dir / f'{checkpoint_name}_best.pt'}")

    return metrics
