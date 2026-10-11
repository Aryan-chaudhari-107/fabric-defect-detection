"""English: Unit tests for Phase 1 -> Phase 2 fine-tuning transition.
Hindi: Phase 1 se Phase 2 fine-tuning transition ka unit test.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import build_efficientnet
from training import train


class SyntheticDataset(Dataset):
    """Synthetic dataset generating random tensors for testing training loops."""

    def __init__(self, count: int = 16, num_classes: int = 2):
        self.count = count
        self.num_classes = num_classes
        self.stage = "binary"
        self.records = [
            {
                "binary_label": i % 2,
                "class_name": "Normal" if (i % 2 == 0) else "Defective",
                "source": "synthetic",
            }
            for i in range(count)
        ]

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = torch.randn(3, 256, 256)
        y = self.records[idx]["binary_label"]
        return x, y

    def distribution(self) -> Dict[str, int]:
        return {"Normal": self.count // 2, "Defective": self.count // 2}

    def source_distribution(self) -> Dict[str, int]:
        return {"synthetic": self.count}

    def binary_label_distribution(self) -> Dict[str, int]:
        return {"Normal": self.count // 2, "Defective": self.count // 2}


def test_finetuning_two_phase_transition(tmp_path):
    """English: Test that Phase 1 finishes and smoothly transitions to Phase 2.
    Hindi: Test karta hai ki Phase 1 ke baad Phase 2 transition sahi tareeqe se hota hai.
    """
    train_ds = SyntheticDataset(count=16, num_classes=2)
    val_ds = SyntheticDataset(count=8, num_classes=2)

    metrics = train(
        stage="binary",
        model_name="efficientnet",
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        num_workers=0,
        finetune=True,
        finetune_epochs=1,
        patience=1,
        balance=False,
        threshold=0.0002,
    )

    assert "phase1_best_epoch" in metrics
    assert "phase2_best_epoch" in metrics
    assert "winning_phase" in metrics
    assert "unfrozen_blocks" in metrics
    assert len(metrics["unfrozen_blocks"]) > 0
    assert metrics["winning_phase"] in ("head", "fine_tune")
