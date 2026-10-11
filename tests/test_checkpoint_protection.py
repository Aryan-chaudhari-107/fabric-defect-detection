"""Unit tests for checkpoint overwrite protection and backup mechanism.
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training import safe_save_checkpoint
from config import BACKUP_MODEL_DIR


def test_safe_save_checkpoint_protection(tmp_path, monkeypatch):
    # Redirect BACKUP_MODEL_DIR to tmp_path / "_backup"
    fake_backup_dir = tmp_path / "models" / "_backup"
    monkeypatch.setattr("training.BACKUP_MODEL_DIR", fake_backup_dir)

    target_ckpt = tmp_path / "models" / "classifier" / "test_model_best.pt"

    # Step 1: Save initial checkpoint with score 0.85
    ckpt_initial = {
        "model_state": {"w": torch.tensor([1.0])},
        "val_balanced_accuracy": 0.85,
        "epoch": 5,
    }
    saved = safe_save_checkpoint(ckpt_initial, target_ckpt, overwrite=False)
    assert saved is True
    assert target_ckpt.exists()

    # Step 2: Attempt to overwrite with lower score 0.70 without overwrite flag -> MUST REFUSE
    ckpt_lower = {
        "model_state": {"w": torch.tensor([2.0])},
        "val_balanced_accuracy": 0.70,
        "epoch": 6,
    }
    saved_refused = safe_save_checkpoint(ckpt_lower, target_ckpt, overwrite=False)
    assert saved_refused is False
    # Check that file still has the old score 0.85
    loaded = torch.load(target_ckpt, map_location="cpu")
    assert loaded["val_balanced_accuracy"] == 0.85

    # Step 3: Overwrite with higher score 0.90 -> MUST SUCCEED and BACK UP OLD
    ckpt_higher = {
        "model_state": {"w": torch.tensor([3.0])},
        "val_balanced_accuracy": 0.90,
        "epoch": 7,
    }
    saved_higher = safe_save_checkpoint(ckpt_higher, target_ckpt, overwrite=False)
    assert saved_higher is True
    loaded_new = torch.load(target_ckpt, map_location="cpu")
    assert loaded_new["val_balanced_accuracy"] == 0.90

    # Verify backup exists
    backups = list(fake_backup_dir.glob("test_model_best_*.pt"))
    assert len(backups) >= 1
    old_backup = torch.load(backups[0], map_location="cpu")
    assert old_backup["val_balanced_accuracy"] == 0.85

    # Step 4: Overwrite with lower score 0.60 WITH overwrite=True -> MUST SUCCEED
    ckpt_force = {
        "model_state": {"w": torch.tensor([4.0])},
        "val_balanced_accuracy": 0.60,
        "epoch": 8,
    }
    saved_force = safe_save_checkpoint(ckpt_force, target_ckpt, overwrite=True)
    assert saved_force is True
    loaded_forced = torch.load(target_ckpt, map_location="cpu")
    assert loaded_forced["val_balanced_accuracy"] == 0.60


def test_safe_save_checkpoint_smoke_mode(tmp_path):
    target_smoke_ckpt = tmp_path / "_smoketest_archive" / "test_model_best.pt"
    ckpt_data = {
        "model_state": {"w": torch.tensor([1.0])},
        "val_balanced_accuracy": 0.50,
        "epoch": 1,
    }
    saved = safe_save_checkpoint(ckpt_data, target_smoke_ckpt, overwrite=False)
    assert saved is True
    assert target_smoke_ckpt.exists()
