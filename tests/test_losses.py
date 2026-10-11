"""English: Unit tests for Tversky Loss and FocalTverskyBCELoss.
Hindi: Tversky loss aur FocalTverskyBCELoss ke liye unit tests.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import FocalTverskyBCELoss, tversky_loss


def test_tversky_loss_perfect_prediction():
    """English: When prediction matches ground truth perfectly, loss should be ~0.
    Hindi: Jab prediction exact match ho, loss ~0 honi chahiye.
    """
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    probs = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = tversky_loss(probs, targets, alpha=0.7, beta=0.3, smooth=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_tversky_loss_empty_mask_edge_case():
    """English: When target mask is all zeros (empty), smooth avoids div by zero.
    Hindi: Jab mask all zeros ho, smooth factor division by zero avoid karta hai.
    """
    targets = torch.zeros((2, 1, 16, 16))
    # Predict all zeros (perfect background match)
    probs_clean = torch.zeros((2, 1, 16, 16))
    loss_clean = tversky_loss(probs_clean, targets, alpha=0.7, beta=0.3, smooth=1.0)
    assert loss_clean.item() == pytest.approx(0.0, abs=1e-5)

    # Predict some false positives on empty mask
    probs_noisy = torch.ones((2, 1, 16, 16)) * 0.5
    loss_noisy = tversky_loss(probs_noisy, targets, alpha=0.7, beta=0.3, smooth=1.0)
    assert loss_noisy.item() > 0.0


def test_focal_tversky_bce_loss():
    """English: Test FocalTverskyBCELoss behavior with perfect and imperfect logits.
    Hindi: FocalTverskyBCELoss ka behavior test karta hai.
    """
    criterion = FocalTverskyBCELoss(alpha=0.7, beta=0.3, gamma=1.33, bce_weight=0.5, smooth=1.0)

    # Large positive logits for 1, large negative logits for 0 (near perfect)
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits_perfect = torch.tensor([[[[15.0, -15.0], [-15.0, 15.0]]]])

    loss_perf = criterion(logits_perfect, targets)
    assert loss_perf.item() < 0.01

    # Completely wrong logits
    logits_wrong = torch.tensor([[[[-15.0, 15.0], [15.0, -15.0]]]])
    loss_wrong = criterion(logits_wrong, targets)
    assert loss_wrong.item() > 1.0
