"""English: Unit tests for balanced sampler behavior on binary and defect_type stages.
Hindi: Binary aur defect_type stages ke liye balanced sampler ke unit tests.
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import make_balanced_sampler


def test_binary_sampler_balance():
    """English: Test that binary sampler yields approximately 50/50 balance between
    Normal and Defective patches, regardless of defect type distribution.
    Hindi: Test karta hai ki binary sampler Normal aur Defective patches ko ~50/50
    balance karta hai chahe defect types kitne bhi hon.
    """
    records = []
    # 900 Normal patches
    for i in range(900):
        records.append({
            "index": len(records),
            "class_name": "Normal",
            "binary_label": 0,
            "defect_type_index": None,
        })

    # 100 Defective patches spread unevenly across 10 defect types
    defect_names = [f"defect_{i}" for i in range(10)]
    for i in range(100):
        c_name = defect_names[i % len(defect_names)]
        records.append({
            "index": len(records),
            "class_name": c_name,
            "binary_label": 1,
            "defect_type_index": i % len(defect_names),
        })

    fake_dataset = SimpleNamespace(stage="binary", records=records)
    sampler = make_balanced_sampler(fake_dataset)

    # Sample a large number of indices to verify empirical distribution
    generator = torch.Generator().manual_seed(12345)
    sampled_indices = torch.multinomial(
        sampler.weights, 20000, replacement=True, generator=generator
    ).tolist()

    defective_count = sum(1 for idx in sampled_indices if records[idx]["binary_label"] == 1)
    defective_fraction = defective_count / len(sampled_indices)

    # Defective fraction should be very close to 0.50 (e.g. within 0.47 - 0.53)
    assert 0.47 <= defective_fraction <= 0.53, (
        f"Expected binary sampler defective fraction close to 0.50, got {defective_fraction:.4f}"
    )


def test_defect_type_sampler_balance():
    """English: Test that defect_type sampler equalises weight across distinct defect classes.
    Hindi: Test karta hai ki defect_type sampler alag-alag defect classes ko barabar weight deta hai.
    """
    records = []
    # Imbalanced defect classes: 100 of class_A, 20 of class_B, 5 of class_C, 1 of class_D
    class_counts = {"class_A": 100, "class_B": 20, "class_C": 5, "class_D": 1}
    for c_name, count in class_counts.items():
        for _ in range(count):
            records.append({
                "index": len(records),
                "class_name": c_name,
                "binary_label": 1,
                "defect_type_index": list(class_counts.keys()).index(c_name),
            })

    fake_dataset = SimpleNamespace(stage="defect_type", records=records)
    sampler = make_balanced_sampler(fake_dataset)

    generator = torch.Generator().manual_seed(54321)
    sampled_indices = torch.multinomial(
        sampler.weights, 40000, replacement=True, generator=generator
    ).tolist()

    counts = {c: 0 for c in class_counts}
    for idx in sampled_indices:
        counts[records[idx]["class_name"]] += 1

    total = len(sampled_indices)
    expected_prob = 1.0 / len(class_counts)  # 0.25

    for c_name, count in counts.items():
        fraction = count / total
        assert abs(fraction - expected_prob) < 0.03, (
            f"Expected {c_name} to have fraction ~{expected_prob:.3f}, got {fraction:.4f}"
        )
