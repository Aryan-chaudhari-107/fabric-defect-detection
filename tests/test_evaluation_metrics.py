"""English: Unit tests for evaluation metrics, Wilson confidence intervals, and support logic.
Hindi: Evaluation metrics aur Wilson confidence intervals ke unit tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import wilson_interval


def test_wilson_interval():
    """English: Verify Wilson 95% confidence intervals against known ground truth values.
    Hindi: Known values ke against Wilson confidence interval test karta hai.
    """
    # 14 / 16 successes -> ~[0.6398, 0.9650]
    low_14, high_14 = wilson_interval(14, 16)
    assert low_14 == pytest.approx(0.6398, abs=0.005)
    assert high_14 == pytest.approx(0.9650, abs=0.005)

    # 3 / 21 successes -> ~[0.0504, 0.3464]
    low_3, high_3 = wilson_interval(3, 21)
    assert low_3 == pytest.approx(0.0504, abs=0.005)
    assert high_3 == pytest.approx(0.3464, abs=0.005)


def test_wilson_interval_edge_cases():
    """English: Test 0/0 and 0/N edge cases for Wilson confidence intervals.
    Hindi: Edge cases test karta hai.
    """
    low_0, high_0 = wilson_interval(0, 0)
    assert low_0 == 0.0 and high_0 == 0.0

    low_zero, high_zero = wilson_interval(0, 10)
    assert low_zero == 0.0
    assert high_zero > 0.0
