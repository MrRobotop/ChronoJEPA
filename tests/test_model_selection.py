"""Tests for label-free model selection by SIGReg-loss ranking."""

import numpy as np
import pytest

from chronojepa.eval import label_free_model_selection, selection_report


def test_spearman_positive_and_picks_agree_when_sigreg_predicts() -> None:
    names = [f"run{i}" for i in range(8)]
    sigreg = np.linspace(1.0, 2.0, 8)
    # downstream error rises monotonically with sigreg loss, so lower sigreg is better
    downstream = 2.0 * sigreg + 0.01 * np.array([0, 1, -1, 0, 1, -1, 0, 1])

    result = label_free_model_selection(names, sigreg, downstream, lower_is_better=True)

    assert result["spearman"] > 0.8
    assert result["label_free_pick"] == result["label_based_pick"] == "run0"
    assert result["agree"] is True


def test_spearman_negative_when_anticorrelated() -> None:
    names = [f"run{i}" for i in range(6)]
    sigreg = np.linspace(1.0, 2.0, 6)
    downstream = -sigreg  # higher sigreg loss maps to lower error: anticorrelated

    result = label_free_model_selection(names, sigreg, downstream, lower_is_better=True)
    assert result["spearman"] < -0.8


def test_requires_at_least_three_runs() -> None:
    with pytest.raises(ValueError):
        label_free_model_selection(["a", "b"], [1.0, 2.0], [1.0, 2.0])


def test_selection_report_is_readable() -> None:
    result = label_free_model_selection(
        ["a", "b", "c"], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], lower_is_better=True
    )
    report = selection_report(result)
    assert "spearman" in report.lower()
    assert "a" in report
