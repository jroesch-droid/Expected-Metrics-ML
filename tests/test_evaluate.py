"""Unit tests for the metrics + reporting module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluate import compute_metrics, export_report


def make_predictions(n: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.02, 0.9, size=n)
    y = rng.binomial(1, prob).astype(float)
    return y, prob


def test_metrics_are_sane() -> None:
    y, prob = make_predictions()
    report = compute_metrics(y, prob)
    assert report.n_samples == 400
    assert 0.0 < report.brier_score < 0.25
    assert report.brier_skill_score > 0.0  # informed probs beat base rate
    assert report.roc_auc > 0.7
    assert not report.calibration_table.empty


def test_perfectly_calibrated_gaps_are_small() -> None:
    y, prob = make_predictions(n=5000, seed=1)
    report = compute_metrics(y, prob)
    assert report.calibration_table["gap"].abs().max() < 0.1


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, 0.0]), np.array([0.5, 1.5]))  # prob > 1
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, 1.0]), np.array([0.5, 0.6]))  # single class
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))  # empty


def test_markdown_export(tmp_path: Path) -> None:
    y, prob = make_predictions()
    report = compute_metrics(y, prob)
    path = export_report(report, tmp_path / "reports" / "perf.md", model_name="unit-test-model")
    text = path.read_text()
    assert "# Model Performance Report" in text
    assert "Brier skill score" in text
    assert "| Bin | N |" in text
