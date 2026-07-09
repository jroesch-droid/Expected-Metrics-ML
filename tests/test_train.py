"""Smoke tests for the training workflow (tiny study, local MLflow)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.train import build_model_pipeline, generate_synthetic_shots, run_study


def test_synthetic_data_has_signal() -> None:
    frame = generate_synthetic_shots(n_rows=1200, seed=0)
    assert set(frame.columns) >= {"player_id", "x", "y", "goal", "shot_type"}
    assert 0.05 < frame["goal"].mean() < 0.95


def test_pipeline_fits_and_predicts() -> None:
    frame = generate_synthetic_shots(n_rows=800, seed=1)
    pipeline = build_model_pipeline({"n_estimators": 40, "max_depth": 3, "learning_rate": 0.2})
    pipeline.fit(frame, frame["goal"])
    proba = pipeline.predict_proba(frame)[:, 1]
    assert proba.shape == (len(frame),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


@pytest.mark.slow
def test_run_study_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    study = run_study(n_trials=2, n_rows=600, tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    assert study.best_value > 0.0
    assert (tmp_path / "mlflow.db").exists()
