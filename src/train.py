"""Optuna hyperparameter optimization with native MLflow tracking.

Runs an Optuna study over an XGBoost expected-goals classifier built on
the leak-safe feature pipeline. Every trial logs its parameters and
cross-validated metrics to MLflow; the final refit logs the model
artifact, a feature-importance matrix, and a calibration curve.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS, build_feature_pipeline

LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger: logging.Logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42
TARGET_COL: str = "goal"


def generate_synthetic_shots(n_rows: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic synthetic shot dataset for reproducible runs.

    Goal probability is driven by distance, visible angle, defender
    pressure, and shooter skill, so the learned model has real signal to
    recover and calibration is meaningful.

    Args:
        n_rows: Number of shot events to synthesize.
        seed: RNG seed for reproducibility.

    Returns:
        A DataFrame with raw columns expected by the feature pipeline.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    player_ids: np.ndarray = rng.integers(1, 120, size=n_rows)
    skill: dict[int, float] = {pid: float(rng.normal(0.0, 0.4)) for pid in np.unique(player_ids)}
    x: np.ndarray = rng.uniform(70.0, 104.5, size=n_rows)
    y: np.ndarray = rng.uniform(10.0, 58.0, size=n_rows)
    defenders: np.ndarray = rng.integers(0, 5, size=n_rows)
    shot_type: np.ndarray = rng.choice(
        ["open_play", "counter", "set_piece", "penalty"], size=n_rows, p=[0.6, 0.15, 0.2, 0.05]
    )
    body_part: np.ndarray = rng.choice(["right_foot", "left_foot", "head"], size=n_rows, p=[0.55, 0.3, 0.15])
    dist: np.ndarray = np.hypot(105.0 - x, 34.0 - y)
    logit: np.ndarray = (
        1.2
        - 0.16 * dist
        - 0.35 * defenders
        + np.vectorize(skill.get)(player_ids)
        + np.where(shot_type == "penalty", 2.4, 0.0)
        + np.where(body_part == "head", -0.5, 0.0)
    )
    prob: np.ndarray = 1.0 / (1.0 + np.exp(-logit))
    goal: np.ndarray = rng.binomial(1, prob)
    frame: pd.DataFrame = pd.DataFrame(
        {
            "player_id": player_ids,
            "x": x,
            "y": y,
            "defenders_between": defenders.astype(float),
            "shot_type": shot_type,
            "body_part": body_part,
            TARGET_COL: goal,
        }
    )
    logger.info("Synthesized %d shots, base rate=%.3f", n_rows, frame[TARGET_COL].mean())
    return frame


def build_model_pipeline(params: dict[str, Any]) -> Pipeline:
    """Compose feature engineering + XGBoost into one pipeline.

    Args:
        params: XGBoost hyperparameters from an Optuna trial.

    Returns:
        Unfitted end-to-end :class:`Pipeline`.
    """
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                    tree_method="hist",
                    **params,
                ),
            ),
        ]
    )


def cross_validated_logloss(frame: pd.DataFrame, params: dict[str, Any], n_splits: int = 4) -> float:
    """Stratified k-fold cross-validated log loss for a parameter set.

    The full pipeline (including target encoding and rolling form) is
    refit inside each fold, so no fold's statistics leak across.

    Args:
        frame: Full raw dataset including the target column.
        params: XGBoost hyperparameters.
        n_splits: Number of stratified folds.

    Returns:
        Mean validation log loss across folds (lower is better).
    """
    splitter: StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    losses: list[float] = []
    target: pd.Series = frame[TARGET_COL]
    for train_idx, valid_idx in splitter.split(frame, target):
        train_frame: pd.DataFrame = frame.iloc[train_idx].reset_index(drop=True)
        valid_frame: pd.DataFrame = frame.iloc[valid_idx].reset_index(drop=True)
        pipeline: Pipeline = build_model_pipeline(params)
        pipeline.fit(train_frame, train_frame[TARGET_COL])
        proba: np.ndarray = pipeline.predict_proba(valid_frame)[:, 1]
        losses.append(float(log_loss(valid_frame[TARGET_COL], proba)))
    return float(np.mean(losses))


def make_objective(frame: pd.DataFrame) -> optuna.study.study.ObjectiveFuncType:
    """Create the Optuna objective closure over the dataset.

    Each trial opens a nested MLflow run recording its parameters and
    cross-validated score.

    Args:
        frame: Full raw training dataset.

    Returns:
        An Optuna-compatible objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            score: float = cross_validated_logloss(frame, params)
            mlflow.log_metric("cv_logloss", score)
        logger.info("Trial %d cv_logloss=%.5f", trial.number, score)
        return score

    return objective


def log_final_artifacts(pipeline: Pipeline, frame: pd.DataFrame) -> None:
    """Log model, feature importance matrix, and calibration curve.

    Args:
        pipeline: Fully fitted end-to-end pipeline.
        frame: Dataset used for the diagnostic plots.

    Returns:
        None.
    """
    proba: np.ndarray = pipeline.predict_proba(frame)[:, 1]
    target: np.ndarray = frame[TARGET_COL].to_numpy()

    mlflow.log_metric("final_roc_auc", float(roc_auc_score(target, proba)))
    mlflow.log_metric("final_brier", float(brier_score_loss(target, proba)))
    mlflow.log_metric("final_logloss", float(log_loss(target, proba)))

    model: XGBClassifier = pipeline.named_steps["model"]
    importance: pd.DataFrame = pd.DataFrame(
        {"feature": list(FEATURE_COLUMNS), "gain_importance": model.feature_importances_.astype(float)}
    ).sort_values("gain_importance", ascending=False)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir: Path = Path(tmp)
        importance_path: Path = tmp_dir / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        frac_pos, mean_pred = calibration_curve(target, proba, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mean_pred, frac_pos, marker="o", label="model")
        ax.plot([0, 1], [0, 1], linestyle="--", label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed goal frequency")
        ax.set_title("xG calibration curve")
        ax.legend()
        calibration_path: Path = tmp_dir / "calibration_curve.png"
        fig.savefig(calibration_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(calibration_path))

    mlflow.sklearn.log_model(
        pipeline,
        name="xg_model",
        skops_trusted_types=[
            "src.features.ColumnSelector",
            "src.features.PitchGeometry",
            "src.features.RollingFormTransformer",
            "src.features.ShotGeometryTransformer",
            "src.features.SmoothedTargetEncoder",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )
    logger.info("Logged model, importance matrix, and calibration curve to MLflow")


def run_study(
    n_trials: int = 25,
    n_rows: int = 6000,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "expected-goals",
) -> optuna.Study:
    """Execute the full optimization + final training workflow.

    Args:
        n_trials: Number of Optuna trials.
        n_rows: Synthetic dataset size.
        tracking_uri: MLflow tracking URI (SQLite backend by default).
        experiment_name: MLflow experiment name.

    Returns:
        The completed Optuna study.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    frame: pd.DataFrame = generate_synthetic_shots(n_rows=n_rows)

    with mlflow.start_run(run_name="optuna-study"):
        study: optuna.Study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(make_objective(frame), n_trials=n_trials, show_progress_bar=False)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_logloss", study.best_value)
        logger.info("Best trial #%d cv_logloss=%.5f", study.best_trial.number, study.best_value)

        final_pipeline: Pipeline = build_model_pipeline(study.best_params)
        final_pipeline.fit(frame, frame[TARGET_COL])
        log_final_artifacts(final_pipeline, frame)
    return study


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Optuna + MLflow xG training")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--rows", type=int, default=6000, help="Synthetic dataset size")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    return parser


if __name__ == "__main__":
    args: argparse.Namespace = build_parser().parse_args()
    run_study(n_trials=args.trials, n_rows=args.rows, tracking_uri=args.tracking_uri)
