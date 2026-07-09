"""Validation module: custom sports metrics and markdown reporting.

Computes probabilistic quality metrics (Brier score, ROC-AUC, log loss),
Brier skill score against the naive base-rate forecaster, and a decile
calibration table, then renders everything as a markdown performance
report suitable for committing next to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricReport:
    """Container for all computed validation metrics.

    Attributes:
        n_samples: Number of evaluated events.
        base_rate: Empirical positive rate of the evaluation set.
        brier_score: Mean squared probability error (lower is better).
        brier_skill_score: Improvement over the base-rate forecaster;
            positive values mean the model beats "always predict the
            base rate".
        roc_auc: Ranking quality in ``[0.5, 1.0]``.
        logloss: Negative log likelihood (lower is better).
        calibration_table: Decile bins with predicted vs observed rates.
    """

    n_samples: int
    base_rate: float
    brier_score: float
    brier_skill_score: float
    roc_auc: float
    logloss: float
    calibration_table: pd.DataFrame


def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Validate metric inputs.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.

    Raises:
        ValueError: On shape mismatch, empty input, out-of-range
            probabilities, non-binary labels, or single-class labels.
    """
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    unique: np.ndarray = np.unique(y_true)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(f"y_true must be binary; found values {unique}")
    if unique.size < 2:
        raise ValueError("y_true contains a single class; ROC-AUC is undefined")


def decile_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions into quantiles and compare to observed rates.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.
        n_bins: Number of quantile bins.

    Returns:
        DataFrame with one row per bin: count, mean prediction, observed
        rate, and the calibration gap.
    """
    frame: pd.DataFrame = pd.DataFrame({"y": y_true.astype(float), "p": y_prob.astype(float)})
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    grouped: pd.DataFrame = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    grouped["gap"] = grouped["observed_rate"] - grouped["mean_predicted"]
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricReport:
    """Compute the full sports-model validation metric suite.

    Args:
        y_true: Binary outcomes (goal / no goal).
        y_prob: Model probabilities.

    Returns:
        Populated :class:`MetricReport`.

    Raises:
        ValueError: If inputs fail validation.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    _validate_inputs(y_true, y_prob)

    base_rate: float = float(y_true.mean())
    brier: float = float(brier_score_loss(y_true, y_prob))
    reference_brier: float = float(brier_score_loss(y_true, np.full_like(y_prob, base_rate)))
    skill: float = 1.0 - brier / reference_brier if reference_brier > 0.0 else 0.0

    report: MetricReport = MetricReport(
        n_samples=int(y_true.size),
        base_rate=base_rate,
        brier_score=brier,
        brier_skill_score=float(skill),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        logloss=float(log_loss(y_true, y_prob)),
        calibration_table=decile_calibration_table(y_true, y_prob),
    )
    logger.info(
        "Metrics: n=%d brier=%.4f bss=%.4f auc=%.4f logloss=%.4f",
        report.n_samples,
        report.brier_score,
        report.brier_skill_score,
        report.roc_auc,
        report.logloss,
    )
    return report


def render_markdown_report(report: MetricReport, model_name: str = "expected-goals-xgb") -> str:
    """Render a :class:`MetricReport` as a markdown document.

    Args:
        report: Computed metrics.
        model_name: Display name of the evaluated model.

    Returns:
        Markdown source of the performance report.
    """
    generated: str = datetime.now(tz=UTC).isoformat(timespec="seconds")
    lines: list[str] = [
        f"# Model Performance Report — `{model_name}`",
        "",
        f"Generated: {generated}",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---|---|",
        f"| Samples | {report.n_samples} | Evaluation set size |",
        f"| Base rate | {report.base_rate:.4f} | Empirical conversion rate |",
        f"| Brier score | {report.brier_score:.4f} | Mean squared probability error (lower is better) |",
        f"| Brier skill score | {report.brier_skill_score:.4f} "
        "| Improvement vs. base-rate forecaster (>0 beats naive) |",
        f"| ROC-AUC | {report.roc_auc:.4f} | Ranking quality (0.5 = random) |",
        f"| Log loss | {report.logloss:.4f} | Probabilistic likelihood penalty (lower is better) |",
        "",
        "## Calibration by Predicted-Probability Decile",
        "",
        "| Bin | N | Mean predicted | Observed rate | Gap |",
        "|---|---|---|---|---|",
    ]
    for row in report.calibration_table.itertuples(index=False):
        lines.append(
            f"| {row.bin} | {row.n} | {row.mean_predicted:.4f} | {row.observed_rate:.4f} | {row.gap:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- A well-calibrated expected-metrics model shows per-decile gaps near zero:",
            "  shots the model rates at 0.30 should convert about 30% of the time.",
            "- Brier skill score is the headline number for stakeholders: it answers",
            '  "how much better than just guessing the league conversion rate?"',
            "- ROC-AUC alone is insufficient for xG-style models — a model can rank",
            "  perfectly while being badly miscalibrated, which corrupts aggregated",
            "  season totals.",
            "",
        ]
    )
    return "\n".join(lines)


def export_report(report: MetricReport, output_path: Path, model_name: str = "expected-goals-xgb") -> Path:
    """Write the markdown report to disk.

    Args:
        report: Computed metrics.
        output_path: Destination ``.md`` file.
        model_name: Display name of the evaluated model.

    Returns:
        The written path.

    Raises:
        OSError: If the parent directory cannot be created or written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(report, model_name=model_name), encoding="utf-8")
    logger.info("Wrote performance report to %s", output_path)
    return output_path
