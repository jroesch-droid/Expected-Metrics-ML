"""Scikit-Learn feature engineering pipeline for expected-metrics models.

Implements three leak-safe custom transformers and a factory that
assembles them into a single ``sklearn.pipeline.Pipeline``:

* ``ShotGeometryTransformer`` — spatial distance/angle features from raw
  (x, y) shot coordinates against a configurable goal location.
* ``RollingFormTransformer`` — shooter-level rolling conversion rates
  computed strictly from *prior* attempts (shifted before rolling).
* ``SmoothedTargetEncoder`` — Bayesian-smoothed target encoding for
  high-cardinality categoricals, fit on training folds only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PitchGeometry:
    """Coordinate frame of the playing surface.

    Attributes:
        goal_x: Goal / basket / plate x-coordinate.
        goal_y: Goal / basket / plate y-coordinate.
        goal_width: Physical width of the target mouth, used for the
            visible-angle feature.
    """

    goal_x: float = 105.0
    goal_y: float = 34.0
    goal_width: float = 7.32


class ShotGeometryTransformer(BaseEstimator, TransformerMixin):
    """Derive spatial features from raw shot coordinates.

    Adds ``dist_to_goal`` (Euclidean), ``angle_to_goal`` (visible angle
    of the goal mouth in radians, the classic xG geometry feature), and
    ``abs_lateral_offset``.
    """

    def __init__(self, geometry: PitchGeometry | None = None, x_col: str = "x", y_col: str = "y") -> None:
        """Configure the transformer.

        Args:
            geometry: Pitch coordinate frame; defaults to a standard
                105x68 football pitch with the goal at (105, 34).
            x_col: Name of the shot x-coordinate column.
            y_col: Name of the shot y-coordinate column.
        """
        self.geometry: PitchGeometry = geometry if geometry is not None else PitchGeometry()
        self.x_col: str = x_col
        self.y_col: str = y_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ShotGeometryTransformer:
        """Validate required columns exist (stateless otherwise).

        Args:
            X: Input frame containing shot coordinates.
            y: Ignored; present for API compatibility.

        Returns:
            self.

        Raises:
            KeyError: If the coordinate columns are missing.
        """
        missing: list[str] = [c for c in (self.x_col, self.y_col) if c not in X.columns]
        if missing:
            raise KeyError(f"Missing coordinate columns: {missing}")
        self.n_features_in_: int = int(X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append geometric features.

        Args:
            X: Input frame containing shot coordinates.

        Returns:
            Copy of ``X`` with ``dist_to_goal``, ``angle_to_goal``, and
            ``abs_lateral_offset`` appended.
        """
        out: pd.DataFrame = X.copy()
        dx: pd.Series = self.geometry.goal_x - out[self.x_col].astype(float)
        dy: pd.Series = self.geometry.goal_y - out[self.y_col].astype(float)
        out["dist_to_goal"] = np.hypot(dx, dy)
        half_width: float = self.geometry.goal_width / 2.0
        left_dy: pd.Series = (self.geometry.goal_y - half_width) - out[self.y_col].astype(float)
        right_dy: pd.Series = (self.geometry.goal_y + half_width) - out[self.y_col].astype(float)
        angle: np.ndarray = np.abs(np.arctan2(right_dy, dx) - np.arctan2(left_dy, dx))
        out["angle_to_goal"] = np.where(angle > math.pi, 2.0 * math.pi - angle, angle)
        out["abs_lateral_offset"] = dy.abs()
        return out


class RollingFormTransformer(BaseEstimator, TransformerMixin):
    """Leak-safe rolling shooter form (historical conversion rate).

    For each entity (shooter), computes the rolling mean of the target
    over the previous ``window`` attempts — shifted by one so the current
    row's outcome never leaks into its own feature. Rows without history
    fall back to the global training base rate learned in :meth:`fit`.
    """

    def __init__(self, entity_col: str = "player_id", target_col: str = "goal", window: int = 20) -> None:
        """Configure the transformer.

        Args:
            entity_col: Column identifying the shooter.
            target_col: Binary outcome column (must exist in ``X``).
            window: Rolling window length in attempts.
        """
        self.entity_col: str = entity_col
        self.target_col: str = target_col
        self.window: int = window

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RollingFormTransformer:
        """Learn the global base rate used as cold-start fallback.

        Args:
            X: Training frame containing the target column.
            y: Ignored; the target is read from ``X`` so the rolling
                history stays aligned with entity ordering.

        Returns:
            self.

        Raises:
            KeyError: If required columns are missing.
        """
        for col in (self.entity_col, self.target_col):
            if col not in X.columns:
                raise KeyError(f"Missing column for rolling form: {col}")
        self.global_rate_: float = float(X[self.target_col].astype(float).mean())
        logger.info("RollingFormTransformer fit: global rate=%.4f", self.global_rate_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the ``rolling_form`` feature.

        Args:
            X: Frame ordered chronologically within each entity.

        Returns:
            Copy of ``X`` with ``rolling_form`` appended.
        """
        out: pd.DataFrame = X.copy()
        shifted: pd.Series = (
            out.groupby(self.entity_col, sort=False)[self.target_col]
            .transform(lambda s: s.astype(float).shift(1).rolling(self.window, min_periods=1).mean())
        )
        out["rolling_form"] = shifted.fillna(self.global_rate_)
        return out


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Bayesian-smoothed target encoding for categorical columns.

    Encodes each category as a blend of its observed target mean and the
    global prior, weighted by category frequency:
    ``(n * mean + k * prior) / (n + k)``. Unseen categories at transform
    time receive the prior.
    """

    def __init__(self, columns: tuple[str, ...] = ("shot_type", "body_part"), smoothing: float = 10.0) -> None:
        """Configure the encoder.

        Args:
            columns: Categorical columns to encode.
            smoothing: Prior weight ``k``; larger values shrink small
                categories harder toward the global mean.
        """
        self.columns: tuple[str, ...] = columns
        self.smoothing: float = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SmoothedTargetEncoder:
        """Learn smoothed per-category encodings from training data.

        Args:
            X: Training frame containing the categorical columns.
            y: Binary target aligned with ``X``.

        Returns:
            self.

        Raises:
            ValueError: If ``y`` is not provided.
            KeyError: If a configured column is missing.
        """
        if y is None:
            raise ValueError("SmoothedTargetEncoder requires y during fit")
        target: pd.Series = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.prior_: float = float(target.mean())
        self.mappings_: dict[str, dict[str, float]] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"Missing categorical column: {col}")
            stats: pd.DataFrame = target.groupby(X[col].astype(str)).agg(["mean", "count"])
            encoded: pd.Series = (stats["count"] * stats["mean"] + self.smoothing * self.prior_) / (
                stats["count"] + self.smoothing
            )
            self.mappings_[col] = encoded.to_dict()
        logger.info("Target encoder fit on %d columns (prior=%.4f)", len(self.columns), self.prior_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categorical columns with their smoothed encodings.

        Args:
            X: Frame to encode.

        Returns:
            Copy of ``X`` with ``<col>_te`` columns appended and raw
            categorical columns dropped.
        """
        out: pd.DataFrame = X.copy()
        for col in self.columns:
            mapping: dict[str, float] = self.mappings_.get(col, {})
            out[f"{col}_te"] = out[col].astype(str).map(mapping).fillna(self.prior_)
            out = out.drop(columns=[col])
        return out


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Terminal step that projects to the numeric model matrix."""

    def __init__(self, feature_columns: tuple[str, ...]) -> None:
        """Configure the selector.

        Args:
            feature_columns: Final feature columns, in model order.
        """
        self.feature_columns: tuple[str, ...] = feature_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ColumnSelector:
        """Record fitted state (the projection itself is stateless).

        Args:
            X: Frame used only to record input width.
            y: Ignored.

        Returns:
            self.
        """
        self.n_features_in_: int = int(X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project to the configured columns.

        Args:
            X: Frame containing at least ``feature_columns``.

        Returns:
            Numeric feature matrix as a DataFrame.

        Raises:
            KeyError: If any configured column is missing.
        """
        missing: list[str] = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise KeyError(f"Feature columns missing after transforms: {missing}")
        return X.loc[:, list(self.feature_columns)].astype(float)


FEATURE_COLUMNS: tuple[str, ...] = (
    "dist_to_goal",
    "angle_to_goal",
    "abs_lateral_offset",
    "rolling_form",
    "shot_type_te",
    "body_part_te",
    "defenders_between",
)


def build_feature_pipeline(window: int = 20, smoothing: float = 10.0) -> Pipeline:
    """Assemble the full leak-safe feature engineering pipeline.

    Args:
        window: Rolling-form window in attempts.
        smoothing: Target-encoder prior weight.

    Returns:
        An unfitted :class:`sklearn.pipeline.Pipeline` producing the
        numeric matrix defined by ``FEATURE_COLUMNS``.
    """
    return Pipeline(
        steps=[
            ("geometry", ShotGeometryTransformer()),
            ("rolling_form", RollingFormTransformer(window=window)),
            ("target_encode", SmoothedTargetEncoder(smoothing=smoothing)),
            ("select", ColumnSelector(feature_columns=FEATURE_COLUMNS)),
        ]
    )
