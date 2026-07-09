"""Unit tests for the custom feature-engineering transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    RollingFormTransformer,
    ShotGeometryTransformer,
    SmoothedTargetEncoder,
    build_feature_pipeline,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2],
            "x": [100.0, 95.0, 88.0, 104.0, 80.0],
            "y": [34.0, 30.0, 40.0, 34.0, 20.0],
            "defenders_between": [0.0, 2.0, 1.0, 0.0, 3.0],
            "shot_type": ["open_play", "open_play", "set_piece", "penalty", "counter"],
            "body_part": ["right_foot", "head", "left_foot", "right_foot", "right_foot"],
            "goal": [1, 0, 0, 1, 0],
        }
    )


def test_geometry_distance_and_angle() -> None:
    out = ShotGeometryTransformer().fit_transform(sample_frame())
    # shot from (100, 34): straight in front, 5m out
    assert out.loc[0, "dist_to_goal"] == pytest.approx(5.0)
    assert out.loc[0, "abs_lateral_offset"] == pytest.approx(0.0)
    # central shots see a wider angle than wide shots at similar range
    assert out.loc[0, "angle_to_goal"] > out.loc[4, "angle_to_goal"]


def test_geometry_missing_columns_raises() -> None:
    with pytest.raises(KeyError):
        ShotGeometryTransformer().fit(pd.DataFrame({"a": [1]}))


def test_rolling_form_is_leak_free() -> None:
    frame = sample_frame()
    out = RollingFormTransformer(window=10).fit_transform(frame)
    base_rate = frame["goal"].mean()
    # first attempt per player has no history -> global rate fallback
    assert out.loc[0, "rolling_form"] == pytest.approx(base_rate)
    assert out.loc[3, "rolling_form"] == pytest.approx(base_rate)
    # second attempt for player 1 sees ONLY the first outcome (1.0), not its own
    assert out.loc[1, "rolling_form"] == pytest.approx(1.0)
    # third attempt sees mean of first two outcomes (1, 0)
    assert out.loc[2, "rolling_form"] == pytest.approx(0.5)


def test_target_encoder_smoothing_and_unseen() -> None:
    frame = sample_frame()
    encoder = SmoothedTargetEncoder(columns=("shot_type",), smoothing=10.0)
    encoded = encoder.fit_transform(frame, frame["goal"])
    assert "shot_type_te" in encoded.columns
    assert "shot_type" not in encoded.columns
    prior = frame["goal"].mean()
    # heavy smoothing pulls a 1-observation category close to the prior
    assert abs(encoded.loc[3, "shot_type_te"] - prior) < abs(1.0 - prior)
    # unseen category at transform time -> exactly the prior
    unseen = frame.copy()
    unseen.loc[0, "shot_type"] = "bicycle_kick"
    out = encoder.transform(unseen)
    assert out.loc[0, "shot_type_te"] == pytest.approx(prior)


def test_target_encoder_requires_y() -> None:
    with pytest.raises(ValueError):
        SmoothedTargetEncoder(columns=("shot_type",)).fit(sample_frame(), None)


def test_full_pipeline_output_matrix() -> None:
    frame = sample_frame()
    pipeline = build_feature_pipeline()
    matrix = pipeline.fit_transform(frame, frame["goal"])
    assert list(matrix.columns) == list(FEATURE_COLUMNS)
    assert matrix.shape == (len(frame), len(FEATURE_COLUMNS))
    assert np.isfinite(matrix.to_numpy()).all()
