"""Tests for cancellation live-prediction helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def test_heuristic_score_bounded():
    from src.module_1_predictive.cancellation.predict import heuristic_cancellation_probability

    result = heuristic_cancellation_probability(
        lead_time=200,
        hotel_type="City Hotel",
        deposit_type="No Deposit",
        is_repeated=0,
        prev_cancellations=1,
        special_requests=0,
    )
    assert result["source"] == "heuristic"
    assert 0.05 <= result["probability"] <= 0.99


def test_build_feature_row_overrides_known_inputs():
    from src.module_1_predictive.cancellation.predict import build_feature_row

    feature_names = [
        "lead_time",
        "adr",
        "adults",
        "hotel_encoded",
        "deposit_type_encoded",
        "is_repeated_guest",
        "previous_cancellations",
        "total_of_special_requests",
        "market_segment_Online TA",
        "market_segment_Groups",
    ]
    medians = {name: 0.0 for name in feature_names}
    row = build_feature_row(
        feature_names=feature_names,
        medians=medians,
        lead_time=120,
        adr=150.0,
        adults=2,
        hotel_type="City Hotel",
        deposit_type="Non Refund",
        market_segment="Online TA",
        is_repeated=1,
        prev_cancellations=2,
        special_requests=3,
    )
    assert row.loc[0, "lead_time"] == 120
    assert row.loc[0, "hotel_encoded"] == 1
    assert row.loc[0, "deposit_type_encoded"] == 2
    assert row.loc[0, "market_segment_Online TA"] == 1
    assert row.loc[0, "market_segment_Groups"] == 0


def test_predict_returns_none_without_artifacts(tmp_path, monkeypatch):
    from src.module_1_predictive.cancellation import predict as predict_mod

    monkeypatch.setattr(predict_mod, "MODELS_DIR", tmp_path)
    assert predict_mod.predict_cancellation_probability(
        lead_time=10,
        adr=100.0,
        adults=2,
        hotel_type="City Hotel",
        deposit_type="No Deposit",
        market_segment="Online TA",
        is_repeated=0,
        prev_cancellations=0,
        special_requests=1,
    ) is None


def test_predict_uses_model_when_artifacts_present(tmp_path, monkeypatch):
    from src.module_1_predictive.cancellation import predict as predict_mod

    feature_names = ["lead_time", "adr", "adults", "hotel_encoded", "deposit_type_encoded"]
    metadata = {
        "best_model": "XGBoost",
        "feature_names": feature_names,
        "feature_medians": {name: 1.0 for name in feature_names},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "best_model.joblib").write_bytes(b"placeholder")

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    monkeypatch.setattr(predict_mod, "MODELS_DIR", tmp_path)
    with patch.object(predict_mod.joblib, "load", return_value=mock_model):
        result = predict_mod.predict_cancellation_probability(
            lead_time=10,
            adr=100.0,
            adults=2,
            hotel_type="City Hotel",
            deposit_type="No Deposit",
            market_segment="Online TA",
            is_repeated=0,
            prev_cancellations=0,
            special_requests=1,
        )

    assert result is not None
    assert result["source"] == "model"
    assert result["probability"] == pytest.approx(0.8)
