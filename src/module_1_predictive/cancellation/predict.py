"""
Cancellation risk prediction helpers for the Streamlit live-demo form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / "models" / "cancellation"

HOTEL_MAP = {"Resort Hotel": 0, "City Hotel": 1}
DEPOSIT_MAP = {"No Deposit": 0, "Refundable": 1, "Non Refund": 2}


def artifacts_available() -> bool:
    return (MODELS_DIR / "best_model.joblib").exists() and (MODELS_DIR / "metadata.json").exists()


def load_metadata() -> dict:
    with open(MODELS_DIR / "metadata.json") as f:
        return json.load(f)


def _base_feature_row(feature_names: list[str], medians: dict) -> dict:
    row = {}
    for name in feature_names:
        row[name] = float(medians.get(name, 0.0))
    return row


def build_feature_row(
    *,
    feature_names: list[str],
    medians: dict,
    lead_time: int,
    adr: float,
    adults: int,
    hotel_type: str,
    deposit_type: str,
    market_segment: str,
    is_repeated: int,
    prev_cancellations: int,
    special_requests: int,
) -> pd.DataFrame:
    """Build a single-row feature frame matching training columns."""
    row = _base_feature_row(feature_names, medians)

    overrides = {
        "lead_time": float(lead_time),
        "adr": float(adr),
        "adults": float(adults),
        "is_repeated_guest": float(is_repeated),
        "previous_cancellations": float(prev_cancellations),
        "total_of_special_requests": float(special_requests),
        "hotel_encoded": float(HOTEL_MAP.get(hotel_type, 0)),
        "deposit_type_encoded": float(DEPOSIT_MAP.get(deposit_type, 0)),
    }
    for key, value in overrides.items():
        if key in row:
            row[key] = value

    # Market segment one-hots produced by get_dummies(drop_first=True)
    segment_cols = [c for c in feature_names if c.startswith("market_segment_")]
    for col in segment_cols:
        row[col] = 0.0
    target_col = f"market_segment_{market_segment}"
    if target_col in row:
        row[target_col] = 1.0

    return pd.DataFrame([row], columns=feature_names)


def predict_cancellation_probability(
    *,
    lead_time: int,
    adr: float,
    adults: int,
    hotel_type: str,
    deposit_type: str,
    market_segment: str,
    is_repeated: int,
    prev_cancellations: int,
    special_requests: int,
) -> Optional[dict]:
    """
    Return prediction dict when model artifacts exist, else None.
    """
    if not artifacts_available():
        return None

    metadata = load_metadata()
    feature_names = metadata["feature_names"]
    medians = metadata.get("feature_medians", {})

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    X = build_feature_row(
        feature_names=feature_names,
        medians=medians,
        lead_time=lead_time,
        adr=adr,
        adults=adults,
        hotel_type=hotel_type,
        deposit_type=deposit_type,
        market_segment=market_segment,
        is_repeated=is_repeated,
        prev_cancellations=prev_cancellations,
        special_requests=special_requests,
    )

    # Logistic Regression was trained on scaled features; tree models were not.
    best_name = metadata.get("best_model", "")
    if best_name == "Logistic Regression" and (MODELS_DIR / "scaler.joblib").exists():
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        X_input = pd.DataFrame(scaler.transform(X), columns=feature_names)
    else:
        X_input = X

    proba = float(model.predict_proba(X_input)[0][1])
    return {
        "probability": proba,
        "model_name": best_name,
        "source": "model",
    }


def heuristic_cancellation_probability(
    *,
    lead_time: int,
    hotel_type: str,
    deposit_type: str,
    is_repeated: int,
    prev_cancellations: int,
    special_requests: int,
) -> dict:
    """Fallback score used when trained artifacts are unavailable."""
    risk_score = min(
        0.15
        + (lead_time / 1000)
        + (0.3 if deposit_type == "No Deposit" else -0.2)
        + (prev_cancellations * 0.1)
        - (special_requests * 0.05)
        - (is_repeated * 0.15)
        + (0.1 if hotel_type == "City Hotel" else 0),
        0.99,
    )
    risk_score = max(risk_score, 0.05)
    return {
        "probability": float(risk_score),
        "model_name": "heuristic",
        "source": "heuristic",
    }
