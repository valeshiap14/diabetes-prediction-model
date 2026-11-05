"""Auditable inference helpers for a non-clinical classification demonstration.

This module deliberately contains no Streamlit calls, so input handling and
threshold decisions can be tested without starting a web server.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import pandas as pd

FEATURES = (
    "age", "hypertension", "heart_disease", "bmi",
    "HbA1c_level", "blood_glucose_level", "gender", "smoking_history",
)
NUMERIC_RANGE = {
    "age": (0, 120),
    "bmi": (10, 80),
    "HbA1c_level": (3, 20),
    "blood_glucose_level": (40, 500),
}
BINARY = {"hypertension", "heart_disease"}


@dataclass(frozen=True)
class Prediction:
    probability: float
    label: int
    threshold: float


def make_patient_row(values: dict[str, Any]) -> pd.DataFrame:
    """Validate an explicit fictional input record and preserve training column names.

    Ranges are application-level plausibility guards, not diagnostic cutoffs.
    """
    absent = set(FEATURES) - set(values)
    if absent:
        raise ValueError(f"Missing fields: {', '.join(sorted(absent))}")
    row = {key: values[key] for key in FEATURES}
    for name, (minimum, maximum) in NUMERIC_RANGE.items():
        try:
            value = float(row[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        row[name] = value
    for name in BINARY:
        if row[name] not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
        row[name] = int(row[name])
    for name in ("gender", "smoking_history"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f"{name} must be a non-empty category")
        row[name] = row[name].strip()
    return pd.DataFrame([row], columns=FEATURES)


def validate_metadata(metadata: dict[str, Any]) -> float:
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    try:
        threshold = float(metadata["decision_threshold"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("model_metadata.json must include decision_threshold") from exc
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("decision_threshold must be between 0 and 1")
    if "features" in metadata and set(metadata["features"]) != set(FEATURES):
        raise ValueError("Saved model metadata feature list does not match this dashboard")
    return threshold


def load_metadata(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_metadata(data)
    return data


def predict(pipeline: Any, row: pd.DataFrame, threshold: float) -> Prediction:
    """Use the model's true probability, never a fabricated dashboard score."""
    validate_metadata({"decision_threshold": threshold})
    if not all(feature in row.columns for feature in FEATURES):
        raise ValueError("The model requires all eight input features")
    scores = pipeline.predict_proba(row)
    if getattr(scores, "shape", None) != (1, 2):
        raise ValueError("Expected one prediction with two class probabilities")
    score = float(scores[0, 1])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid probability")
    return Prediction(score, int(score >= threshold), threshold)


def metric_rows(metadata: dict[str, Any]) -> list[tuple[str, float]]:
    """Only show metrics that come from a saved actual evaluation artifact."""
    result = []
    for key, value in metadata.get("test_metrics", {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append((key, number))
    return result
