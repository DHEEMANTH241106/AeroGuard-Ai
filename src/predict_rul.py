from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import prepare_features


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a previously saved model artifact."""
    artifact = joblib.load(path)
    required = {"model", "algorithm", "subset", "feature_columns"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Model artifact at {path} is missing required metadata.")
    return artifact


def predict_rul(
    model: Any,
    observations: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Predict non-negative cycle RUL using only supplied current/past observations."""
    values = np.asarray(model.predict(prepare_features(observations, feature_columns)))
    if values.ndim != 1 or len(values) != len(observations) or not np.isfinite(values).all():
        raise ValueError("The model returned invalid RUL predictions.")
    return np.maximum(0, values)
