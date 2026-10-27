from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard RUL regression metrics in cycle units."""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
    }


def make_evaluation_frame(
    source: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    result = source[["engine_id", "cycle"]].copy()
    result["actual_rul"] = actual
    result["predicted_rul"] = predicted
    result["residual"] = result["predicted_rul"] - result["actual_rul"]
    return result


def model_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "feature_importances_"):
        return None
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": estimator.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
