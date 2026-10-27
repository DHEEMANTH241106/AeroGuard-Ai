from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate_model import make_evaluation_frame, regression_metrics
from src.preprocessing import add_temporal_features, get_feature_columns, prepare_features
from src.rul_calculator import add_training_rul

ALGORITHMS = ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]
RANDOM_STATE = 42


def _make_estimator(algorithm: str) -> Any:
    if algorithm == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install it with `pip install xgboost`, "
                "or explicitly select Random Forest."
            ) from exc
        return XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if algorithm == "Random Forest":
        return RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )
    if algorithm == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        )
    if algorithm == "Linear Regression":
        return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _engine_split(frame: pd.DataFrame, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    if frame["engine_id"].nunique() < 2:
        raise ValueError("At least two distinct engines are required for engine-level validation.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    return next(
        splitter.split(frame, groups=frame["engine_id"])
    )


def compare_models(
    train_frame: pd.DataFrame,
    algorithms: list[str] | None = None,
    clip_threshold: int | None = 125,
    feature_columns: list[str] | None = None,
    rolling_window: int | None = None,
    include_differences: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Compare algorithms on held-out engines; no row-level leakage between splits."""
    engineered = add_temporal_features(train_frame, rolling_window, include_differences)
    labeled = add_training_rul(engineered, clip_threshold)
    features = (
        feature_columns
        if feature_columns is not None
        else get_feature_columns(labeled, rolling_window, include_differences)
    )
    if not features:
        raise ValueError("Select at least one model feature.")
    train_index, validation_index = _engine_split(labeled)
    x_train = prepare_features(labeled.iloc[train_index], features)
    x_validation = prepare_features(labeled.iloc[validation_index], features)
    y_train = labeled.iloc[train_index]["rul"].to_numpy()
    y_validation = labeled.iloc[validation_index]["rul"].to_numpy()
    reports: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}

    for algorithm in algorithms or ALGORITHMS:
        model = _make_estimator(algorithm)
        model.fit(x_train, y_train)
        predicted = np.maximum(0, model.predict(x_validation))
        reports.append({"Algorithm": algorithm, **regression_metrics(y_validation, predicted)})
        predictions[algorithm] = {
            "model": model,
            "evaluation": make_evaluation_frame(
                labeled.iloc[validation_index], y_validation, predicted
            ),
        }
    return pd.DataFrame(reports).sort_values("MAE").reset_index(drop=True), predictions


def fit_and_save_model(
    train_frame: pd.DataFrame,
    algorithm: str,
    subset: str,
    model_dir: str | Path = "models",
    clip_threshold: int | None = 125,
    feature_columns: list[str] | None = None,
    rolling_window: int | None = None,
    include_differences: bool = False,
    data_source: str = "user-provided training data",
) -> tuple[Any, Path]:
    """Fit the selected model on all provided training engines and persist metadata."""
    engineered = add_temporal_features(train_frame, rolling_window, include_differences)
    labeled = add_training_rul(engineered, clip_threshold)
    features = (
        feature_columns
        if feature_columns is not None
        else get_feature_columns(labeled, rolling_window, include_differences)
    )
    if not features:
        raise ValueError("Select at least one model feature.")
    model = _make_estimator(algorithm)
    model.fit(prepare_features(labeled, features), labeled["rul"].to_numpy())
    artifact = {
        "model": model,
        "algorithm": algorithm,
        "subset": subset,
        "feature_columns": features,
        "clip_threshold": clip_threshold,
        "rolling_window": rolling_window,
        "include_differences": include_differences,
        "data_source": data_source,
    }
    destination = Path(model_dir) / f"{subset}_{algorithm.lower().replace(' ', '_')}.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return model, destination


def evaluate_official_test(
    model: Any,
    test_frame: pd.DataFrame,
    reference_rul: pd.Series,
    feature_columns: list[str],
    rolling_window: int | None = None,
    include_differences: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate final-cycle test observations against NASA's official RUL labels."""
    engines = sorted(test_frame["engine_id"].unique())
    labels = reference_rul.reset_index(drop=True)
    if len(engines) != len(labels):
        raise ValueError(
            f"Test data has {len(engines)} engines but the RUL file has {len(labels)} labels."
        )
    engineered = add_temporal_features(test_frame, rolling_window, include_differences)
    final_rows = (
        engineered.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True)
        .tail(1)
        .sort_values("engine_id")
    )
    actual = labels.to_numpy(dtype=float)
    predicted = np.maximum(
        0, model.predict(prepare_features(final_rows, feature_columns))
    )
    return regression_metrics(actual, predicted), make_evaluation_frame(
        final_rows, actual, predicted
    )
