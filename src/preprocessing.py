from __future__ import annotations

import re

import pandas as pd


def get_feature_columns(
    frame: pd.DataFrame,
    rolling_window: int | None = None,
    include_differences: bool = False,
) -> list[str]:
    """Return settings and sensor features in deterministic order."""
    settings = [f"setting_{number}" for number in range(1, 4)]
    sensors = sorted(
        [
            column
            for column in frame.columns
            if re.fullmatch(r"sensor_\d+", column)
        ],
        key=lambda column: int(column.split("_", 1)[1]),
    )
    missing = [column for column in settings if column not in frame.columns]
    if missing or not sensors:
        raise ValueError("Features require setting_1..setting_3 and at least one sensor_N column.")
    columns = settings + sensors
    if rolling_window is not None:
        if rolling_window < 2:
            raise ValueError("Rolling feature window must be at least 2 cycles.")
        for sensor in sensors:
            columns.extend([f"{sensor}_roll_mean_{rolling_window}", f"{sensor}_roll_std_{rolling_window}"])
    if include_differences:
        columns.extend(f"{sensor}_diff" for sensor in sensors)
    return columns


def add_temporal_features(
    frame: pd.DataFrame,
    rolling_window: int | None = None,
    include_differences: bool = False,
) -> pd.DataFrame:
    """Create within-engine rolling/difference features from current and prior cycles only."""
    result = frame.sort_values(["engine_id", "cycle"]).copy()
    sensors = sorted(
        [
            column
            for column in result.columns
            if re.fullmatch(r"sensor_\d+", column)
        ],
        key=lambda column: int(column.split("_", 1)[1]),
    )
    grouped = result.groupby("engine_id", sort=False)
    if rolling_window is not None:
        if rolling_window < 2:
            raise ValueError("Rolling feature window must be at least 2 cycles.")
        for sensor in sensors:
            rolling = grouped[sensor].rolling(rolling_window, min_periods=1)
            result[f"{sensor}_roll_mean_{rolling_window}"] = (
                rolling.mean().reset_index(level=0, drop=True).sort_index()
            )
            result[f"{sensor}_roll_std_{rolling_window}"] = (
                rolling.std(ddof=0).fillna(0).reset_index(level=0, drop=True).sort_index()
            )
    if include_differences:
        for sensor in sensors:
            result[f"{sensor}_diff"] = grouped[sensor].diff().fillna(0)
    return result.sort_index()


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Select and validate the same ordered features used during model fitting."""
    columns = (
        feature_columns
        if feature_columns is not None
        else get_feature_columns(frame)
    )
    if not columns:
        raise ValueError("Select at least one model feature.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing model feature(s): {', '.join(missing)}.")
    features = frame.loc[:, columns].copy()
    if features.isna().any().any():
        raise ValueError("Prediction features contain missing values.")
    if not features.apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError("Prediction features must contain numeric values only.")
    return features.astype(float)
