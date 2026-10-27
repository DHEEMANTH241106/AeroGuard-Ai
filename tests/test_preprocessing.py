import pandas as pd
import pytest

from src.data_loader import DatasetValidationError, NASA_COLUMNS, load_dataset
from src.preprocessing import add_temporal_features, get_feature_columns, prepare_features


def sample_frame():
    return pd.DataFrame(
        {
            "engine_id": [2, 1],
            "cycle": [2, 1],
            "setting_1": [0.2, 0.1],
            "setting_2": [0.0, 0.0],
            "setting_3": [100.0, 100.0],
            "sensor_1": [3.0, 2.0],
            "sensor_2": [4.0, 3.0],
        }
    )


def test_validation_sorts_by_engine_and_cycle():
    result = load_dataset(sample_frame().to_csv(index=False).encode(), "data.csv")
    assert list(zip(result["engine_id"], result["cycle"])) == [(1, 1), (2, 2)]


def test_validation_rejects_duplicate_engine_cycle():
    frame = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(DatasetValidationError, match="duplicate"):
        load_dataset(frame.to_csv(index=False).encode(), "data.csv")


def test_validation_rejects_missing_values():
    frame = sample_frame()
    frame.loc[0, "sensor_1"] = float("nan")
    with pytest.raises(DatasetValidationError, match="missing"):
        load_dataset(frame.to_csv(index=False).encode(), "data.csv")


def test_validation_assigns_nasa_column_names_to_headerless_csv():
    rows = [[1, 1, 0.1, 0.2, 0.3, *range(1, 22)]]
    text = pd.DataFrame(rows).to_csv(index=False, header=False)
    result = load_dataset(text.encode(), "data.csv")
    assert list(result.columns) == NASA_COLUMNS
    assert result.iloc[0]["sensor_21"] == 21


def test_feature_order_is_settings_then_numbered_sensors():
    frame = sample_frame()
    assert get_feature_columns(frame) == [
        "setting_1", "setting_2", "setting_3", "sensor_1", "sensor_2"
    ]


def test_prediction_feature_order_and_missing_validation():
    frame = sample_frame()
    assert prepare_features(frame, ["sensor_2", "setting_1"]).columns.tolist() == [
        "sensor_2", "setting_1"
    ]
    with pytest.raises(ValueError, match="missing"):
        prepare_features(frame, ["sensor_9"])


def test_temporal_features_use_only_current_and_prior_engine_cycles():
    frame = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2],
            "cycle": [2, 1, 3, 1],
            "sensor_1": [5.0, 1.0, 9.0, 20.0],
        }
    )
    result = add_temporal_features(
        frame, rolling_window=2, include_differences=True
    ).set_index(["engine_id", "cycle"])
    assert result.loc[(1, 1), "sensor_1_roll_mean_2"] == 1
    assert result.loc[(1, 2), "sensor_1_roll_mean_2"] == 3
    assert result.loc[(1, 2), "sensor_1_roll_std_2"] == 2
    assert result.loc[(1, 2), "sensor_1_diff"] == 4
    assert result.loc[(2, 1), "sensor_1_roll_mean_2"] == 20
    assert result.loc[(2, 1), "sensor_1_diff"] == 0


def test_temporal_features_require_valid_window():
    with pytest.raises(ValueError, match="at least 2"):
        add_temporal_features(sample_frame(), rolling_window=1)


def test_temporal_feature_names_are_included_in_model_feature_choices():
    frame = sample_frame()
    options = get_feature_columns(frame, rolling_window=5, include_differences=True)
    engineered = add_temporal_features(
        frame, rolling_window=5, include_differences=True
    )
    columns = get_feature_columns(
        engineered, rolling_window=5, include_differences=True
    )
    assert columns == options
    assert "sensor_1_roll_mean_5" in columns
    assert "sensor_1_roll_std_5" in columns
    assert "sensor_1_diff" in columns


def test_empty_feature_selection_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        prepare_features(sample_frame(), [])
