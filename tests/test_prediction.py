import numpy as np
import pandas as pd
import pytest

from src.predict_rul import load_model, predict_rul
from src.preprocessing import add_temporal_features, get_feature_columns
from src.train_model import _engine_split, fit_and_save_model


class FakeModel:
    def predict(self, features):
        return np.full(len(features), 12.5)


def test_prediction_returns_one_nonnegative_estimate_per_observation():
    frame = pd.DataFrame({"setting_1": [0.2], "sensor_1": [3.4]})
    result = predict_rul(FakeModel(), frame, ["setting_1", "sensor_1"])
    assert result.tolist() == [12.5]


def test_prediction_rejects_invalid_model_output():
    class InvalidModel:
        def predict(self, features):
            return np.array([np.nan])

    frame = pd.DataFrame({"sensor_1": [3.4]})
    with pytest.raises(ValueError, match="invalid"):
        predict_rul(InvalidModel(), frame, ["sensor_1"])


def test_engine_split_keeps_engine_ids_disjoint():
    frame = pd.DataFrame(
        {"engine_id": [1, 1, 2, 2, 3, 3, 4, 4], "cycle": range(1, 9)}
    )
    train_index, validation_index = _engine_split(frame)
    train_engines = set(frame.iloc[train_index]["engine_id"])
    validation_engines = set(frame.iloc[validation_index]["engine_id"])
    assert train_engines.isdisjoint(validation_engines)


def test_engine_split_requires_multiple_engines():
    frame = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 2]})
    with pytest.raises(ValueError, match="two distinct engines"):
        _engine_split(frame)


def test_saved_model_reloads_and_predicts_consistently(tmp_path):
    rows = []
    for engine_id in range(1, 5):
        for cycle in range(1, 5):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "setting_1": float(engine_id) / 10,
                    "setting_2": 0.1,
                    "setting_3": 100.0,
                    "sensor_1": float(engine_id + cycle),
                }
            )
    train_frame = pd.DataFrame(rows)
    model, path = fit_and_save_model(
        train_frame,
        "Linear Regression",
        "FD001",
        tmp_path,
        clip_threshold=None,
        data_source="synthetic-demo",
    )
    artifact = load_model(path)
    assert artifact["data_source"] == "synthetic-demo"
    observation = train_frame.iloc[[0]]
    expected = predict_rul(model, observation, artifact["feature_columns"])
    actual = predict_rul(
        artifact["model"], observation, artifact["feature_columns"]
    )
    assert actual.tolist() == expected.tolist()


def test_training_and_prediction_support_past_only_temporal_features(tmp_path):
    rows = []
    for engine_id in range(1, 5):
        for cycle in range(1, 5):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "setting_1": float(engine_id) / 10,
                    "setting_2": 0.1,
                    "setting_3": 100.0,
                    "sensor_1": float(engine_id + cycle),
                }
            )
    train_frame = pd.DataFrame(rows)
    window = 2
    differences = True
    features = get_feature_columns(train_frame, window, differences)
    model, path = fit_and_save_model(
        train_frame,
        "Linear Regression",
        "FD001",
        tmp_path,
        feature_columns=features,
        rolling_window=window,
        include_differences=differences,
    )
    artifact = load_model(path)
    history = train_frame.loc[
        (train_frame["engine_id"] == 1) & (train_frame["cycle"] <= 3)
    ]
    engineered = add_temporal_features(history, window, differences)
    prediction = predict_rul(
        artifact["model"],
        engineered.tail(1),
        artifact["feature_columns"],
    )
    assert prediction.shape == (1,)
    assert prediction[0] >= 0
    assert artifact["rolling_window"] == window
