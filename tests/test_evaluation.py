import numpy as np
import pandas as pd

from src.train_model import evaluate_official_test


class FixedModel:
    def predict(self, features):
        return np.full(len(features), 7.0)


def test_official_test_evaluation_uses_final_row_and_sorted_engine_labels():
    test_frame = pd.DataFrame(
        {
            "engine_id": [2, 1, 2, 1],
            "cycle": [5, 2, 6, 3],
            "setting_1": [0.1] * 4,
            "setting_2": [0.2] * 4,
            "setting_3": [0.3] * 4,
            "sensor_1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    metrics, predictions = evaluate_official_test(
        FixedModel(),
        test_frame,
        pd.Series([5, 8]),
        ["setting_1", "setting_2", "setting_3", "sensor_1"],
    )
    assert predictions["engine_id"].tolist() == [1, 2]
    assert predictions["cycle"].tolist() == [3, 6]
    assert predictions["actual_rul"].tolist() == [5, 8]
    assert predictions["predicted_rul"].tolist() == [7, 7]
    assert metrics["MAE"] == 1.5
