import pandas as pd
import pytest

from src.rul_calculator import add_test_rul, add_training_rul


def test_training_rul_is_max_cycle_minus_current_cycle():
    frame = pd.DataFrame({"engine_id": [1, 1, 1, 2, 2], "cycle": [100, 150, 200, 1, 5]})
    result = add_training_rul(frame)
    assert result["rul"].tolist() == [100, 50, 0, 4, 0]


def test_training_rul_optional_clipping():
    frame = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 200]})
    assert add_training_rul(frame, clip_threshold=125)["rul"].tolist() == [125, 0]


def test_training_rul_rejects_nonpositive_clip():
    frame = pd.DataFrame({"engine_id": [1], "cycle": [1]})
    with pytest.raises(ValueError, match="positive"):
        add_training_rul(frame, clip_threshold=0)


def test_test_rul_aligns_by_sorted_engine_id_and_cycle_offset():
    frame = pd.DataFrame({"engine_id": [2, 1, 2, 1], "cycle": [4, 2, 5, 3]})
    result = add_test_rul(frame, pd.Series([7, 10]))
    assert result["rul"].tolist() == [11, 8, 10, 7]


def test_test_rul_rejects_reference_count_mismatch():
    frame = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 2]})
    with pytest.raises(ValueError, match="labels"):
        add_test_rul(frame, pd.Series([1, 2]))
