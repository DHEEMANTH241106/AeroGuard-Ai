from pandas.testing import assert_frame_equal

from src.demo_data import make_synthetic_demo_data


def test_synthetic_demo_data_is_explicitly_cmapss_shaped_and_reproducible():
    train_a, test_a, rul_a = make_synthetic_demo_data()
    train_b, test_b, rul_b = make_synthetic_demo_data()

    assert_frame_equal(train_a, train_b)
    assert_frame_equal(test_a, test_b)
    assert rul_a.equals(rul_b)
    assert train_a["engine_id"].nunique() == 24
    assert test_a["engine_id"].nunique() == len(rul_a) == 6
    assert len([column for column in train_a if column.startswith("sensor_")]) == 21
    assert train_a["cycle"].min() == 1
