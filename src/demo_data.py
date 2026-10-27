from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import NASA_COLUMNS


def _make_observations(
    engine_ids: range,
    lengths: list[int],
    post_observation_rul: list[int] | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[list[float | int]] = []
    sensor_baselines = rng.uniform(5.0, 100.0, size=21)
    sensor_slopes = rng.uniform(-0.15, 0.25, size=21)
    sensor_slopes[[0, 1, 5, 10, 15, 20]] += 0.55

    for engine_id, length, future_rul in zip(
        engine_ids,
        lengths,
        post_observation_rul or [0] * len(lengths),
        strict=True,
    ):
        operating_life = length + future_rul
        engine_offset = rng.normal(0.0, 0.3, size=21)
        settings = [
            0.1 + (engine_id % 3) * 0.005,
            0.2 + (engine_id % 2) * 0.01,
            100.0 + (engine_id % 4) * 0.5,
        ]
        for cycle in range(1, length + 1):
            degradation = cycle / operating_life
            measurements = (
                sensor_baselines
                + engine_offset
                + sensor_slopes * degradation * 30
                + rng.normal(0.0, 0.12, size=21)
            )
            rows.append(
                [
                    engine_id,
                    cycle,
                    *settings,
                    *measurements.tolist(),
                ]
            )
    return pd.DataFrame(rows, columns=NASA_COLUMNS)


def make_synthetic_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate explicitly synthetic, reproducible C-MAPSS-shaped demo trajectories."""
    train_lengths = [int(value) for value in np.linspace(38, 66, 24)]
    train = _make_observations(range(1, 25), train_lengths, None, seed=4201)

    test_lengths = [42, 45, 48, 51, 54, 57]
    reference = pd.Series([8, 12, 16, 20, 24, 28], name="rul", dtype="int64")
    test = _make_observations(
        range(1, 7),
        test_lengths,
        reference.tolist(),
        seed=4202,
    )
    return train, test, reference
