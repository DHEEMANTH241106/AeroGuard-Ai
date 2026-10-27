from __future__ import annotations

import pandas as pd


def add_training_rul(frame: pd.DataFrame, clip_threshold: int | None = None) -> pd.DataFrame:
    """Label each training observation as its engine's final cycle minus current cycle."""
    if clip_threshold is not None and clip_threshold < 1:
        raise ValueError("RUL clipping threshold must be a positive integer or None.")
    result = frame.copy()
    max_cycles = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = max_cycles - result["cycle"]
    if clip_threshold is not None:
        result["rul"] = result["rul"].clip(upper=clip_threshold)
    return result


def add_test_rul(frame: pd.DataFrame, final_cycle_rul: pd.Series) -> pd.DataFrame:
    """Align official test labels by engine ID and propagate them to each observed cycle."""
    engines = sorted(frame["engine_id"].unique())
    labels = final_cycle_rul.reset_index(drop=True)
    if len(engines) != len(labels):
        raise ValueError(
            f"Test data has {len(engines)} engines but the RUL file has {len(labels)} labels."
        )
    final_rul = dict(zip(engines, labels, strict=True))
    result = frame.copy()
    last_cycles = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = [
        final_rul[engine] + last_cycle - cycle
        for engine, last_cycle, cycle in zip(
            result["engine_id"], last_cycles, result["cycle"], strict=True
        )
    ]
    return result
