from __future__ import annotations

import pandas as pd
import plotly.express as px


def sensor_trend_figure(frame: pd.DataFrame, engine_id: int, sensors: list[str]):
    engine = frame.loc[frame["engine_id"] == engine_id]
    long = engine.melt(
        id_vars=["cycle"], value_vars=sensors, var_name="sensor", value_name="measurement"
    )
    return px.line(
        long,
        x="cycle",
        y="measurement",
        color="sensor",
        title=f"Engine {engine_id}: sensor trend by observed cycle",
    )


def operating_condition_figure(frame: pd.DataFrame):
    long = frame.melt(
        id_vars=["cycle"],
        value_vars=["setting_1", "setting_2", "setting_3"],
        var_name="operating setting",
        value_name="value",
    )
    return px.scatter(
        long,
        x="cycle",
        y="value",
        color="operating setting",
        title="Observed operating settings by cycle",
        opacity=0.55,
    )


def prediction_scatter_figure(evaluation: pd.DataFrame):
    return px.scatter(
        evaluation,
        x="actual_rul",
        y="predicted_rul",
        color="engine_id",
        title="Actual versus predicted RUL (cycles)",
        labels={"actual_rul": "Reference RUL (cycles)", "predicted_rul": "Predicted RUL (cycles)"},
    )
