from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import (
    DatasetValidationError,
    find_local_subset,
    load_dataset,
    load_rul_reference,
)
from src.demo_data import make_synthetic_demo_data
from src.evaluate_model import model_feature_importance
from src.predict_rul import load_model, predict_rul
from src.preprocessing import add_temporal_features, get_feature_columns
from src.rul_calculator import add_training_rul
from src.train_model import (
    ALGORITHMS,
    compare_models,
    evaluate_official_test,
    fit_and_save_model,
)
from src.visualization import (
    operating_condition_figure,
    prediction_scatter_figure,
    sensor_trend_figure,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

st.set_page_config(
    page_title="AeroGuard AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f3f7fa; }
    [data-testid="stSidebar"] { background: #071b33; }
    [data-testid="stSidebar"] * { color: #eef7ff; }
    .hero {
        padding: 1.6rem 2rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg, #071b33, #0d4260);
        border-bottom: 4px solid #20c9c3;
    }
    .hero h1 { margin: 0; letter-spacing: .08em; }
    .notice {
        padding: .8rem 1rem; border-left: 4px solid #e34b59;
        background: #fff1f2; border-radius: 6px; color: #5b1720;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    for key, value in {
        "subset": "FD001",
        "navigation": "Overview",
        "next_navigation": None,
        "loaded_subset": None,
        "data_source": None,
        "train_frame": None,
        "test_frame": None,
        "test_rul": None,
        "model_artifact": None,
        "model_path": None,
        "comparison": None,
        "subset_comparison": None,
        "validation_predictions": None,
        "official_metrics": None,
        "official_predictions": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_local(role: str, subset: str) -> pd.DataFrame | None:
    path = find_local_subset(DATA_DIR, subset, role)
    if path is None:
        return None
    return load_dataset(path)


def _load_model_files(subset: str) -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob(f"{subset}_*.joblib"))


def _activate_demo_data(subset: str) -> None:
    train, test, reference = make_synthetic_demo_data()
    st.session_state.train_frame = train
    st.session_state.test_frame = test
    st.session_state.test_rul = reference
    st.session_state.loaded_subset = subset
    st.session_state.data_source = "synthetic-demo"
    st.session_state.model_artifact = None
    st.session_state.model_path = None
    st.session_state.comparison = None
    st.session_state.subset_comparison = None
    st.session_state.validation_predictions = None
    st.session_state.official_metrics = None
    st.session_state.official_predictions = None


def _header() -> None:
    st.markdown(
        '<div class="hero"><h1>✈ AEROGUARD AI</h1>'
        "<div>Aircraft Engine Remaining Useful Life Prediction</div></div>",
        unsafe_allow_html=True,
    )
    model_is_demo = (
        st.session_state.model_artifact is not None
        and st.session_state.model_artifact.get("data_source") == "synthetic-demo"
    )
    if st.session_state.data_source == "synthetic-demo":
        st.warning(
            "DEMO MODE — synthetic sample trajectories and synthetic reference labels are active. "
            "All model metrics and predictions are illustrative, not NASA dataset results."
        )
    elif model_is_demo:
        st.warning(
            "DEMO MODEL — this model was trained on synthetic sample trajectories. "
            "Its estimates and metrics are illustrative, not NASA dataset results."
        )
    elif st.session_state.data_source:
        st.caption(f"Active data source: {st.session_state.data_source}")
    st.markdown(
        '<div class="notice"><b>Academic prototype only.</b> '
        "Predictions are experimental and are not approved for real-world aviation "
        "safety or maintenance decisions. RUL is an estimate, not a safety clearance "
        "or probability of failure.</div>",
        unsafe_allow_html=True,
    )


def _dataset_page() -> None:
    st.header("Dataset Explorer")
    subset = st.selectbox("C-MAPSS subset", SUBSETS, key="subset")
    st.markdown(
        "**No NASA data yet?** Load an explicitly synthetic demonstration dataset to exercise "
        "the training, prediction, monitoring, and chart controls. Its results are illustrative only."
    )
    if st.button("Load synthetic demo data", type="secondary"):
        _activate_demo_data(subset)
        st.success(
            "Synthetic demo data loaded. It is not NASA data and should never be used for aviation decisions."
        )
        st.rerun()

    train_upload = st.file_uploader(
        "Training trajectory file (compatible CSV or NASA TXT)",
        type=["csv", "txt"],
        key="train_upload",
    )
    test_upload = st.file_uploader(
        "Optional test trajectory file", type=["csv", "txt"], key="test_upload"
    )
    rul_upload = st.file_uploader(
        "Optional official RUL reference file", type=["txt", "csv"], key="rul_upload"
    )

    if st.button("Load selected data", type="primary"):
        try:
            train = (
                load_dataset(train_upload, train_upload.name)
                if train_upload is not None
                else _load_local("train", subset)
            )
            test = (
                load_dataset(test_upload, test_upload.name)
                if test_upload is not None
                else _load_local("test", subset)
                if train_upload is None
                else None
            )
            reference = (
                load_rul_reference(rul_upload)
                if rul_upload is not None
                else load_rul_reference(DATA_DIR / f"RUL_{subset}.txt")
                if (
                    train_upload is None
                    and test_upload is None
                    and (DATA_DIR / f"RUL_{subset}.txt").is_file()
                )
                else None
            )
            if train is None:
                st.session_state.train_frame = None
                st.session_state.loaded_subset = None
                st.session_state.data_source = None
                st.error(
                    f"No training file is available for {subset}. Add "
                    f"`train_{subset}.txt` to `data/raw/` or upload a compatible file. "
                    "No synthetic training data is substituted."
                )
            else:
                st.session_state.train_frame = train
                st.session_state.loaded_subset = subset
                st.session_state.data_source = (
                    "NASA C-MAPSS " + subset
                    if train_upload is None
                    else "user-uploaded dataset"
                )
                st.success(f"Loaded {subset} training data.")
            st.session_state.test_frame = test
            st.session_state.test_rul = reference
            st.session_state.model_artifact = None
            st.session_state.model_path = None
            st.session_state.comparison = None
            st.session_state.subset_comparison = None
            st.session_state.validation_predictions = None
            st.session_state.official_metrics = None
            st.session_state.official_predictions = None
            if test is None:
                st.info("Optional test trajectories are not available; training and validation still work.")
            if test is not None and reference is None:
                st.info("Test trajectories loaded without reference RUL labels; official test evaluation is unavailable.")
        except (DatasetValidationError, ValueError, OSError) as exc:
            st.error(str(exc))

    train_frame = st.session_state.train_frame
    test_frame = st.session_state.test_frame
    if train_frame is None:
        st.info("Load real NASA C-MAPSS data above. Dataset files are not included with this prototype.")
        st.code(
            "data/raw/train_FD001.txt\n"
            "data/raw/test_FD001.txt\n"
            "data/raw/RUL_FD001.txt",
            language="text",
        )
        return
    if st.session_state.loaded_subset != subset:
        st.warning("The selected subset differs from the loaded data. Load the selected subset before training.")
        return

    if st.session_state.data_source == "synthetic-demo":
        st.info(
            "Synthetic-only demonstration data. This subset selector is for display; "
            "the generated observations are not an FD001–FD004 NASA subset."
        )
    st.subheader("Dataset summary")
    total_cycles = train_frame.groupby("engine_id")["cycle"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training observations", f"{len(train_frame):,}")
    c2.metric("Training engines", f"{train_frame['engine_id'].nunique():,}")
    c3.metric("Observed cycles", f"{int(train_frame['cycle'].nunique()):,}")
    c4.metric("Sensors", len([name for name in train_frame if name.startswith("sensor_")]))
    st.caption(
        f"Per-engine observed cycle counts range from {int(total_cycles.min())} "
        f"to {int(total_cycles.max())}. Operating settings are dataset measurements, "
        "not domestic/international route labels."
    )
    tabs = st.tabs(["Raw training data", "Processed data", "Sensor statistics", "Operating conditions"])
    with tabs[0]:
        st.dataframe(train_frame.head(100), use_container_width=True)
    with tabs[1]:
        processed = add_training_rul(train_frame)
        st.dataframe(processed.head(100), use_container_width=True)
        st.caption("Training RUL is max observed cycle for that engine minus the current cycle.")
    with tabs[2]:
        sensors = [name for name in train_frame if name.startswith("sensor_")]
        stats = train_frame[sensors].describe().T
        stats["missing"] = train_frame[sensors].isna().sum()
        stats["constant"] = train_frame[sensors].nunique() <= 1
        stats["near_constant"] = (stats["std"] <= 1e-8) & ~stats["constant"]
        st.dataframe(stats, use_container_width=True)
        st.caption(
            "Constant sensors and sensors with standard deviation ≤ 1e-8 are flagged, not automatically removed. "
            "Choose features explicitly during training."
        )
    with tabs[3]:
        st.plotly_chart(operating_condition_figure(train_frame), use_container_width=True)

    if test_frame is not None:
        st.subheader("Optional test data")
        st.write(
            f"{len(test_frame):,} observations across "
            f"{test_frame['engine_id'].nunique():,} engines."
        )
        if st.session_state.test_rul is not None:
            if st.session_state.data_source == "synthetic-demo":
                st.caption(
                    f"Loaded {len(st.session_state.test_rul)} synthetic demo reference labels; "
                    "these are not NASA RUL labels."
                )
            else:
                st.caption(
                    f"Loaded {len(st.session_state.test_rul)} official reference RUL labels."
                )


def _training_page() -> None:
    st.header("Model Training")
    train_frame = st.session_state.train_frame
    if train_frame is None:
        st.warning("Load a real training dataset in Dataset Explorer before training.")
        return
    if st.session_state.loaded_subset != st.session_state.subset:
        st.warning("The selected subset differs from the loaded data. Load the selected subset before training.")
        return
    algorithm = st.selectbox("Selected model to save", ALGORITHMS, index=0)
    clip = st.number_input(
        "RUL clipping threshold (cycles; 0 disables clipping)",
        min_value=0,
        max_value=1000,
        value=125,
        help="Clipping caps large training targets; this changes the learning target, not the cycle measurements.",
    )
    rolling_window = st.selectbox(
        "Past-only rolling features",
        [None, 5, 10, 20],
        format_func=lambda value: "Disabled" if value is None else f"{value} cycles",
        help="Rolling mean and standard deviation include the current and preceding cycles within the same engine.",
    )
    include_differences = st.checkbox(
        "Add differences from the previous observed cycle",
        help="The first cycle for each engine has a difference of zero.",
    )
    available_features = get_feature_columns(
        train_frame, rolling_window, include_differences
    )
    selected_features = st.multiselect(
        "Model features",
        available_features,
        default=available_features,
        help="Constant sensors remain available; choose model inputs explicitly.",
    )
    selected_algorithms = st.multiselect(
        "Algorithms to compare on held-out engines",
        ALGORITHMS,
        default=ALGORITHMS,
    )
    if not selected_algorithms:
        st.warning("Choose at least one comparison algorithm.")
    if not selected_features:
        st.warning("Choose at least one model feature.")
    if st.button(
        "Compare, train and save",
        type="primary",
        disabled=not selected_algorithms or not selected_features,
    ):
        threshold = int(clip) or None
        try:
            progress = st.progress(0, text="Comparing models on held-out engine IDs…")
            comparison, predictions = compare_models(
                train_frame,
                selected_algorithms,
                threshold,
                selected_features,
                rolling_window,
                include_differences,
            )
            progress.progress(55, text="Fitting selected model on all training engines…")
            model, path = fit_and_save_model(
                train_frame,
                algorithm,
                st.session_state.subset,
                MODEL_DIR,
                threshold,
                selected_features,
                rolling_window,
                include_differences,
                data_source=st.session_state.data_source,
            )
            features = selected_features
            st.session_state.comparison = comparison
            st.session_state.validation_predictions = predictions
            st.session_state.model_artifact = {
                "model": model,
                "algorithm": algorithm,
                "subset": st.session_state.subset,
                "data_source": st.session_state.data_source,
                "feature_columns": features,
                "clip_threshold": threshold,
                "rolling_window": rolling_window,
                "include_differences": include_differences,
            }
            st.session_state.model_path = path
            evaluation_report = {
                "subset": st.session_state.subset,
                "data_source": st.session_state.data_source,
                "results_are_synthetic_demo_only": synthetic_demo,
                "selected_algorithm": algorithm,
                "validation_method": "GroupShuffleSplit by engine_id; random_state=42",
                "clip_threshold_cycles": threshold,
                "rolling_window": rolling_window,
                "include_differences": include_differences,
                "feature_columns": features,
                "validation_metrics": comparison.to_dict(orient="records"),
                "official_test_metrics": None,
                "test_metrics": None,
            }
            progress.progress(100, text="Training and save complete.")
            if synthetic_demo := st.session_state.data_source == "synthetic-demo":
                fit_source = "synthetic demonstration data"
            elif st.session_state.data_source == "user-uploaded dataset":
                fit_source = "user-uploaded data"
            else:
                fit_source = f"NASA C-MAPSS {st.session_state.subset} data"
            st.success(
                f"{algorithm} fit using {fit_source} and saved to "
                f"`{path.relative_to(ROOT)}`."
            )

            test = st.session_state.test_frame
            reference = st.session_state.test_rul
            if test is not None and reference is not None:
                metrics, evaluation = evaluate_official_test(
                    model,
                    test,
                    reference,
                    features,
                    rolling_window,
                    include_differences,
                )
                st.session_state.official_metrics = metrics
                st.session_state.official_predictions = evaluation
                evaluation_report["test_metrics"] = metrics
                if synthetic_demo:
                    st.info(
                        "Synthetic demonstration holdout metrics use generated reference labels; "
                        "they are not official NASA test results."
                    )
                else:
                    evaluation_report["official_test_metrics"] = metrics
                    st.info("Official test metrics use only the last observed cycle for each test engine.")
            report_path = MODEL_DIR / f"evaluation_{st.session_state.subset}.json"
            report_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
            st.caption(f"Evaluation report saved to `{report_path.relative_to(ROOT)}`.")
        except (ImportError, ValueError, DatasetValidationError, OSError) as exc:
            st.error(str(exc))

    comparison = st.session_state.comparison
    if comparison is not None:
        st.subheader("Held-out-engine validation results")
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.caption(
            "Engines are split as groups; observations from a validation engine are never used to fit that validation model. "
            "Metrics are in cycles. MAE/RMSE lower is better; R² measures explained variance."
        )
    if st.session_state.official_metrics is not None:
        label = (
            "Synthetic demonstration holdout evaluation"
            if st.session_state.data_source == "synthetic-demo"
            else "Official NASA test evaluation"
        )
        st.subheader(label)
        st.json(st.session_state.official_metrics)


def _restore_model() -> dict | None:
    artifact = st.session_state.model_artifact
    if artifact is not None:
        if artifact.get("subset") == st.session_state.subset:
            return artifact
        st.session_state.model_artifact = None
    model_files = _load_model_files(st.session_state.subset)
    if not model_files:
        return None
    with st.sidebar:
        selected = st.selectbox(
            "Saved model artifact",
            model_files,
            format_func=lambda path: path.name,
            key="saved_model_picker",
        )
        if st.button("Load saved model"):
            try:
                st.session_state.model_artifact = load_model(selected)
                st.session_state.model_path = selected
                st.rerun()
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Could not load model artifact: {exc}")
    artifact = st.session_state.model_artifact
    if artifact is not None and artifact.get("subset") == st.session_state.subset:
        return artifact
    return None


def _prediction_page() -> None:
    st.header("RUL Prediction")
    model_artifact = _restore_model()
    if model_artifact is None:
        st.warning("Train a model or load a saved artifact from the sidebar first.")
        return
    st.caption(
        f"Model: {model_artifact['algorithm']} · Dataset subset: {model_artifact['subset']} · "
        f"Features: {len(model_artifact['feature_columns'])}"
    )
    if model_artifact.get("data_source") == "synthetic-demo":
        st.caption("This model was fit on synthetic demo trajectories; its output is illustrative only.")
    frame = st.session_state.test_frame
    uploaded = st.file_uploader(
        "Or upload a compatible historical sensor sequence",
        type=["csv", "txt"],
        key="prediction_upload",
    )
    if uploaded is not None:
        try:
            frame = load_dataset(uploaded, uploaded.name)
        except DatasetValidationError as exc:
            st.error(str(exc))
            return
    elif frame is None:
        st.info("Load test trajectories or upload a compatible CSV/TXT sequence.")
        return

    engines = sorted(frame["engine_id"].unique())
    engine_id = st.selectbox("Engine ID", engines)
    engine = frame.loc[frame["engine_id"] == engine_id].sort_values("cycle")
    cycle = st.select_slider(
        "Observed cycle (prediction uses this cycle only)",
        options=engine["cycle"].astype(int).tolist(),
    )
    history = engine.loc[engine["cycle"] <= cycle]
    engineered_history = add_temporal_features(
        history,
        model_artifact.get("rolling_window"),
        model_artifact.get("include_differences", False),
    )
    observation = engineered_history.loc[engineered_history["cycle"] == cycle].tail(1)
    try:
        predicted = float(
            predict_rul(
                model_artifact["model"],
                observation,
                model_artifact["feature_columns"],
            )[0]
        )
    except (ValueError, KeyError) as exc:
        st.error(f"Prediction could not be calculated: {exc}")
        return
    st.metric("Predicted Remaining Useful Life", f"{predicted:.1f} cycles")
    st.write(
        f"The model estimates approximately **{predicted:.1f} additional operating cycles** "
        "before reaching the end-of-useful-life condition represented by the training data. "
        "This is an experimental estimate, not a guarantee or safety determination."
    )
    with st.expander("Sensor and operating-setting values at this observation"):
        st.dataframe(observation, hide_index=True, use_container_width=True)
    sensors = [name for name in engine if name.startswith("sensor_")]
    st.plotly_chart(sensor_trend_figure(history, engine_id, sensors), use_container_width=True)
    if "rul" in observation.columns:
        st.caption("Reference labels are not included in prediction inputs.")


def _monitoring_page() -> None:
    st.header("Engine Monitoring")
    artifact = _restore_model()
    if artifact is None:
        st.warning("Train or load a model before generating monitoring estimates.")
        return
    frame = st.session_state.test_frame
    if frame is None:
        st.info("Load test trajectories or upload historical engine data in Dataset Explorer.")
        return
    increased_at = st.number_input("Increased monitoring below (cycles)", min_value=1, value=50)
    review_at = st.number_input("Maintenance review suggested below (cycles)", min_value=1, value=20)
    if review_at >= increased_at:
        st.error("Review threshold must be lower than the increased-monitoring threshold.")
        return
    engineered = add_temporal_features(
        frame,
        artifact.get("rolling_window"),
        artifact.get("include_differences", False),
    )
    latest = engineered.sort_values(["engine_id", "cycle"]).groupby("engine_id", sort=True).tail(1).copy()
    try:
        latest["predicted_rul"] = predict_rul(
            artifact["model"], latest, artifact["feature_columns"]
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    latest["monitoring_category"] = np.select(
        [latest["predicted_rul"] <= review_at, latest["predicted_rul"] <= increased_at],
        ["Maintenance review suggested", "Increased monitoring"],
        default="Normal monitoring",
    )
    latest["suggested_follow_up"] = latest["monitoring_category"].map(
        {
            "Maintenance review suggested": "Review the estimate with qualified personnel.",
            "Increased monitoring": "Observe trends in this academic demonstration.",
            "Normal monitoring": "Continue demonstration monitoring.",
        }
    )
    category = st.multiselect(
        "Filter categories",
        latest["monitoring_category"].unique().tolist(),
        default=latest["monitoring_category"].unique().tolist(),
    )
    sort_order = st.selectbox("Sort by", ["Lowest predicted RUL first", "Engine ID"])
    if sort_order == "Lowest predicted RUL first":
        latest = latest.sort_values("predicted_rul")
    else:
        latest = latest.sort_values("engine_id")
    visible = latest.loc[latest["monitoring_category"].isin(category)]
    st.dataframe(
        visible[
            [
                "engine_id",
                "cycle",
                "predicted_rul",
                "monitoring_category",
                "suggested_follow_up",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "These categories are configurable demonstration settings, not official aircraft maintenance limits. "
        "RUL is not a probability of failure or aviation safety clearance."
    )


def _comparison_page() -> None:
    st.header("Model Comparison")
    comparison = st.session_state.comparison
    predictions = st.session_state.validation_predictions
    if st.session_state.loaded_subset != st.session_state.subset:
        st.info("Load the currently selected subset before viewing its model comparison.")
        comparison = None
        predictions = None
    if comparison is None or predictions is None:
        st.info("Run model comparison on the Model Training page to generate real metrics.")
    else:
        best = comparison.iloc[0]
        st.metric(
            "Lowest validation MAE",
            f"{best['MAE']:.2f} cycles",
            help=f"{best['Algorithm']} ranked first by held-out-engine MAE; this is a validation result, not a safety conclusion.",
        )
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        names = list(predictions)
        selected = st.selectbox("Validation predictions", names)
        evaluation = predictions[selected]["evaluation"]
        st.plotly_chart(prediction_scatter_figure(evaluation), use_container_width=True)
        fig = px.histogram(evaluation, x="residual", nbins=35, title="Validation residual distribution (cycles)")
        st.plotly_chart(fig, use_container_width=True)
        feature_columns = st.session_state.model_artifact["feature_columns"]
        importance = model_feature_importance(predictions[selected]["model"], feature_columns)
        if importance is not None:
            st.subheader("Feature importance")
            st.caption("Relative model importance is not a causal explanation.")
            st.plotly_chart(
                px.bar(
                    importance.head(20).sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"{selected}: feature importance",
                ),
                use_container_width=True,
            )
        else:
            st.info("Feature importance is unavailable for this algorithm.")
    if (
        st.session_state.loaded_subset == st.session_state.subset
        and st.session_state.official_predictions is not None
    ):
        st.subheader("Official NASA test predictions")
        st.json(st.session_state.official_metrics)
        st.plotly_chart(
            prediction_scatter_figure(st.session_state.official_predictions),
            use_container_width=True,
        )

    st.subheader("Compare performance across C-MAPSS subsets")
    subset_algorithm = st.selectbox(
        "Algorithm for subset comparison",
        ALGORITHMS,
        key="subset_comparison_algorithm",
    )
    if st.button("Compare available subsets"):
        subset_rows = []
        missing_subsets = []
        for subset in SUBSETS:
            if (
                st.session_state.data_source != "synthetic-demo"
                and
                subset == st.session_state.loaded_subset
                and st.session_state.train_frame is not None
            ):
                subset_frame = st.session_state.train_frame
            else:
                path = find_local_subset(DATA_DIR, subset, "train")
                if path is None:
                    missing_subsets.append(subset)
                    continue
                try:
                    subset_frame = load_dataset(path)
                except DatasetValidationError as exc:
                    st.error(f"{subset}: {exc}")
                    continue
            try:
                result, _ = compare_models(
                    subset_frame,
                    algorithms=[subset_algorithm],
                    clip_threshold=125,
                )
                subset_rows.append(
                    {
                        "Subset": subset,
                        "Operating conditions": (
                            "Multiple" if subset in {"FD002", "FD004"} else "Single"
                        ),
                        "Engines": int(subset_frame["engine_id"].nunique()),
                        **result.iloc[0].to_dict(),
                    }
                )
            except (ImportError, ValueError) as exc:
                st.error(f"{subset}: {exc}")
        st.session_state.subset_comparison = (
            pd.DataFrame(subset_rows) if subset_rows else None
        )
        if missing_subsets:
            st.info(
                "No training files found for: "
                + ", ".join(missing_subsets)
                + ". Add their standard files to data/raw/ to include them."
            )
        if len(subset_rows) < 2:
            st.warning("At least two valid subsets are needed for a cross-subset comparison.")
    if st.session_state.subset_comparison is not None:
        st.dataframe(
            st.session_state.subset_comparison,
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Each subset is validated separately by engine ID. FD002 and FD004 represent multiple "
            "operating conditions; operating settings are not flight-route labels."
        )


def _about_page() -> None:
    st.header("About AeroGuard AI")
    st.markdown(
        """
        **Problem.** Estimate remaining useful life (RUL) from historical degradation
        trajectories for simulated turbofan engines.

        **Approach.** Validate C-MAPSS engine/cycle records; calculate per-cycle training
        labels; compare regression algorithms using engine-level validation; fit a
        selected model; and explore experimental predictions and sensor trends.

        **Dataset.** NASA C-MAPSS simulated turbofan degradation data (FD001–FD004).
        FD002 and FD004 include multiple operating conditions. The three operating
        settings are dataset values, not domestic or international flight categories.
        The files do not supply actual routes, flight durations, or engine operating hours.

        **Algorithms.** XGBoost Regressor (default), Random Forest, Gradient Boosting,
        and Linear Regression. XGBoost must be installed explicitly; there is no silent
        algorithm substitution.

        **Limitations.** This is an academic prototype using simulated data. Cycle-based
        RUL does not directly translate to hours or route-specific risk. A predicted RUL
        is not a probability of failure, does not guarantee safe operation, and must not
        be used for real aircraft maintenance or safety decisions.

        **Potential extensions.** Add validated uncertainty estimation, past-only rolling
        features, external operating-hour data, and domain-reviewed alert policies.
        """
    )


_init_state()
if st.session_state.next_navigation is not None:
    st.session_state.navigation = st.session_state.next_navigation
    st.session_state.next_navigation = None
_header()
with st.sidebar:
    st.title("AEROGUARD AI")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Dataset Explorer",
            "Model Training",
            "RUL Prediction",
            "Engine Monitoring",
            "Model Comparison",
            "About",
        ],
        key="navigation",
    )
    st.caption(f"Dataset subset: {st.session_state.subset}")
    st.caption(
        "Model: "
        + (
            st.session_state.model_artifact["algorithm"]
            if st.session_state.model_artifact
            else "Not trained"
        )
    )

if page == "Overview":
    st.header("Overview")
    st.write(
        "AeroGuard AI is an experimental machine-learning prototype for exploring "
        "remaining useful life estimates from NASA C-MAPSS turbofan engine simulations."
    )
    cards = st.columns(3)
    cards[0].metric("Training data", "Not loaded" if st.session_state.train_frame is None else "Loaded")
    cards[1].metric(
        "Model status",
        "Not trained" if st.session_state.model_artifact is None else "Ready",
    )
    cards[2].metric("Subset", st.session_state.subset)
    st.subheader("ML pipeline")
    st.markdown(
        """
        `NASA C-MAPSS data` → `schema & data checks` → `cycle-based RUL labels` →
        `engine-level validation` → `model fitting` → `experimental estimate & plots`
        """
    )
    st.info(
        "Start in **Dataset Explorer** to upload data or load files from `data/raw/`. "
        "Then compare and train models. NASA data is not bundled; choose the clearly "
        "labeled synthetic demo option to exercise the prototype without it."
    )
    if st.button("Try interactive synthetic demo", type="primary"):
        _activate_demo_data(st.session_state.subset)
        st.session_state.next_navigation = "Dataset Explorer"
        st.rerun()
elif page == "Dataset Explorer":
    _dataset_page()
elif page == "Model Training":
    _training_page()
elif page == "RUL Prediction":
    _prediction_page()
elif page == "Engine Monitoring":
    _monitoring_page()
elif page == "Model Comparison":
    _comparison_page()
else:
    _about_page()
