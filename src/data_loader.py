from __future__ import annotations

import io
import re
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

NASA_COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{number}" for number in range(1, 22)],
]
REQUIRED_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]


class DatasetValidationError(ValueError):
    """Raised when an uploaded or local dataset is not a valid C-MAPSS table."""


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "unit": "engine_id",
        "unit_number": "engine_id",
        "engine": "engine_id",
        "engineid": "engine_id",
        "time": "cycle",
        "time_in_cycles": "cycle",
        "cycle_number": "cycle",
    }
    renamed: dict[object, str] = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        renamed[column] = aliases.get(normalized, normalized)
    frame = frame.rename(columns=renamed)

    if "sensor_1" not in frame.columns:
        sensor_columns = [
            column
            for column in frame.columns
            if str(column).startswith(("sensor_", "s_"))
        ]
        if sensor_columns and all(str(column).startswith("s_") for column in sensor_columns):
            frame = frame.rename(
                columns={column: f"sensor_{str(column)[2:]}" for column in sensor_columns}
            )
    return frame


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and values, then return a stable, cycle-sorted copy."""
    frame = _normalize_columns(frame.copy())
    if list(frame.columns) == NASA_COLUMNS:
        pass
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    sensor_columns = [column for column in frame.columns if str(column).startswith("sensor_")]
    if any(re.fullmatch(r"sensor_\d+", str(column)) is None for column in sensor_columns):
        raise DatasetValidationError("Sensor columns must use names such as sensor_1, sensor_2, etc.")
    sensor_columns.sort(key=lambda name: int(str(name).split("_", 1)[1]))
    if missing_columns or not sensor_columns:
        missing = ", ".join(missing_columns) if missing_columns else "at least one sensor_N column"
        raise DatasetValidationError(f"Required dataset columns are missing: {missing}.")

    selected = REQUIRED_COLUMNS + sensor_columns
    frame = frame.loc[:, selected]
    if frame.empty:
        raise DatasetValidationError("The dataset contains no observations.")
    if frame.isna().any().any():
        missing_count = int(frame.isna().sum().sum())
        raise DatasetValidationError(
            f"The dataset contains {missing_count} missing value(s). Clean the data before upload."
        )
    for column in selected:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise DatasetValidationError("All engine, cycle, setting, and sensor values must be numeric.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise DatasetValidationError("The dataset contains infinite or invalid numeric values.")
    if (frame["engine_id"] <= 0).any() or (frame["cycle"] <= 0).any():
        raise DatasetValidationError("Engine IDs and cycle numbers must be positive.")
    if (frame["engine_id"] % 1 != 0).any() or (frame["cycle"] % 1 != 0).any():
        raise DatasetValidationError("Engine IDs and cycle numbers must be whole numbers.")
    duplicates = frame.duplicated(["engine_id", "cycle"])
    if duplicates.any():
        raise DatasetValidationError(
            f"Found {int(duplicates.sum())} duplicate engine/cycle record(s)."
        )
    frame["engine_id"] = frame["engine_id"].astype("int64")
    frame["cycle"] = frame["cycle"].astype("int64")
    return frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def _read_frame(source: str | Path | IO[bytes] | IO[str] | bytes, filename: str = "") -> pd.DataFrame:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    name = filename or str(getattr(source, "name", source))
    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".txt":
            raw = pd.read_csv(source, sep=r"\s+", header=None, comment="#")
            if raw.shape[1] == len(NASA_COLUMNS):
                raw.columns = NASA_COLUMNS
                return raw
            raise DatasetValidationError(
                f"NASA whitespace files need {len(NASA_COLUMNS)} columns; found {raw.shape[1]}."
            )
        raw = pd.read_csv(source, sep=None, engine="python")
    except DatasetValidationError:
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DatasetValidationError(f"Could not read dataset file: {exc}") from exc

    normalized = _normalize_columns(raw)
    if all(column in normalized.columns for column in REQUIRED_COLUMNS):
        return normalized
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        raw = pd.read_csv(source, sep=None, engine="python", header=None)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DatasetValidationError(f"Could not read dataset file: {exc}") from exc
    if raw.shape[1] != len(NASA_COLUMNS):
        raise DatasetValidationError(
            "Headerless NASA files must contain exactly 26 whitespace-separated or CSV columns."
        )
    raw.columns = NASA_COLUMNS
    return raw


def load_dataset(
    source: str | Path | IO[bytes] | IO[str] | bytes, filename: str = ""
) -> pd.DataFrame:
    """Load a headered compatible CSV or standard headerless NASA TXT/CSV file."""
    return validate_dataset(_read_frame(source, filename))


def load_rul_reference(source: str | Path | IO[bytes] | IO[str]) -> pd.Series:
    """Read the official one-label-per-test-engine RUL reference file."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        values = pd.read_csv(source, sep=r"\s+|,", header=None, engine="python").iloc[:, 0]
        values = pd.to_numeric(values, errors="raise")
    except (pd.errors.ParserError, ValueError, IndexError, OSError) as exc:
        raise DatasetValidationError(f"Could not read RUL reference file: {exc}") from exc
    if values.empty or values.isna().any() or (values < 0).any():
        raise DatasetValidationError("RUL reference labels must be non-empty, numeric, and non-negative.")
    return values.reset_index(drop=True).rename("rul")


def find_local_subset(data_dir: str | Path, subset: str, role: str) -> Path | None:
    """Find a standard NASA dataset file without substituting synthetic data."""
    path = Path(data_dir) / f"{role}_{subset}.txt"
    return path if path.is_file() else None
