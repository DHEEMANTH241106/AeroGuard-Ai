# AeroGuard AI

**Aircraft Engine Remaining Useful Life Prediction** is an academic Streamlit
prototype for learning from NASA C-MAPSS simulated turbofan degradation trajectories.
It includes schema validation, cycle-based RUL labels, held-out-engine evaluation,
model persistence, predictions, and interactive sensor/operating-setting charts.

> **Academic prototype only.** Estimates are experimental and are not approved for
> real-world aviation safety or maintenance decisions. Predicted RUL is neither a
> probability of failure nor a guarantee that an engine will remain safe for that
> number of cycles.

## Project layout

```text
app.py                       Streamlit pages and user interface
src/data_loader.py           NASA/CSV parsing, schema validation, local file lookup
src/preprocessing.py         Deterministic feature selection and input checks
src/rul_calculator.py        Training and official-test RUL label construction
src/train_model.py           Engine-level validation, model comparison, training
src/predict_rul.py            Artifact loading and prediction validation
src/evaluate_model.py        Regression metrics and feature importance
src/visualization.py         Sensor, operating-setting, and prediction charts
data/README.md               NASA dataset instructions
data/raw/                    User-provided NASA data (not included)
models/                      Model artifacts saved after training
tests/                       RUL, data, preprocessing, split, and prediction tests
requirements.txt             Python dependencies
```

## Windows PowerShell setup

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, either allow it for the current
terminal session with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`,
or call the environment executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

## Prepare NASA C-MAPSS data

1. Download the C-MAPSS dataset from the official NASA source:
   [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
   or [NASA Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
2. Extract the archive and place the selected subset's standard text files under
   `data/raw/`. For FD001, the filenames are:
   `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.
   The same naming pattern applies to FD002, FD003, and FD004.
3. Alternatively, upload a compatible CSV or TXT through Dataset Explorer. Headerless
   NASA-format TXT files and 26-column CSV files are supported; headed CSV files must
   contain `engine_id`, `cycle`, `setting_1`–`setting_3`, and one or more `sensor_N`
   fields. The test reference file is optional.

The app does not download data or silently generate replacement training data.
Without a real dataset, the interface remains usable and shows setup instructions;
training and predictions require real data.

For an interactive walkthrough before NASA data is available, click **Load synthetic
demo data** in Dataset Explorer. This is an explicit opt-in generator for reproducible,
synthetic trajectories with synthetic reference labels. The app marks this mode and
its metrics as synthetic; it is for demonstrating controls only, not for claiming NASA
performance or making real-world decisions. Load real NASA data again to replace it.

## Run the application and tests

```powershell
streamlit run app.py
```

In the UI, choose **Try interactive synthetic demo** on Overview for a quick
walkthrough, or load a real subset in **Dataset Explorer**. Then select an algorithm
and choose **Compare, train and save** in **Model Training**. A model artifact is saved under
`models/`. Predictions can use the loaded test engines or an uploaded historical
sequence. Run the tests from the project directory:

```powershell
python -m pytest -q
```

The default model is XGBoost. If the optional XGBoost import is unavailable, the app
reports the installation command; choose Random Forest explicitly for a fallback.
No algorithm is silently substituted.

## RUL targets and evaluation

For each labeled training row, the target is
`maximum observed cycle for the engine - current cycle`. By default, training targets
are clipped at 125 cycles; this is configurable in Model Training, and setting the
threshold to zero disables clipping. Clipping caps the target values used for fitting;
it does not alter cycles or sensor data. Rolling means/standard deviations (window
choices 5, 10, or 20) and per-cycle sensor differences can be enabled in Model
Training. They are calculated within each engine using current/past observations only.
Model feature selection is configurable, and constant/near-constant sensors are flagged
but not silently removed.

Validation uses a reproducible engine-level split: an engine's rows do not cross from
training to validation. For the official test set, its final observed row is evaluated
against the corresponding NASA reference label. If deriving labels for all observed
test cycles, the reference RUL at the final observed cycle is offset by
`final observed cycle - current cycle`. MAE and RMSE are measured in cycles; R² is
dimensionless. Results are computed at runtime from the loaded dataset, never hardcoded.

The model uses operating settings and sensor values at the selected observation only.
It does not use later sensor readings to predict an earlier cycle. Operating settings
are not domestic/international flight labels; C-MAPSS contains no actual routes,
flight durations, or operating-hour labels. Cycle-based RUL does not directly represent
operating hours or route-specific risk.

## Troubleshooting

- **Dataset unavailable:** Check `data/raw/` names, select the matching subset, or use
  the upload widgets. Dataset files are intentionally not included.
- **Invalid schema:** Keep the engine ID, cycle, three setting, and sensor columns;
  remove missing/non-numeric values and duplicate engine/cycle rows.
- **XGBoost import error:** Run `pip install xgboost` in the active environment or
  explicitly choose Random Forest.
- **PowerShell activation error:** Use the `Set-ExecutionPolicy -Scope Process ...`
  command above or run `.venv\Scripts\python.exe` / `.venv\Scripts\streamlit.exe`.
- **Model comparison requires at least two engines:** A held-out-engine validation
  cannot be performed from a single engine trajectory.

## Demonstration outline

1. Explain the simulated NASA dataset and the distinction between cycle number and RUL.
2. Show schema validation, engine/cycle counts, and sensor/operating-setting plots.
3. Train and compare models; describe held-out-engine validation and cycle-based metrics.
4. Select an engine and observed cycle, then show its available input values, estimate,
   and sensor history.
5. Emphasize that the estimate and configurable monitoring category are academic
   illustrations only and are not maintenance guidance or a safety clearance.
