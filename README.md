# S&P 500 Market Direction MLOps Pipeline

### MLOps Project - NOVA IMS 2025/2026

This repository contains the final Kedro-based MLOps implementation for predicting the next-day direction of the **S&P 500** from historical OHLCV market data.

The final version is intentionally simple: it focuses on a reproducible pipeline, model tracking, explainability, drift monitoring, testing, and a small serving API. Earlier exploratory ideas are kept only as reference material where useful, but the current Kedro registry is the source of truth for the implemented project.

> **Dataset:** Yahoo Finance historical S&P 500 index data (`^GSPC`)  
> **Task:** binary classification: predict whether the next trading day's close is higher than today's close  
> **Main MLOps components:** Kedro, MLflow, SHAP, Evidently, Great Expectations, FastAPI, Docker, pytest

---

## Repository Structure

```text
sp500-mlops-pipeline/
|
+-- conf/
|   +-- base/
|   |   +-- catalog.yml              # Dataset locations and artifacts
|   |   +-- parameters.yml           # Global parameters
|   |   +-- parameters_data.yml      # Market data settings
|   |   +-- parameters_model.yml     # Model and MLflow settings
|   +-- local/
|       +-- credentials.yml          # Local credentials if needed, gitignored
|
+-- data/
|   +-- 01_raw/                      # Original Yahoo Finance sample data
|   +-- 02_intermediate/             # Standardized market data
|   +-- 03_primary/                  # Cleaned market dataset
|   +-- 04_feature/                  # Local feature layer
|   +-- 05_model_input/              # Chronological train/validation/test datasets
|   +-- 06_models/                   # Serialized local model artifacts
|   +-- 07_model_output/             # Predictions and metrics
|   +-- 08_reporting/                # SHAP, drift, validation, MLflow artifacts
|
+-- docs/
|   +-- model_card.md
|   +-- production_notes.md
|   +-- project_plan.md
|   +-- report_evidence.md
|
+-- notebooks/
|   +-- 00_check_data_ingestion_plot.ipynb
|   +-- 01_final_project_walkthrough.ipynb
|   +-- thesis_reference/
|       +-- 01_market_data_ingestion.ipynb
|       +-- 03_market_feature_engineering.ipynb
|       +-- 04_dataset_construction_legacy_sentiment.ipynb
|       +-- README.md
|
+-- src/
|   +-- sp500_mlops_pipeline/
|       +-- pipeline_registry.py
|       +-- pipelines/
|       |   +-- data_cleaning/
|       |   +-- data_drift/
|       |   +-- data_expectations/
|       |   +-- data_feat_engineering/
|       |   +-- data_quality/
|       |   +-- data_split/
|       |   +-- model_explainability/
|       |   +-- model_predict/
|       |   +-- model_predict_challenger/
|       |   +-- model_train/
|       |   +-- model_train_challenger/
|       +-- serving/
|           +-- app.py
|           +-- model_loader.py
|           +-- schemas.py
|
+-- docker/
|   +-- docker-compose.yml          # Builds/runs the serving API via the root Dockerfile
|
+-- tests/
|   +-- test_*.py
|
+-- Dockerfile                       # Serving API image (used by docker build and compose)
+-- requirements.txt
+-- pyproject.toml
+-- README.md
```

The notebooks under `notebooks/thesis_reference/` are legacy reference notebooks. They are not used as the source of truth for the final implementation.

---

## Pipeline Design

The project is organized into modular Kedro pipelines. The current registry exposes these pipelines:

```text
__default__
data_cleaning
data_drift
data_expectations
data_feat_engineering
data_quality
data_split
model_explainability
model_predict
model_predict_challenger
model_train
model_train_challenger
```

The core/default workflow is:

```text
data_quality
  -> data_cleaning
  -> data_feat_engineering
  -> data_split
  -> model_train
  -> model_train_challenger
  -> model_predict
  -> model_predict_challenger
```

Additional reporting and monitoring pipelines are run separately when needed:

- `data_expectations`
- `model_explainability`
- `data_drift`

This keeps the default run focused on producing the feature data, splits, trained models, and predictions. The reporting pipelines generate extra validation, SHAP, and drift artifacts for analysis and the final report.

---

## Data & Target Definition

The source data is historical S&P 500 index data from Yahoo Finance using ticker `^GSPC`.

Required market columns:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`

The target is:

```text
target_next_day_up = 1 if Close[t + 1] > Close[t] else 0
```

Because this is time series data, the train/validation/test split is chronological. The project does not shuffle the rows before splitting, since that would leak future market information into training.

---

## Feature Layer

The implemented project uses a local feature layer stored under:

```text
data/04_feature/
```

The main feature dataset is:

```text
data/04_feature/sp500_feature_data.csv
```

The final production features used by the implemented model pipelines are:

- `simple_return`
- `log_return`
- `sma_10`
- `sma_20`
- `sma_ratio_10`
- `rsi_14`
- `volatility_10`
- `volume_change`

Earlier in the project, broader feature ideas such as MACD, calendar variables, lagged returns, and longer rolling windows were considered. They are not part of the final implemented model pipeline. I kept the final feature set smaller so the model and explanations stayed easier to validate for the report.

---

## Models

The final implementation compares two models:

- **Logistic Regression** — baseline and selected champion model
- **Random Forest** — challenger model

The Logistic Regression model is trained in the `model_train` pipeline. The Random Forest challenger is trained in the `model_train_challenger` pipeline.

Other candidates such as XGBoost, LightGBM, Support Vector Machines, and shallow neural networks were planning-stage ideas. They are not part of the final implemented Kedro model workflow and should be treated as future work rather than current project output.

---

## Final Results

The final test results are:

### Logistic Regression

| Metric | Value |
|---|---:|
| Accuracy | 0.5363 |
| Precision | 0.5422 |
| Recall | 0.9218 |
| F1-score | 0.6828 |
| ROC-AUC | 0.4991 |

### Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.5030 |
| Precision | 0.5474 |
| Recall | 0.4730 |
| F1-score | 0.5075 |
| ROC-AUC | 0.5020 |

Logistic Regression was selected as the champion model because it achieved better accuracy, recall, and F1-score while staying simpler and more interpretable.

At the same time, both models have ROC-AUC values close to 0.50. That is an important limitation: the current feature signal is weak, and the result should be interpreted as an MLOps implementation success more than a strong market-prediction model.

---

## MLflow Tracking and Registry

Experiments are tracked with MLflow. The champion model is registered as:

```text
sp500_direction_model
```

The serving code loads the model using the registry alias:

```text
candidate_champion
```

MLflow is used to keep model versions, metrics, parameters, and artifacts connected to the pipeline runs. This makes the API and the reporting artifacts easier to trace back to the run that produced them.

---

## Evaluation & Explainability

The `model_explainability` pipeline generates SHAP artifacts for the champion Logistic Regression model.

Main outputs:

```text
data/08_reporting/shap/logistic_regression_v2_summary_plot.png
data/08_reporting/shap/logistic_regression_v2_feature_importance_bar.png
data/08_reporting/shap/logistic_regression_v2_explainability_summary.json
```

SHAP is used here to understand how the final input features contribute to model predictions. This is especially useful because the model is simple enough to inspect, but the feature effects still benefit from a consistent explanation method.

---

## Drift Monitoring

The implemented Evidently drift pipeline is named:

```text
data_drift
```

It compares:

- reference data = oldest 70%
- current data = newest 30%

The generated outputs are:

```text
data/08_reporting/drift/data_drift_report.html
data/08_reporting/drift/data_drift_summary.json
```

The drift report is part of the monitoring layer. It does not automatically retrain the model, but it gives a concrete signal about whether the feature distributions seen by the model have changed over time.

---

## Serving & Docker

The serving component uses FastAPI and loads the registered MLflow model.

Available endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

The Docker image used for the API is:

```text
sp500-direction-api
```

The API is intentionally small: it mainly validates the input features, loads the selected model from MLflow, and returns the predicted class and probability.

---

## Getting Started

This is a step-by-step guide to set up the project and run it end to end. The
commands are written for **Windows PowerShell** (the project's primary
environment); equivalent macOS/Linux commands are given where they differ.

### Prerequisites

- **Python 3.11 or newer** (validated on 3.11; `pyproject.toml` targets 3.13).
- **git** to clone the repository.
- **Docker Desktop** — optional, only needed for the containerised API (Step 11).

> The repository already ships a ready-to-use sample dataset at
> `data/01_raw/sp500_yahoo_finance_raw.csv`, so **no data download is required**
> to run the pipeline.

### Step 1 — Clone the repository

```powershell
git clone https://github.com/vehnie/sp500-mlops-pipeline.git
cd sp500-mlops-pipeline
```

### Step 2 — Create and activate a virtual environment

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

> If PowerShell blocks the activation script, allow it for the current session
> with: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — (Optional) Review the data source settings

The provenance of the bundled sample (ticker, date range, interval) is recorded
in `conf/base/parameters_data.yml`. The default workflow reads the bundled CSV
directly, so you can skip this step unless you want to regenerate the extract.

### Step 5 — Run the full pipeline

> **Start MLflow first.** The `model_train` and `model_train_challenger`
> pipelines log runs to the MLflow tracking server at `http://127.0.0.1:5000`,
> so it must be running **before** `kedro run`. Open a separate terminal and
> start it with the MLflow command from [Step 9, Terminal A](#step-9--serve-the-champion-model-mlflow--fastapi), then run:

```powershell
kedro run
```

This executes the default workflow:

```text
data_quality -> data_cleaning -> data_feat_engineering -> data_split
  -> model_train -> model_train_challenger
  -> model_predict -> model_predict_challenger
```

It produces the cleaned dataset, feature table, chronological train/validation/
test splits, the trained champion and challenger models, and their
predictions/metrics under `data/`.

### Step 6 — (Optional) Run individual pipelines

The reporting and monitoring pipelines are run on demand:

```powershell
kedro run --pipeline=data_quality
kedro run --pipeline=data_expectations      # Great Expectations validation reports
kedro run --pipeline=data_feat_engineering
kedro run --pipeline=model_train
kedro run --pipeline=model_train_challenger
kedro run --pipeline=model_explainability   # SHAP artifacts
kedro run --pipeline=data_drift             # Evidently drift report
```

### Step 7 — (Optional) Visualise the pipeline graph

```powershell
pip install kedro-viz
kedro viz
```

Then open the URL it prints (defaults to `http://127.0.0.1:4141`).

### Step 8 — Run the tests

```powershell
pytest
```

Expected result: **80 passed, 11 warnings**.

### Step 9 — Serve the champion model (MLflow + FastAPI)

The API loads the champion model from a local MLflow server, so you need **two
terminals** (both with the virtual environment activated).

**Terminal A — start the MLflow Tracking Server and Model Registry:**

```powershell
$artifactUri = [System.Uri]::new(
    (Resolve-Path .\data\08_reporting\mlflow_artifacts).Path
).AbsoluteUri

mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root mlflow-artifacts:/ `
  --serve-artifacts `
  --artifacts-destination $artifactUri
```

**Terminal B — start the FastAPI application.** The package lives under `src/`,
so add it to `PYTHONPATH` first:

```powershell
$env:PYTHONPATH = "src"
uvicorn sp500_mlops_pipeline.serving.app:app --reload --port 8000
```

macOS / Linux equivalent:

```bash
PYTHONPATH=src uvicorn sp500_mlops_pipeline.serving.app:app --reload --port 8000
```

Then open the interactive API docs and exercise the endpoints
(`GET /health`, `GET /model-info`, `POST /predict`):

```text
http://127.0.0.1:8000/docs
```

### Step 10 — (Optional) Run the dashboard

```powershell
streamlit run streamlit_app.py
```

The dashboard is read-only and renders the existing artifacts under `data/`.

### Step 11 — (Alternative) Run the API with Docker

This replaces Step 9's FastAPI process with a container. Make sure **Docker
Desktop is running**, then start MLflow on the host so the container can reach
it at `host.docker.internal:5000`:

```powershell
$artifactUri = [System.Uri]::new(
    (Resolve-Path .\data\08_reporting\mlflow_artifacts).Path
).AbsoluteUri

mlflow server --host 0.0.0.0 --port 5000 `
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:5000" `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root mlflow-artifacts:/ `
  --serve-artifacts `
  --artifacts-destination $artifactUri
```

Then either build and run the image directly:

```powershell
docker build -t sp500-direction-api .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 sp500-direction-api
```

…or use the bundled Compose file (builds from the root `Dockerfile`):

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

---

## Validation Status

The final local validation completed with:

```text
80 passed, 11 warnings
```

The full `kedro run` also completed successfully, and Kedro-Viz was used to inspect the pipeline graph.

---

## Requirements

Main dependencies used by the final implementation:

- `kedro`
- `kedro-datasets`
- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `mlflow`
- `shap`
- `fastapi`
- `uvicorn`
- `pydantic`
- `great_expectations`
- `evidently`
- `streamlit`
- `pytest`

`requirements.txt` lists exactly these dependencies. Earlier experimentation packages such as `xgboost` and `seaborn` have been removed, since they are not used by the final implemented Kedro model pipeline.

---

## License

Academic project - NOVA IMS MLOps course, 2025/2026.
