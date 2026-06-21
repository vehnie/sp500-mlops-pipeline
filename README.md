# S&P 500 Market Direction MLOps Pipeline
### MLOps Project - NOVA IMS 2025/2026

An end-to-end MLOps proof of concept for predicting the next-day market direction of the **S&P 500** using historical OHLCV data from **Yahoo Finance**. The project is organized as a modular, Kedro-style pipeline so each component can run independently or as part of the full workflow.

> **Dataset:** Yahoo Finance historical S&P 500 index data (`^GSPC`)
> **Task:** Binary classification: predict whether the next trading day's close is higher than today's close
> **Main MLOps components:** data quality, feature store, MLflow, model versioning, explainability, serving, containers, drift monitoring, tests
> **Report limit:** maximum 6 pages
> **Submission:** report plus zipped code with a runnable sample of data, or Git link, according to course instructions

---

## Repository Structure

The project follows the organization recommended in the project guidelines and the Kedro-style layout shown in class.

```text
sp500-mlops-pipeline/
|
+-- conf/
|   +-- base/
|   |   +-- catalog.yml              # Dataset locations and artifact definitions
|   |   +-- parameters.yml           # Global project parameters
|   |   +-- parameters_data.yml      # Yahoo Finance ticker/date settings
|   |   +-- parameters_model.yml     # Model and MLflow settings
|   +-- local/
|       +-- credentials.yml          # Local credentials if needed, gitignored
|
+-- data/
|   +-- 01_raw/                      # Original Yahoo Finance downloads
|   +-- 02_intermediate/             # Standardized raw data after schema checks
|   +-- 03_primary/                  # Cleaned market dataset
|   +-- 04_feature/                  # Feature store tables
|   +-- 05_model_input/              # Time-based train/validation/test datasets
|   +-- 06_models/                   # Serialized models and preprocessors
|   +-- 07_model_output/             # Predictions, metrics, SHAP values
|   +-- 08_reporting/                # Figures, reports, drift outputs
|
+-- docs/
|   +-- project_plan.md              # Sprint planning and responsibility tracking
|   +-- model_card.md                # Final model summary and limitations
|   +-- production_notes.md          # Production risks and mitigation proposals
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb    # EDA and dataset motivation (Member 1)
|   +-- 02_feature_analysis.ipynb    # Feature behavior and leakage checks (Member 2)
|   +-- 03_model_experiments.ipynb   # Initial modeling and MLflow comparison (Member 3)
|   +-- 04_evaluation_backtest.ipynb # Metrics, plots, backtesting, SHAP (Member 4)
|   +-- 05_drift_serving_demo.ipynb  # Drift and serving demo (Member 5)
|
+-- src/
|   +-- sp500_mlops_pipeline/
|       +-- __init__.py
|       +-- __main__.py
|       +-- settings.py
|       +-- pipeline_registry.py
|       +-- pipelines/
|           +-- __init__.py
|           +-- data_ingestion/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Download Yahoo Finance data
|           |   +-- pipeline.py
|           +-- data_quality/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Schema checks, null checks, date checks
|           |   +-- pipeline.py
|           +-- data_cleaning/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Sorting, deduplication, adjusted prices
|           |   +-- pipeline.py
|           +-- data_feat_engineering/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Returns, volatility, RSI, MACD, target
|           |   +-- pipeline.py
|           +-- data_split/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Chronological train/validation/test split
|           |   +-- pipeline.py
|           +-- model_selection/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Baselines and candidate comparison
|           |   +-- pipeline.py
|           +-- model_train/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Final training, MLflow logging, registry
|           |   +-- pipeline.py
|           +-- model_predict/
|           |   +-- __init__.py
|           |   +-- nodes.py         # Batch and API prediction logic
|           |   +-- pipeline.py
|           +-- model_explainability/
|           |   +-- __init__.py
|           |   +-- nodes.py         # SHAP values and feature importance
|           |   +-- pipeline.py
|           +-- data_drifts/
|               +-- __init__.py
|               +-- nodes.py         # Drift reports on features and predictions
|               +-- pipeline.py
|
+-- app/
|   +-- main.py                      # FastAPI model-serving app
|
+-- docker/
|   +-- Dockerfile                   # Container for pipeline/API execution
|   +-- docker-compose.yml           # Optional API + MLflow stack
|
+-- tests/
|   +-- test_data_quality.py
|   +-- test_feature_engineering.py
|   +-- test_data_split.py
|   +-- test_model_train.py
|   +-- test_api.py
|
+-- report/
|   +-- report.pdf                   # Final report, maximum 6 pages
|
+-- requirements.txt
+-- pyproject.toml
+-- README.md
+-- LICENSE
+-- .gitignore
```

> **Versioning note:** Generated data, MLflow runs, serialized models, and local credentials should not be committed. The small Yahoo Finance raw CSV in `data/01_raw/` is kept as the runnable sample required by the project guidelines.

---

## Team & Responsibilities

| Member | Role | Primary Responsibilities |
|--------|------|--------------------------|
| **Member 1** | Project Lead & Data Engineer | Repo setup, Kedro structure, Yahoo Finance ingestion, raw data sample, project planning, final packaging |
| **Member 2** | Data Quality & Feature Store Lead | Data tests, schema validation, feature engineering, feature store layer, leakage prevention |
| **Member 3** | Modeling & MLflow Lead | Baseline models, model selection, hyperparameter tuning, MLflow tracking, model versioning |
| **Member 4** | Evaluation & Explainability Lead | Metrics, plots, backtesting, SHAP, feature importance, model comparison conclusions |
| **Member 5** | Serving, Drift & Report Lead | FastAPI serving, Docker, drift evaluation, production discussion, final 6-page report integration |

---

## Sprint Timeline

| Sprint | Dates | Milestones | Owner(s) |
|--------|-------|------------|----------|
| **Sprint 1** | May 7-13, 2026 | Kedro-style repo structure, `conf/`, data catalog draft, Yahoo Finance ingestion, initial EDA | M1 |
| **Sprint 2** | May 14-20, 2026 | Data quality pipeline, cleaned primary dataset, feature store pipeline, leakage-safe target | M1, M2 |
| **Sprint 3** | May 21-27, 2026 | Time-based split, baseline models, MLflow experiment tracking, first model comparison | M2, M3 |
| **Sprint 4** | May 28-Jun 3, 2026 | Tuned candidate models, final training pipeline, metrics, SHAP, evaluation figures | M3, M4 |
| **Sprint 5** | Jun 4-10, 2026 | FastAPI serving, Docker container, drift evaluation, tests for relevant pipelines | M5 |
| **Sprint 6** | Jun 11-17, 2026 | Final report, production risk discussion, reproducibility run, submission package | All |
| **Deadline** | TBD | Submit report plus code/sample data or Git link | M1, M5 |

---

## Pipeline Design

Each pipeline should be executable separately or as part of the full sequence.

```text
data_ingestion
  -> data_quality
  -> data_cleaning
  -> data_feat_engineering
  -> data_split
  -> model_selection
  -> model_train
  -> model_predict
  -> model_explainability
  -> data_drifts
```

Examples of separate runs:

```bash
kedro run --pipeline=data_quality
kedro run --pipeline=model_train
kedro run --pipeline=data_drifts
```

Full reproducibility run:

```bash
kedro run
```

---

## Data & Target Definition

The source data comes from Yahoo Finance for the S&P 500 index ticker `^GSPC`.

Required columns:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`

`Close` is the official price source for all price-derived features, returns, technical indicators, future returns, and targets; `Adj Close` is retained in the raw and cleaned schemas but is not used for feature engineering or label creation.

Default target:

```text
target_next_day_up = 1 if Close[t + 1] > Close[t] else 0
```

Time-series rules:

- Do not shuffle the dataset before splitting.
- Use only past and current data to create features.
- Shift the target forward by one trading day.
- Keep the final test period untouched until final evaluation.
- Document every split date in MLflow and in the report.

---

## Feature Store

The `data_feat_engineering` pipeline writes reusable model features to `data/04_feature/`.

Candidate features:

- Daily returns and log returns
- Lagged returns over recent trading days
- Rolling mean returns over 5, 10, 20, and 60 trading days
- Rolling volatility over 5, 10, 20, and 60 trading days
- Moving average ratios, such as `Close / SMA_20`
- Momentum indicators, such as RSI and MACD
- Volume change and rolling volume averages
- Calendar features, such as month, quarter, and day of week

Feature tests should verify:

- No unexpected nulls after preprocessing
- No duplicated dates
- Chronological ordering
- Target values are only `0` or `1`
- Feature columns do not use future information
- Train, validation, and test periods do not overlap

---

## Models

### Baselines

Baseline models establish whether the pipeline is working and provide a minimum benchmark:

- Majority-class classifier
- Previous-day direction classifier
- Logistic Regression
- Decision Tree

### Candidate Models

Final candidates for model selection:

- Random Forest
- XGBoost or LightGBM
- Support Vector Machine
- Optional shallow neural network if time permits

### MLflow Tracking

Every experiment should log:

- Model name and version
- Feature set version
- Train, validation, and test date ranges
- Hyperparameters
- Accuracy, F1-score, ROC-AUC, precision, recall
- Confusion matrix
- Feature importance or SHAP plots
- Serialized model artifact

---

## Evaluation & Explainability

The final report should include results and conclusions from both exploration and modeling.

Evaluation outputs:

- Accuracy
- F1-score
- Precision and recall
- ROC-AUC
- Confusion matrix
- Walk-forward or time-window validation
- Baseline comparison
- Optional simple trading simulation, such as cumulative return and max drawdown

Explainability outputs:

- Feature importance
- SHAP summary plot
- Short interpretation of the strongest drivers
- Known model limitations

All reporting artifacts should be saved under `data/08_reporting/`.

---

## Serving & Containers

The serving component should expose a small FastAPI application.

Required endpoints:

- `GET /health`
- `POST /predict`
- `GET /model-info`

The prediction response should include:

- Predicted class
- Probability score
- Model version
- Feature timestamp

The Docker setup should allow another user to run the API without manually rebuilding the environment.

---

## Drift Evaluation

The `data_drifts` pipeline should compare a reference dataset against a newer sample.

Recommended checks:

- Missing value changes
- Feature distribution drift
- Prediction distribution drift
- Target/performance drift when labels are available
- Drift report saved to `data/08_reporting/`

For the proof of concept, a synthetic drift scenario can be created by perturbing a sample of the data and showing how the monitoring report changes.

---

## Report Structure

The final report must be at most 6 pages and should cover:

1. Dataset choice, business/technical objective, and success metrics
2. Project planning and sprint organization
3. Data exploration results
4. Modeling results, metrics, feature importance, and SHAP explanations
5. MLOps implementation: pipeline orchestration, MLflow, tests, serving, containers, drift
6. Production discussion: advantages of chosen technologies, risks, limitations, and mitigations
7. Package list and versions used

---

## Getting Started

### 1. Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure data source

Set the Yahoo Finance ticker and date range in `conf/base/parameters_data.yml`:

```yaml
ticker: "^GSPC"
start_date: "2000-01-01"
end_date: "2026-05-07"
```

### 3. Run the full pipeline

```bash
kedro run
```

### 4. Run selected pipelines

```bash
kedro run --pipeline=data_ingestion
kedro run --pipeline=model_selection
kedro run --pipeline=data_drifts
```

### 5. Serve the candidate-champion model

Activate the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

The MLflow server must proxy artifacts instead of returning Windows-local
`file:///C:/...` paths to clients. If this project database was created before
artifact proxying was enabled, stop MLflow and run this one-time, idempotent
migration from the repository root:

```powershell
python .\scripts\migrate_mlflow_artifact_uris.py
```

The migration changes only artifact-location metadata, creates
`mlflow.db.before-artifact-proxy.bak`, and does not modify model files, runs,
metrics, registered versions, or aliases.

Start the local MLflow Tracking Server and Model Registry:

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

Start the FastAPI application:

```powershell
uvicorn sp500_mlops_pipeline.serving.app:app --reload --port 8000
```

Open the automatic API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Run the local MLOps dashboard

Ensure Streamlit is installed in the active environment, then run:

```powershell
streamlit run streamlit_app.py
```

The dashboard is read-only and uses the existing Great Expectations reports,
model metrics, predictions, and SHAP explainability artefacts under `data/`.

### 7. Run the FastAPI application with Docker

Confirm that Docker Desktop is open.

Start MLflow on the Windows host so it is reachable from Docker:

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

Run the one-time URI migration shown in step 5 before this command if the
database still contains `file:///C:/...` artifact locations. With this setup,
the Registry returns `mlflow-artifacts:/...` locations and the container
downloads the selected model through the MLflow HTTP server.

The explicit `file:///...` URI is important on Windows: passing a raw
`C:\...` path makes MLflow interpret the drive letter as an artifact-store
scheme.

Build the API image:

```powershell
docker build -t sp500-direction-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 sp500-direction-api
```

Open the API documentation and health endpoint:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

Stop the foreground container with `Ctrl+C`. If it was started with `-d`, use:

```powershell
docker stop <container-id-or-name>
```

---

## Git Workflow

1. Each member works on a dedicated branch: `feature/<topic>`.
2. Open a Pull Request into `main` when a milestone is complete.
3. Member 1 reviews structure, reproducibility, and merge conflicts.
4. At least one technical owner reviews modeling, serving, and monitoring PRs.
5. Commit messages follow the format: `[M3] Add MLflow logging to XGBoost training`.

Suggested branches:

```text
feature/kedro-structure
feature/data-quality
feature/feature-store
feature/model-selection
feature/model-serving
feature/drift-monitoring
feature/report
```

---

## Submission Checklist

- [ ] Maximum 6-page report exported to `report/report.pdf`
- [ ] Project planning included in the report
- [ ] Code organized in modular Kedro-style pipelines
- [ ] Yahoo Finance sample data included or easy to regenerate
- [ ] Data quality tests implemented
- [ ] Feature store layer implemented
- [ ] MLflow experiment tracking and model versioning included
- [ ] Main metrics and explainability artifacts saved
- [ ] SHAP or feature importance included in the report
- [ ] Model serving implemented with FastAPI
- [ ] Docker container builds and runs
- [ ] Data drift evaluation implemented
- [ ] Unit tests for relevant functions and pipelines pass
- [ ] Requirements/package versions listed
- [ ] Another user can run the pipeline and reproduce the results
- [ ] Final zip or Git link prepared according to course instructions

---

## Requirements

Key dependencies to include in `requirements.txt`:

- `kedro`
- `kedro-datasets`
- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `mlflow`
- `shap`
- `fastapi`
- `uvicorn`
- `pydantic`
- `great_expectations`
- `evidently`
- `pytest`

---

## License

Academic project - NOVA IMS MLOps course, 2025/2026.
