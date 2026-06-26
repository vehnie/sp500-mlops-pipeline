# S&P 500 MLOps Technical Manual

## Detailed Outline

This document is the planned structure for the full technical manual of the final S&P 500 MLOps project. It is intentionally an outline only. The full manual content should be written later using the implemented Kedro project, generated artifacts, and final report evidence as the source of truth.

The expected final manual length is around 20-40 pages, depending on the number of figures, tables, and screenshots included.

Primary sources for the manual:

- `README.md`
- `docs/CODEX_DOCUMENTATION_GUIDE.md`
- `notebooks/01_final_project_walkthrough.ipynb`
- `conf/base/catalog.yml`
- `src/sp500_mlops_pipeline/pipeline_registry.py`

Supporting implementation sources to reference when writing the full manual:

- `src/sp500_mlops_pipeline/pipelines/`
- `src/sp500_mlops_pipeline/serving/`
- `data/04_feature/sp500_feature_data.csv`
- `data/05_model_input/`
- `data/07_model_output/`
- `data/08_reporting/`
- `Dockerfile`
- `docker/`
- `tests/`

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Repository Structure](#2-repository-structure)
3. [Project Objectives](#3-project-objectives)
4. [Problem Definition](#4-problem-definition)
5. [Dataset Description](#5-dataset-description)
6. [Kedro Pipeline Architecture](#6-kedro-pipeline-architecture)
7. [Data Quality and Validation](#7-data-quality-and-validation)
8. [Data Cleaning](#8-data-cleaning)
9. [Feature Engineering and Feature Layer](#9-feature-engineering-and-feature-layer)
10. [Train/Validation/Test Split](#10-trainvalidationtest-split)
11. [Model Training](#11-model-training)
12. [Model Evaluation](#12-model-evaluation)
13. [MLflow Tracking and Model Registry](#13-mlflow-tracking-and-model-registry)
14. [Explainability with SHAP](#14-explainability-with-shap)
15. [Data Drift Monitoring with Evidently](#15-data-drift-monitoring-with-evidently)
16. [FastAPI Serving Layer](#16-fastapi-serving-layer)
17. [Docker Deployment](#17-docker-deployment)
18. [Testing and Reproducibility](#18-testing-and-reproducibility)
19. [Experimental Results](#19-experimental-results)
20. [MLOps Results](#20-mlops-results)
21. [Limitations](#21-limitations)
22. [Future Work](#22-future-work)
23. [Appendix](#23-appendix)

---

## 1. Introduction

### Chapter purpose

Introduce the technical manual and explain what part of the project it documents. The tone should be practical: this manual is for someone who wants to understand and reproduce the final implemented MLOps workflow.

### Key points to cover later

- The project predicts next-day S&P 500 market direction from historical OHLCV data.
- The final implementation is based on Kedro pipelines.
- The project includes model tracking, validation, explainability, drift monitoring, serving, Docker, and tests.
- The manual describes the implemented state, not the earlier planning-stage ideas.

### Expected figures, tables, or screenshots

- Small project overview diagram.
- Optional table summarizing the main MLOps components.

### Source files or artifacts to use

- `README.md`
- `notebooks/01_final_project_walkthrough.ipynb`
- `src/sp500_mlops_pipeline/pipeline_registry.py`

### Placeholder

To be written.

---

## 2. Repository Structure

### Chapter purpose

Document the current repository layout so another user can navigate the project.

### Key points to cover later

- Explain the main folders: `conf/`, `data/`, `docs/`, `notebooks/`, `src/`, `tests/`, `docker/`.
- Explain that `notebooks/thesis_reference/` contains legacy reference material only.
- Highlight that the final project source of truth is the Kedro implementation under `src/sp500_mlops_pipeline/`.
- Mention where model artifacts, predictions, reports, and drift outputs are stored.

### Expected figures, tables, or screenshots

- Repository tree table or code block.
- Table mapping folders to their role.

### Source files or artifacts to use

- `README.md`
- `conf/base/catalog.yml`
- Current repository tree.

### Placeholder

To be written.

---

## 3. Project Objectives

### Chapter purpose

Explain what the project was designed to achieve from both a modelling and MLOps perspective.

### Key points to cover later

- Build a reproducible ML workflow for S&P 500 direction prediction.
- Use Kedro to organize data, model, reporting, and monitoring pipelines.
- Track model experiments and registry state with MLflow.
- Provide explainability with SHAP.
- Validate data with Great Expectations.
- Monitor data drift with Evidently.
- Serve the selected model with FastAPI and Docker.
- Validate the implementation with pytest.

### Expected figures, tables, or screenshots

- Table separating modelling objectives from MLOps objectives.
- Short checklist of implemented components.

### Source files or artifacts to use

- `README.md`
- `docs/report_evidence.md`
- `docs/model_card.md`
- `notebooks/01_final_project_walkthrough.ipynb`

### Placeholder

To be written.

---

## 4. Problem Definition

### Chapter purpose

Define the supervised learning problem and the prediction target clearly.

### Key points to cover later

- The problem is binary classification.
- The target is `target_next_day_up`.
- The target indicates whether the next trading day's close is higher than the current day's close.
- Explain why this is a difficult prediction task.
- Mention that the objective is not to claim a production trading strategy, but to demonstrate a complete MLOps workflow.

### Expected figures, tables, or screenshots

- Target definition table.
- Optional class distribution table if taken from the final feature dataset.

### Source files or artifacts to use

- `README.md`
- `notebooks/01_final_project_walkthrough.ipynb`
- `data/04_feature/sp500_feature_data.csv`
- `src/sp500_mlops_pipeline/pipelines/data_feat_engineering/`

### Placeholder

To be written.

---

## 5. Dataset Description

### Chapter purpose

Describe the market dataset used by the project and the main columns carried through the pipeline.

### Key points to cover later

- Source data comes from Yahoo Finance for ticker `^GSPC`.
- Main OHLCV fields: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- Explain the date range and number of rows using the final generated dataset.
- Mention that `data/01_raw/` contains the raw sample and `data/04_feature/` contains the final feature dataset.
- Keep this chapter descriptive and avoid exploratory analysis that belongs in old notebooks.

### Expected figures, tables, or screenshots

- Dataset schema table.
- Dataset shape and date range table.
- Optional simple line plot of the closing price if needed.

### Source files or artifacts to use

- `conf/base/catalog.yml`
- `data/01_raw/sp500_yahoo_finance_raw.csv`
- `data/04_feature/sp500_feature_data.csv`
- `notebooks/00_check_data_ingestion_plot.ipynb`
- `notebooks/01_final_project_walkthrough.ipynb`

### Placeholder

To be written.

---

## 6. Kedro Pipeline Architecture

### Chapter purpose

Explain the implemented Kedro architecture and how the registered pipelines fit together.

### Key points to cover later

- Use the current pipeline registry as the source of truth.
- Document registered pipelines:
  - `__default__`
  - `data_cleaning`
  - `data_drift`
  - `data_expectations`
  - `data_feat_engineering`
  - `data_quality`
  - `data_split`
  - `model_explainability`
  - `model_predict`
  - `model_predict_challenger`
  - `model_train`
  - `model_train_challenger`
- Explain the default workflow:
  - `data_quality`
  - `data_cleaning`
  - `data_feat_engineering`
  - `data_split`
  - `model_train`
  - `model_train_challenger`
  - `model_predict`
  - `model_predict_challenger`
- Explain that `data_expectations`, `model_explainability`, and `data_drift` are additional validation/reporting/monitoring pipelines.
- Avoid documenting old pipeline names as implemented.

### Expected figures, tables, or screenshots

- Kedro-Viz pipeline graph screenshot.
- Pipeline registry table.
- Default workflow diagram.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipeline_registry.py`
- `README.md`
- Kedro-Viz screenshot if available.

### Placeholder

To be written.

---

## 7. Data Quality and Validation

### Chapter purpose

Describe the data quality checks and validation layer used before modelling.

### Key points to cover later

- Explain the role of the `data_quality` pipeline.
- Explain the role of the `data_expectations` pipeline.
- Mention Great Expectations validation for raw and feature data.
- Document the generated JSON and static HTML validation reports.
- Explain that Great Expectations formalizes data contracts but does not replace
  pytest tests or the pandas checks in `data_quality`.
- Mention the current validation results:
  - raw market data contract: 19 / 19 expectations passed;
  - feature dataset contract: 27 / 27 expectations passed.
- Explain why validation is important for time-series modelling.

### Expected figures, tables, or screenshots

- Table of validation checks.
- Screenshot or excerpt from Great Expectations report JSON.
- Screenshot placeholder for `Feature Data Validation Report`.
- Screenshot placeholder for `Raw Data Validation Report`.
- Validation artifact summary table.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/data_quality/`
- `src/sp500_mlops_pipeline/pipelines/data_expectations/`
- `data/08_reporting/great_expectations/raw_data_validation_report.json`
- `data/08_reporting/great_expectations/raw_data_validation_report.html`
- `data/08_reporting/great_expectations/feature_data_validation_report.json`
- `data/08_reporting/great_expectations/feature_data_validation_report.html`
- `conf/base/catalog.yml`

### Placeholder

To be written.

---

## 8. Data Cleaning

### Chapter purpose

Explain how raw validated market data is transformed into the cleaned dataset used for feature engineering.

### Key points to cover later

- Explain the role of the `data_cleaning` pipeline.
- Describe cleaning steps only after confirming them from the actual node implementation.
- Mention output dataset `sp500_cleaned_data`.
- Keep the chapter focused on implemented transformations.

### Expected figures, tables, or screenshots

- Input/output dataset table.
- Before/after row count table if available.
- Cleaning step summary table.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/data_cleaning/`
- `conf/base/catalog.yml`
- `data/03_primary/sp500_cleaned_data.csv`

### Placeholder

To be written.

---

## 9. Feature Engineering and Feature Layer

### Chapter purpose

Document the final feature engineering pipeline and the local feature layer.

### Key points to cover later

- Explain the role of `data_feat_engineering`.
- Explain that the implemented feature layer is local and stored under `data/04_feature/`.
- Document the final production model features:
  - `simple_return`
  - `log_return`
  - `sma_10`
  - `sma_20`
  - `sma_ratio_10`
  - `rsi_14`
  - `volatility_10`
  - `volume_change`
- Mention that broader feature ideas such as MACD, calendar features, lagged returns, and longer rolling windows were considered earlier but are not part of the final implemented model pipeline.
- Explain the target column `target_next_day_up`.

### Expected figures, tables, or screenshots

- Feature definition table.
- Feature dataset shape and date range table.
- Small preview of `sp500_feature_data.csv`.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/data_feat_engineering/`
- `data/04_feature/sp500_feature_data.csv`
- `notebooks/01_final_project_walkthrough.ipynb`
- `conf/base/catalog.yml`

### Placeholder

To be written.

---

## 10. Train/Validation/Test Split

### Chapter purpose

Explain how the project splits time-series data without leakage.

### Key points to cover later

- Explain chronological splitting.
- Explain why the data must not be shuffled.
- Document train, validation, and test datasets generated by Kedro.
- Use actual split sizes and date ranges from the generated artifacts.
- Mention that the final test period is kept as the newest period.

### Expected figures, tables, or screenshots

- Split timeline diagram.
- Table with rows and date ranges for train, validation, and test.
- Dataset artifact table for `X_*`, `y_*`, and `dates_*`.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/data_split/`
- `data/05_model_input/X_train.csv`
- `data/05_model_input/y_train.csv`
- `data/05_model_input/dates_train.csv`
- `data/05_model_input/X_val.csv`
- `data/05_model_input/y_val.csv`
- `data/05_model_input/dates_val.csv`
- `data/05_model_input/X_test.csv`
- `data/05_model_input/y_test.csv`
- `data/05_model_input/dates_test.csv`
- `notebooks/01_final_project_walkthrough.ipynb`

### Placeholder

To be written.

---

## 11. Model Training

### Chapter purpose

Document how the final champion and challenger models are trained.

### Key points to cover later

- Explain that Logistic Regression is the baseline and champion model.
- Explain that Random Forest is the challenger model.
- Document the `model_train` pipeline.
- Document the `model_train_challenger` pipeline.
- Explain model inputs from the train/validation/test split.
- Describe preprocessing only as implemented in the model training nodes.
- Mention model artifacts saved under `data/06_models/`.

### Expected figures, tables, or screenshots

- Model comparison setup table.
- Training input/output table.
- Optional MLflow runs screenshot.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/model_train/`
- `src/sp500_mlops_pipeline/pipelines/model_train_challenger/`
- `conf/base/parameters_model.yml`
- `data/06_models/logistic_regression_baseline.pkl`
- `data/06_models/random_forest_challenger.pkl`
- `conf/base/catalog.yml`

### Placeholder

To be written.

---

## 12. Model Evaluation

### Chapter purpose

Explain how the trained models are evaluated and compared.

### Key points to cover later

- Document validation and test metrics.
- Explain the prediction pipelines:
  - `model_predict`
  - `model_predict_challenger`
- Compare Logistic Regression and Random Forest.
- Explain why Logistic Regression was selected as champion.
- Mention that ROC-AUC close to 0.50 is a limitation of the current feature signal.

### Expected figures, tables, or screenshots

- Test metrics table.
- Validation metrics table if needed.
- Confusion matrix if available.
- Model comparison table.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/model_predict/`
- `src/sp500_mlops_pipeline/pipelines/model_predict_challenger/`
- `data/07_model_output/test_metrics.json`
- `data/07_model_output/random_forest_test_metrics.json`
- `data/07_model_output/test_predictions.csv`
- `data/07_model_output/random_forest_test_predictions.csv`
- `README.md`

### Placeholder

To be written.

---

## 13. MLflow Tracking and Model Registry

### Chapter purpose

Document how experiments and model versions are tracked.

### Key points to cover later

- Explain MLflow tracking in the training pipelines.
- Document the registered model name:
  - `sp500_direction_model`
- Document the registry alias:
  - `candidate_champion`
- Explain how the serving layer depends on the registered model alias.
- Mention what is logged only after confirming from the implemented training nodes.

### Expected figures, tables, or screenshots

- MLflow experiment screenshot.
- Registered model page screenshot.
- Table mapping runs, models, and artifacts.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/model_train/`
- `src/sp500_mlops_pipeline/pipelines/model_train_challenger/`
- `src/sp500_mlops_pipeline/serving/model_loader.py`
- `conf/base/parameters_model.yml`
- Local MLflow tracking database/artifacts if available.

### Placeholder

To be written.

---

## 14. Explainability with SHAP

### Chapter purpose

Explain how the champion model is interpreted with SHAP.

### Key points to cover later

- Explain the role of the `model_explainability` pipeline.
- Mention that SHAP is used for the Logistic Regression champion model.
- Describe global feature importance.
- Explain the summary plot and bar plot at a high level.
- Use interpretations from the generated explainability summary, not guesses.

### Expected figures, tables, or screenshots

- SHAP summary plot.
- SHAP feature importance bar plot.
- Table of top features from the explainability summary JSON.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/model_explainability/`
- `data/08_reporting/shap/logistic_regression_v2_summary_plot.png`
- `data/08_reporting/shap/logistic_regression_v2_feature_importance_bar.png`
- `data/08_reporting/shap/logistic_regression_v2_explainability_summary.json`
- `data/08_reporting/shap/logistic_regression_v2_global_feature_importance.csv`

### Placeholder

To be written.

---

## 15. Data Drift Monitoring with Evidently

### Chapter purpose

Document the implemented Evidently data drift monitoring pipeline.

### Key points to cover later

- Explain the role of the `data_drift` pipeline.
- Explain the reference/current split:
  - reference data = oldest 70%
  - current data = newest 30%
- Explain that the monitored features are the final model features.
- Document drift outputs:
  - `data/08_reporting/drift/data_drift_report.html`
  - `data/08_reporting/drift/data_drift_summary.json`
- Mention that drift monitoring informs review or retraining decisions, but does not automatically retrain the model in the current implementation.

### Expected figures, tables, or screenshots

- Evidently HTML report screenshot.
- Drift summary table.
- Reference/current split diagram.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/pipelines/data_drift/`
- `data/08_reporting/drift/data_drift_report.html`
- `data/08_reporting/drift/data_drift_summary.json`
- `notebooks/01_final_project_walkthrough.ipynb`

### Placeholder

To be written.

---

## 16. FastAPI Serving Layer

### Chapter purpose

Explain how the selected model is exposed through the API.

### Key points to cover later

- Explain the FastAPI application.
- Document the implemented endpoints:
  - `GET /health`
  - `GET /model-info`
  - `POST /predict`
- Explain input feature validation.
- Explain that the API loads the model from MLflow using the selected registry alias.
- Include a sample request/response only after checking the current schema implementation.

### Expected figures, tables, or screenshots

- Swagger UI screenshot.
- Endpoint summary table.
- Example JSON request/response table.

### Source files or artifacts to use

- `src/sp500_mlops_pipeline/serving/app.py`
- `src/sp500_mlops_pipeline/serving/model_loader.py`
- `src/sp500_mlops_pipeline/serving/schemas.py`
- `README.md`

### Placeholder

To be written.

---

## 17. Docker Deployment

### Chapter purpose

Document how the API can be packaged and run in Docker.

### Key points to cover later

- Explain the Docker image:
  - `sp500-direction-api`
- Document the Docker build command.
- Document the Docker run command.
- Explain the dependency on the MLflow tracking server when running the container.
- Mention Windows host considerations only if they are part of the final README/run instructions.

### Expected figures, tables, or screenshots

- Docker build/run command table.
- Optional Docker Desktop screenshot.
- API health check screenshot from containerized run.

### Source files or artifacts to use

- `Dockerfile`
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `README.md`

### Placeholder

To be written.

---

## 18. Testing and Reproducibility

### Chapter purpose

Explain how the project is tested and how another user can reproduce the main workflow.

### Key points to cover later

- Document pytest validation.
- Mention final pytest result:
  - `81 passed, 11 warnings`
- Mention that `kedro run` completed successfully.
- Explain the expected setup from project root.
- Include commands for running the full workflow and selected pipelines.
- Mention Kedro-Viz was used to inspect the pipeline graph.

### Expected figures, tables, or screenshots

- pytest terminal output screenshot.
- Kedro run output screenshot.
- Kedro-Viz screenshot.
- Reproducibility checklist.

### Source files or artifacts to use

- `tests/`
- `README.md`
- `notebooks/01_final_project_walkthrough.ipynb`
- `docs/report_evidence.md`
- Kedro-Viz screenshot if available.

### Placeholder

To be written.

---

## 19. Experimental Results

### Chapter purpose

Summarize the final model comparison and selected champion model.

### Key points to cover later

- Present Logistic Regression test metrics:
  - Accuracy: `0.5363`
  - Precision: `0.5422`
  - Recall: `0.9218`
  - F1-score: `0.6828`
  - ROC-AUC: `0.4991`
- Present Random Forest test metrics:
  - Accuracy: `0.5030`
  - Precision: `0.5474`
  - Recall: `0.4730`
  - F1-score: `0.5075`
  - ROC-AUC: `0.5020`
- Explain why Logistic Regression was selected as champion.
- Be explicit that both ROC-AUC values are close to 0.50.
- Keep this chapter focused on model performance, not infrastructure validation.

### Expected figures, tables, or screenshots

- Final model metrics table.
- Champion/challenger comparison table.
- Optional predictions artifact preview.

### Source files or artifacts to use

- `README.md`
- `data/07_model_output/test_metrics.json`
- `data/07_model_output/random_forest_test_metrics.json`
- `notebooks/01_final_project_walkthrough.ipynb`
- `docs/model_card.md`

### Placeholder

To be written.

---

## 20. MLOps Results

### Chapter purpose

Summarize the implemented MLOps outcomes separately from model performance.

### Key points to cover later

- Confirm that the final implementation uses modular Kedro pipelines.
- Mention that MLflow tracking and model registry are used.
- Mention that SHAP explainability artifacts are generated.
- Mention that Great Expectations validation reports are generated.
- Mention that Evidently drift monitoring artifacts are generated.
- Mention that FastAPI serving and Docker deployment are implemented.
- Mention the final pytest validation result:
  - `81 passed, 11 warnings`
- Mention that `kedro run` completed successfully.
- Mention that Kedro-Viz was used to inspect the pipeline graph.

### Expected figures, tables, or screenshots

- MLOps component checklist.
- Kedro-Viz screenshot.
- MLflow registry screenshot.
- API Swagger screenshot.
- pytest output screenshot.

### Source files or artifacts to use

- `README.md`
- `notebooks/01_final_project_walkthrough.ipynb`
- `docs/report_evidence.md`
- `src/sp500_mlops_pipeline/pipeline_registry.py`
- `data/08_reporting/`
- `tests/`

### Placeholder

To be written.

---

## 21. Limitations

### Chapter purpose

Document the known technical and modelling limitations honestly.

### Key points to cover later

- The current predictive signal is weak.
- ROC-AUC is close to random for both final models.
- The feature set is intentionally compact.
- Market direction is noisy and difficult to predict from OHLCV features alone.
- The implementation does not automatically retrain on drift.
- The project is an academic MLOps implementation, not a production trading system.

### Expected figures, tables, or screenshots

- Limitations table with impact and possible mitigation.
- Optional ROC-AUC comparison table.

### Source files or artifacts to use

- `README.md`
- `docs/model_card.md`
- `docs/production_notes.md`
- `data/07_model_output/test_metrics.json`
- `data/07_model_output/random_forest_test_metrics.json`

### Placeholder

To be written.

---

## 22. Future Work

### Chapter purpose

List realistic improvements without presenting them as implemented functionality.

### Key points to cover later

- Improve feature signal with additional validated market features.
- Consider external data only if it is added through a reproducible pipeline.
- Evaluate stronger models such as XGBoost or LightGBM as future work, not current implementation.
- Add scheduled monitoring and retraining triggers.
- Extend API monitoring and logging.
- Improve model evaluation with additional time-window or walk-forward validation if implemented later.

### Expected figures, tables, or screenshots

- Future work roadmap table.
- Priority/effort matrix.

### Source files or artifacts to use

- `README.md`
- `docs/production_notes.md`
- `docs/model_card.md`

### Placeholder

To be written.

---

## 23. Appendix

### Chapter purpose

Collect reference material that supports the main manual without interrupting the narrative.

### Key points to cover later

- Full data catalog artifact mapping.
- Full pipeline registry listing.
- Important commands.
- Environment and dependency notes.
- API schema examples.
- Test list.
- Links or references to generated reports.

### Expected figures, tables, or screenshots

- Data catalog table.
- Pipeline registry table.
- Command reference table.
- Artifact reference table.
- Test file list.

### Source files or artifacts to use

- `conf/base/catalog.yml`
- `src/sp500_mlops_pipeline/pipeline_registry.py`
- `README.md`
- `requirements.txt`
- `tests/`
- `data/08_reporting/`

### Placeholder

To be written.

---

## Outline Quality Review

### 1. Chapters that may be too large and should be considered for splitting

- **Chapter 6. Kedro Pipeline Architecture**
  - This chapter may become large because it covers the full registry, the default workflow, and the additional reporting/monitoring pipelines.
  - Recommendation: if the written chapter becomes too long, split it into:
    - `Kedro Pipeline Registry`
    - `Default Workflow`
    - `Reporting and Monitoring Pipelines`

- **Chapter 9. Feature Engineering and Feature Layer**
  - This chapter combines feature generation, target creation, final feature selection, and the feature layer storage.
  - Recommendation: if needed, split into:
    - `Feature Engineering`
    - `Feature Layer and Model Inputs`

- **Chapter 11. Model Training**
  - This chapter covers both the champion and challenger model training flows.
  - Recommendation: keep it as one chapter if the explanation stays concise. If implementation details are expanded, split into:
    - `Champion Model Training`
    - `Challenger Model Training`

- **Chapter 18. Testing and Reproducibility**
  - This chapter combines pytest validation, Kedro reproducibility, setup commands, and Kedro-Viz evidence.
  - Recommendation: if the manual becomes more operational, split into:
    - `Testing`
    - `Reproducibility and Execution`

- **Chapter 20. MLOps Results**
  - This chapter may collect evidence from many parts of the project: Kedro, MLflow, SHAP, Great Expectations, Evidently, FastAPI, Docker, and pytest.
  - Recommendation: keep it as a summary chapter, but avoid repeating full details already covered in chapters 6-18.

### 2. Chapters that may be too small and could be merged

- **Chapter 1. Introduction** and **Chapter 3. Project Objectives**
  - These may overlap if both are written briefly.
  - Recommendation: keep both only if the Introduction explains the manual scope and Project Objectives clearly separates modelling objectives from MLOps objectives.

- **Chapter 3. Project Objectives** and **Chapter 4. Problem Definition**
  - These are related but not identical.
  - Recommendation: merge only if the final manual needs to be shorter. For a 20-40 page manual, keeping them separate is acceptable.

- **Chapter 17. Docker Deployment** and **Chapter 16. FastAPI Serving Layer**
  - Docker exists mainly to package and run the FastAPI serving layer.
  - Recommendation: keep them separate if screenshots and run commands are included. Merge them if the Docker section remains only a short command reference.

- **Chapter 21. Limitations** and **Chapter 22. Future Work**
  - These naturally connect because future work addresses current limitations.
  - Recommendation: keep separate for clarity, but cross-reference them when writing.

### 3. Missing figures to consider adding

- End-to-end MLOps workflow diagram from data input to API serving.
- Kedro default workflow diagram showing:
  - `data_quality`
  - `data_cleaning`
  - `data_feat_engineering`
  - `data_split`
  - `model_train`
  - `model_train_challenger`
  - `model_predict`
  - `model_predict_challenger`
- Diagram showing additional reporting and monitoring pipelines:
  - `data_expectations`
  - `model_explainability`
  - `data_drift`
- Chronological train/validation/test split timeline.
- Feature layer flow diagram from cleaned data to `data/04_feature/sp500_feature_data.csv`.
- MLflow model lifecycle diagram from training run to registered model alias.
- Serving architecture diagram showing FastAPI, MLflow registry, and Docker container.
- Drift monitoring diagram showing oldest 70% reference data versus newest 30% current data.

### 4. Missing tables to consider adding

- Data catalog table mapping Kedro dataset names to file paths from `conf/base/catalog.yml`.
- Pipeline registry table mapping each pipeline name to its purpose and main outputs.
- Feature definition table for the final eight model features.
- Train/validation/test split table with row counts and date ranges.
- Model training configuration table for Logistic Regression and Random Forest.
- Model artifact table covering files under `data/06_models/`.
- Prediction and metrics artifact table covering files under `data/07_model_output/`.
- MLflow registry table showing model name, alias, and serving use.
- SHAP artifact table listing generated plots, CSV, and JSON summary.
- Drift artifact table listing the HTML report and JSON summary.
- API endpoint table for `/health`, `/model-info`, and `/predict`.
- Docker command table for build and run steps.
- Test coverage table mapping test files to project components.
- Final MLOps evidence checklist.

### 5. Missing screenshots to consider adding

- Kedro-Viz graph screenshot for the implemented pipeline.
- MLflow experiment runs screenshot.
- MLflow registered model screenshot showing `sp500_direction_model`.
- MLflow alias screenshot showing `candidate_champion`, if available.
- SHAP summary plot screenshot or embedded figure.
- SHAP feature importance bar plot screenshot or embedded figure.
- Great Expectations validation report screenshot or JSON excerpt.
- Evidently drift report screenshot from `data/08_reporting/drift/data_drift_report.html`.
- FastAPI Swagger UI screenshot.
- `/health` endpoint response screenshot.
- `/model-info` endpoint response screenshot.
- `/predict` request/response screenshot.
- Docker container running screenshot or terminal output.
- pytest terminal output showing `81 passed, 11 warnings`.
- successful `kedro run` terminal output.

### 6. Missing implementation evidence to collect before writing the full manual

- Exact node-level implementation details for:
  - `data_quality`
  - `data_cleaning`
  - `data_feat_engineering`
  - `data_split`
  - `model_train`
  - `model_train_challenger`
  - `model_predict`
  - `model_predict_challenger`
  - `model_explainability`
  - `data_expectations`
  - `data_drift`
- Final row counts and date ranges for:
  - feature dataset
  - train split
  - validation split
  - test split
- Exact model hyperparameters from `conf/base/parameters_model.yml` and training nodes.
- Confirmation of what each training pipeline logs to MLflow.
- Confirmation of the current registered model version and alias state in MLflow.
- Exact schema for the FastAPI `/predict` request and response from `src/sp500_mlops_pipeline/serving/schemas.py`.
- Confirmation that Docker commands still run from a clean environment.
- The final pytest command used to obtain `81 passed, 11 warnings`.
- The final `kedro run` command output or log excerpt.
- Evidence that the drift summary matches the implemented oldest 70% / newest 30% split.
- Evidence that `docs/CODEX_DOCUMENTATION_GUIDE.md` has no additional content to incorporate, or update the manual sources if that guide is later populated.

### Overall recommendation

The outline is already suitable for a 20-40 page technical manual. The main risk is repetition across the Kedro, MLOps Results, Testing, and Appendix chapters. When writing the full manual, keep chapters 6-18 as the detailed implementation sections, use chapter 20 as a compact evidence summary, and leave command/reference material for the appendix.
