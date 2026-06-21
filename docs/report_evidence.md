# Project Evidence Log

This document records verified project evidence for the final six-page report.

## 1. Use Case

The project aims to predict whether the S&P 500 will move up on the next
trading day. The binary prediction target is `target_next_day_up`.

## 2. Data Quality Pipeline

The first functional Kedro pipeline validates the raw S&P 500 market data
before downstream processing.

- Input: `data/01_raw/sp500_yahoo_finance_raw.csv`
- Output: `data/02_intermediate/sp500_validated_data.csv`
- Validation function: `validate_raw_market_data()`
- Checks performed:
  - required columns are present;
  - the dataset is not empty;
  - dates can be parsed correctly;
  - required columns contain no missing values;
  - dates contain no duplicates;
  - price values are positive;
  - volume values are non-negative.
- Result: validation completed successfully with 6,625 rows.
- Great Expectations raw-data contract: 19 of 19 expectations passed.

## 3. Data Cleaning Pipeline

The `data_cleaning` Kedro pipeline receives the validated S&P 500 dataset and
prepares a clean primary dataset for later feature engineering.

- Input: `data/02_intermediate/sp500_validated_data.csv`
- Output: `data/03_primary/sp500_cleaned_data.csv`
- Cleaning function: `clean_raw_market_data()`
- Steps performed:
  - keeps the required OHLCV columns;
  - converts `Date` to datetime;
  - converts price and volume columns to numeric values;
  - reuses the raw-data validation checks;
  - sorts the dataset chronologically by date.
- Result: pipeline executed successfully and created a cleaned dataset with
  6,625 rows and 7 columns.

## 4. Feature Engineering

The feature-engineering stage creates a compact and leakage-safe dataset for
next-day S&P 500 direction prediction.

- Input: `data/03_primary/sp500_cleaned_data.csv`
- Price source for features and target: `Close`
- Features created:
  - `simple_return`;
  - `log_return`;
  - `sma_10`;
  - `sma_20`;
  - `sma_ratio_10`;
  - `rsi_14`;
  - `volatility_10`;
  - `volume_change`.
- `volume_change` is calculated only when the previous trading-day volume is
  strictly positive. When the previous volume is zero, the value is treated as
  undefined and the affected row is removed.
- Target:
  - `target_next_day_up = 1` when `Close(t+1) > Close(t)`;
  - `target_next_day_up = 0` otherwise.
- Leakage control:
  - all features use only information available up to day `t`;
  - the target uses only the next trading-day close;
  - rows without sufficient rolling-window history or without a future target
    are removed.
- Result:
  - final feature dataset contains 6,604 rows;
  - all selected feature values are finite;
  - the Great Expectations feature-data contract passed 27 of 27
    expectations;
  - target values are only `0` and `1`;
  - target distribution: 3,054 observations with class `0` and 3,550
    observations with class `1`;
  - the target is reasonably balanced, with approximately 46.2% class `0` and
    53.8% class `1`.

## 5. Time-Based Data Split

The `data_split` Kedro pipeline divides the feature dataset chronologically to
preserve the temporal order required for financial forecasting.

- Input: `sp500_feature_data`
- Split strategy: 70% training, 15% validation, and 15% testing.
- Training set:
  - 4,622 rows;
  - date range: 2000-01-31 to 2018-06-13.
- Validation set:
  - 990 rows;
  - date range: 2018-06-14 to 2022-05-18.
- Test set:
  - 992 rows;
  - date range: 2022-05-19 to 2026-05-05.
- Temporal separation:
  - the last training date is earlier than the first validation date;
  - the last validation date is earlier than the first test date;
  - the three subsets have no temporal overlap.

## 6. Baseline Model Training

The `model_train` Kedro pipeline trains a binary classification baseline using
the chronologically separated training and validation datasets.

- Model: Logistic Regression.
- Preprocessing: `StandardScaler`.
- Training data: chronological `X_train` and `y_train` subsets.
- Validation metrics:
  - accuracy: 0.5727;
  - precision: 0.5685;
  - recall: 0.9284;
  - F1-score: 0.7052;
  - ROC-AUC: 0.5182.
- Saved model:
  `data/06_models/logistic_regression_baseline.pkl`
- Saved validation metrics:
  `data/07_model_output/validation_metrics.json`

## 7. Final Test Evaluation

The `model_predict` Kedro pipeline evaluates the saved baseline on the
chronologically separated final test period.

- Test data: `X_test`, `y_test`, and `dates_test`.
- Final test metrics:
  - accuracy: 0.5363;
  - precision: 0.5422;
  - recall: 0.9218;
  - F1-score: 0.6828;
  - ROC-AUC: 0.4991.
- Saved predictions:
  `data/07_model_output/test_predictions.csv`
- Saved test metrics:
  `data/07_model_output/test_metrics.json`
- The test set was not used to train the baseline model.

## 8. MLflow Experiment Tracking

The baseline training run is tracked by the local MLflow server to support
reproducibility and future model comparisons.

- Tracking server: `http://127.0.0.1:5000`
- Experiment: `sp500_direction_prediction`
- Run: `logistic_regression_baseline_v2`
- Run ID: `cbd515043e584a60acda01e0bc0353e3`
- Logged parameters:
  - model type;
  - random state;
  - maximum number of iterations.
- Logged validation metrics:
  - accuracy;
  - precision;
  - recall;
  - F1-score;
  - ROC-AUC.
- Logged metadata and artefacts:
  - dataset, split strategy, model family, and pipeline-stage tags;
  - feature list;
  - inferred model signature;
  - five-row input example;
  - trained sklearn pipeline.
- Purpose: make the baseline reproducible and provide a consistent basis for
  comparing future model candidates.

## 9. Automated Testing with Pytest

The automated test suite was updated to reflect the current
`data_feat_engineering` implementation.

- Files updated exclusively:
  - `tests/test_feature_engineering.py`;
  - `tests/test_data_split.py`.
- The tests were outdated because they imported functions and constants from
  an earlier version that no longer exist in the current feature-engineering
  implementation.
- No production files were changed as part of this correction.
- Pytest result:

```text
56 passed, 1 warning in 24.64s
```

- Test coverage:
  - verifies the eight features defined in `MARKET_FEATURE_COLUMNS`;
  - verifies the current binary target `target_next_day_up`;
  - verifies that the feature dataset is non-empty, chronologically ordered,
    and contains no missing values;
  - verifies that the target contains only `0` and `1`;
  - verifies that `Close` is used as the price source;
  - verifies the chronological 70/15/15 train-validation-test split;
  - verifies that the train, validation, and test periods do not overlap;
  - verifies split fractions and required input columns.
- Interpretation: The automated tests confirm that the critical
  feature-engineering transformations and the chronological data split remain
  consistent with the current implementation. This reduces the risk of
  regressions when challenger models, hyperparameter tuning, and additional
  MLOps components are added.
- Warning note: The single warning is associated with `pd.to_datetime` in an
  intentional invalid-date test case. No tests failed, and the warning does
  not block project execution.

## 10. Model Comparison and Provisional Selection

### Logistic Regression final test metrics

- Accuracy: 0.5363.
- Precision: 0.5422.
- Recall: 0.9218.
- F1-score: 0.6828.
- ROC-AUC: 0.4991.

### Random Forest validation metrics

- Accuracy: 0.5182.
- Precision: 0.5677.
- Recall: 0.5229.
- F1-score: 0.5444.
- ROC-AUC: 0.5273.
- MLflow run: `random_forest_challenger_v2`.
- Run ID: `bd563c2b29dd4446b4f46b65b41542c4`.

### Random Forest final test metrics

- Accuracy: 0.5030.
- Precision: 0.5474.
- Recall: 0.4730.
- F1-score: 0.5075.
- ROC-AUC: 0.5020.

### Random Forest test evaluation details

- Evaluated on 992 final test observations.
- The model was loaded from the persisted challenger artefact.
- No retraining, hyperparameter tuning, threshold adjustment, or model
  selection was performed using `X_test`.
- Predictions saved in:
  `data/07_model_output/random_forest_test_predictions.csv`
- Metrics saved in:
  `data/07_model_output/random_forest_test_metrics.json`
- MLflow run:
  `random_forest_challenger_test_evaluation_v2`
- Run ID:
  `0eec4203f2294f799d5ff21fb5aa1699`

### Comparison interpretation

- The Random Forest obtained a marginally higher ROC-AUC than Logistic
  Regression: 0.5020 versus 0.4991.
- However, both ROC-AUC values are close to random discrimination.
- Logistic Regression achieved higher accuracy, recall, and F1-score.
- The F1-score difference is substantial: Logistic Regression achieved 0.6828
  versus 0.5075 for Random Forest.
- Therefore, Logistic Regression is retained as the provisional champion
  candidate for the current project configuration.
- This is a provisional operational decision based primarily on F1-score and
  recall, not a claim that the model has strong predictive power.

## 11. MLflow Model Registry

The provisional model-selection decision is formalized in the MLflow Model
Registry without claiming that the selected model has strong predictive
power.

- Registered model: `sp500_direction_model`.
- Logistic Regression v2:
  - registered model version: 1;
  - source run: `logistic_regression_baseline_v2`;
  - source run ID: `cbd515043e584a60acda01e0bc0353e3`;
  - alias: `candidate_champion`;
  - selection is based primarily on its higher final-test F1-score and recall.
- Random Forest v2:
  - registered model version: 2;
  - source run: `random_forest_challenger_v2`;
  - source run ID: `bd563c2b29dd4446b4f46b65b41542c4`;
  - alias: `challenger`.
- The `champion` alias was deliberately not assigned at this stage.
- Both models have final-test ROC-AUC values close to random discrimination,
  so the registry status represents a provisional operational choice rather
  than evidence of strong predictive performance.

## 12. FastAPI Model Serving

The Logistic Regression v2 model is served through a local FastAPI
application using the MLflow Model Registry alias `candidate_champion`.

- Registered model URI:
  `models:/sp500_direction_model@candidate_champion`
- Serving endpoints:
  - `GET /health`;
  - `GET /model-info`;
  - `POST /predict`.
- The eight required feature values are validated with Pydantic.
- Missing, non-numeric, NaN, and infinite inputs are rejected with HTTP 422.
- The registered model is loaded once during API startup rather than once per
  prediction request.
- A minimal Docker image packages only the serving source and inference
  dependencies; datasets, local models, reports, and the MLflow database are
  excluded.
- Inside Docker, `MLFLOW_TRACKING_URI` points to the MLflow server running on
  the Windows host, typically `http://host.docker.internal:5000`.
- The MLflow server proxies model artifacts from
  `data/08_reporting/mlflow_artifacts` using `mlflow-artifacts:/` URIs. This
  prevents Docker clients from receiving host-only `file:///C:/...` paths.
- Existing artifact-location metadata was migrated in place with
  `scripts/migrate_mlflow_artifact_uris.py`. The migration preserves the
  physical model files, run IDs, registered model versions, metrics, and the
  `candidate_champion` alias.
- Before migration, registered version 1 had source
  `models:/m-09d8d2dd77f049b8a3a306241652cbc7`, while both its storage
  location and run `cbd515043e584a60acda01e0bc0353e3` artifact URI resolved
  to host-only `file:///C:/...` locations.
- After migration, the logged-model location is
  `mlflow-artifacts:/models/m-09d8d2dd77f049b8a3a306241652cbc7/artifacts`
  and the run artifact URI is
  `mlflow-artifacts:/cbd515043e584a60acda01e0bc0353e3/artifacts`.
- Host command used for Docker-compatible tracking and artifact serving:
  `mlflow server --host 0.0.0.0 --port 5000
  --allowed-hosts
  "localhost:*,127.0.0.1:*,host.docker.internal:5000"
  --backend-store-uri sqlite:///mlflow.db
  --default-artifact-root mlflow-artifacts:/ --serve-artifacts
  --artifacts-destination <file:///absolute/artifact/root>`.
- Docker validation rebuilt `sp500-direction-api`, started the container with
  `MLFLOW_TRACKING_URI=http://host.docker.internal:5000`, and observed HTTP
  200 artifact downloads for `MLmodel`, `model.skops`, and the remaining
  model files through the MLflow server.
- Final serving checks passed for `GET /health`, `GET /model-info`, and
  `POST /predict`. Model info reported version 1, alias
  `candidate_champion`, family `logistic_regression`, and eight expected
  features. The full test suite completed with `77 passed`.

## 13. Docker and Drift

## 14. Production Risks and Mitigations
