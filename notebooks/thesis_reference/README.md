# Thesis Reference Notebooks

These notebooks were created during the initial exploratory phase of the project.

They are kept as historical and thesis-oriented references, especially for:

- early market data ingestion;
- early feature engineering;
- dataset construction;
- sentiment/news exploration.

They are not the final production implementation of the MLOps project.

The final project implementation was refactored into Kedro pipelines under:

`src/sp500_mlops_pipeline/`

The final reproducible workflow is managed through:

- Kedro pipelines;
- MLflow tracking and model registry;
- SHAP explainability;
- Evidently data drift monitoring;
- FastAPI serving;
- Docker deployment;
- pytest validation.