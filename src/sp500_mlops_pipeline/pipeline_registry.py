"""Project pipeline registry."""

from kedro.pipeline import Pipeline

from sp500_mlops_pipeline.pipelines.data_ingestion.pipeline import (
    create_pipeline as create_data_ingestion_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_cleaning.pipeline import (
    create_pipeline as create_data_cleaning_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_feat_engineering.pipeline import (
    create_pipeline as create_data_feat_engineering_pipeline,
)
from sp500_mlops_pipeline.pipelines.feature_store.pipeline import (
    create_pipeline as create_feature_store_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_expectations.pipeline import (
    create_pipeline as create_data_expectations_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_drift.pipeline import (
    create_pipeline as create_data_drift_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_quality.pipeline import (
    create_pipeline as create_data_quality_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_reporting.pipeline import (
    create_pipeline as create_data_reporting_pipeline,
)
from sp500_mlops_pipeline.pipelines.data_split.pipeline import (
    create_pipeline as create_data_split_pipeline,
)
from sp500_mlops_pipeline.pipelines.model_train.pipeline import (
    create_pipeline as create_model_train_pipeline,
)
from sp500_mlops_pipeline.pipelines.model_train_challenger.pipeline import (
    create_pipeline as create_model_train_challenger_pipeline,
)
from sp500_mlops_pipeline.pipelines.model_predict.pipeline import (
    create_pipeline as create_model_predict_pipeline,
)
from sp500_mlops_pipeline.pipelines.model_predict_challenger.pipeline import (
    create_pipeline as create_model_predict_challenger_pipeline,
)
from sp500_mlops_pipeline.pipelines.model_reporting.pipeline import (
    create_pipeline as create_model_reporting_pipeline,
)
from sp500_mlops_pipeline.pipelines.model_explainability.pipeline import (
    create_pipeline as create_model_explainability_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the data preparation pipelines."""
    data_ingestion_pipeline = create_data_ingestion_pipeline()
    data_quality_pipeline = create_data_quality_pipeline()
    data_cleaning_pipeline = create_data_cleaning_pipeline()
    data_feat_engineering_pipeline = create_data_feat_engineering_pipeline()
    feature_store_pipeline = create_feature_store_pipeline()
    data_expectations_pipeline = create_data_expectations_pipeline()
    data_drift_pipeline = create_data_drift_pipeline()
    data_reporting_pipeline = create_data_reporting_pipeline()
    data_split_pipeline = create_data_split_pipeline()
    model_train_pipeline = create_model_train_pipeline()
    model_train_challenger_pipeline = create_model_train_challenger_pipeline()
    model_predict_pipeline = create_model_predict_pipeline()
    model_predict_challenger_pipeline = create_model_predict_challenger_pipeline()
    model_reporting_pipeline = create_model_reporting_pipeline()
    model_explainability_pipeline = create_model_explainability_pipeline()

    return {
        "data_ingestion": data_ingestion_pipeline,
        "data_quality": data_quality_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "data_feat_engineering": data_feat_engineering_pipeline,
        "feature_store": feature_store_pipeline,
        "data_expectations": data_expectations_pipeline,
        "data_drift": data_drift_pipeline,
        "data_reporting": data_reporting_pipeline,
        "data_split": data_split_pipeline,
        "model_train": model_train_pipeline,
        "model_train_challenger": model_train_challenger_pipeline,
        "model_predict": model_predict_pipeline,
        "model_predict_challenger": model_predict_challenger_pipeline,
        "model_reporting": model_reporting_pipeline,
        "model_explainability": model_explainability_pipeline,
        "__default__": (
            data_quality_pipeline
            + data_cleaning_pipeline
            + data_feat_engineering_pipeline
            + data_split_pipeline
            + model_train_pipeline
            + model_train_challenger_pipeline
            + model_predict_pipeline
            + model_predict_challenger_pipeline
        ),
    }
