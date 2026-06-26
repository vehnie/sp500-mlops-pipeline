"""Standalone Hopsworks offline feature-store pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import upload_sp500_features_to_hopsworks


def create_pipeline(**kwargs) -> Pipeline:
    """Create the optional Hopsworks offline feature-store pipeline.

    The default Kedro workflow remains local and reproducible from persisted CSVs.
    This pipeline is only run explicitly when backfilling Hopsworks.
    """
    return pipeline(
        [
            node(
                func=upload_sp500_features_to_hopsworks,
                inputs={
                    "feature_data": "sp500_feature_data",
                    "feature_group_name": "params:feature_store.feature_group_name",
                    "feature_group_version": "params:feature_store.feature_group_version",
                    "feature_view_name": "params:feature_store.feature_view_name",
                    "feature_view_version": "params:feature_store.feature_view_version",
                    "env_path": "params:feature_store.env_path",
                    "wait_for_job": "params:feature_store.wait_for_job",
                },
                outputs="feature_store_upload_summary",
                name="upload_sp500_features_to_hopsworks_node",
            )
        ]
    )
