"""Great Expectations data-contract validation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_feature_validation_report,
    create_feature_validation_html_report,
    create_raw_validation_report,
    create_raw_validation_html_report,
    enforce_validation_contracts,
    validate_feature_data_expectations,
    validate_raw_market_data_expectations,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create raw and feature dataset contract validations."""
    return pipeline(
        [
            node(
                func=validate_raw_market_data_expectations,
                inputs="sp500_yahoo_finance_raw",
                outputs="raw_expectation_validation_result",
                name="validate_raw_market_data_expectations_node",
            ),
            node(
                func=create_raw_validation_report,
                inputs="raw_expectation_validation_result",
                outputs="raw_data_validation_report",
                name="create_raw_validation_report_node",
            ),
            node(
                func=create_raw_validation_html_report,
                inputs="raw_data_validation_report",
                outputs="raw_data_validation_report_html",
                name="create_raw_validation_html_report_node",
            ),
            node(
                func=validate_feature_data_expectations,
                inputs="sp500_feature_data",
                outputs="feature_expectation_validation_result",
                name="validate_feature_data_expectations_node",
            ),
            node(
                func=create_feature_validation_report,
                inputs="feature_expectation_validation_result",
                outputs="feature_data_validation_report",
                name="create_feature_validation_report_node",
            ),
            node(
                func=create_feature_validation_html_report,
                inputs="feature_data_validation_report",
                outputs="feature_data_validation_report_html",
                name="create_feature_validation_html_report_node",
            ),
            node(
                func=enforce_validation_contracts,
                inputs=[
                    "raw_data_validation_report",
                    "feature_data_validation_report",
                ],
                outputs=None,
                name="enforce_validation_contracts_node",
            ),
        ]
    )
