"""Great Expectations contracts for raw market and feature datasets."""

from datetime import datetime, timezone
from uuid import uuid4

import great_expectations as gx
import great_expectations.expectations as gxe
import numpy as np
import pandas as pd
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.validation_definition import ValidationDefinition

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from sp500_mlops_pipeline.pipelines.data_quality.nodes import (
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
)


RAW_VALIDATION_NAME = "raw_market_data_contract"
FEATURE_VALIDATION_NAME = "feature_dataset_contract"
RAW_DATASET_NAME = "sp500_yahoo_finance_raw"
FEATURE_DATASET_NAME = "sp500_feature_data"


def validate_raw_market_data_expectations(data: pd.DataFrame):
    """Run the Great Expectations contract for raw OHLCV market data."""
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=REQUIRED_COLUMNS,
            exact_match=False,
        ),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        gxe.ExpectColumnValuesToNotBeNull(column="Date"),
        gxe.ExpectColumnValuesToBeUnique(column="Date"),
    ]

    for column in PRICE_COLUMNS:
        expectations.extend(
            [
                gxe.ExpectColumnValuesToNotBeNull(column=column),
                gxe.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=0,
                    strict_min=True,
                ),
            ]
        )

    expectations.extend(
        [
            gxe.ExpectColumnValuesToBeBetween(
                column="Volume",
                min_value=0,
            ),
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="Open",
                column_B="Low",
                or_equal=True,
            ),
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="High",
                column_B="Open",
                or_equal=True,
            ),
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="Close",
                column_B="Low",
                or_equal=True,
            ),
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="High",
                column_B="Close",
                or_equal=True,
            ),
        ]
    )

    return _run_in_memory_validation(
        data=data,
        validation_name=RAW_VALIDATION_NAME,
        expectations=expectations,
    )


def validate_feature_data_expectations(data: pd.DataFrame):
    """Run the Great Expectations contract for the engineered feature data."""
    required_columns = ["Date", *MARKET_FEATURE_COLUMNS, TARGET_COLUMN]
    validation_data = data.copy()
    if "Date" in validation_data.columns:
        parsed_dates = pd.to_datetime(validation_data["Date"], errors="coerce")
        validation_data["Date"] = parsed_dates.dt.strftime("%Y-%m-%d")
        validation_data["__date_order"] = parsed_dates.astype("int64")

    expectations = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=required_columns,
            exact_match=False,
        ),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        gxe.ExpectColumnValuesToNotBeNull(column="Date"),
        gxe.ExpectColumnValuesToBeUnique(column="Date"),
        gxe.ExpectColumnValuesToBeIncreasing(
            column="__date_order",
            strictly=False,
        ),
    ]

    for column in [*MARKET_FEATURE_COLUMNS, TARGET_COLUMN]:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=column))

    for column in MARKET_FEATURE_COLUMNS:
        expectations.append(
            gxe.ExpectColumnValuesToNotBeInSet(
                column=column,
                value_set=[np.inf, -np.inf],
            )
        )

    expectations.extend(
        [
            gxe.ExpectColumnValuesToBeInSet(
                column=TARGET_COLUMN,
                value_set=[0, 1],
            ),
            gxe.ExpectColumnValuesToBeBetween(column="sma_10", min_value=0),
            gxe.ExpectColumnValuesToBeBetween(column="sma_20", min_value=0),
            gxe.ExpectColumnValuesToBeBetween(
                column="volatility_10",
                min_value=0,
            ),
            gxe.ExpectColumnValuesToBeBetween(
                column="rsi_14",
                min_value=0,
                max_value=100,
            ),
        ]
    )

    return _run_in_memory_validation(
        data=validation_data,
        validation_name=FEATURE_VALIDATION_NAME,
        expectations=expectations,
    )


def create_raw_validation_report(validation_result) -> dict:
    """Create the serializable raw-data validation artefact."""
    return create_validation_report(
        validation_result=validation_result,
        validation_name=RAW_VALIDATION_NAME,
        dataset_name=RAW_DATASET_NAME,
    )


def create_feature_validation_report(validation_result) -> dict:
    """Create the serializable feature-data validation artefact."""
    return create_validation_report(
        validation_result=validation_result,
        validation_name=FEATURE_VALIDATION_NAME,
        dataset_name=FEATURE_DATASET_NAME,
    )


def create_validation_report(
    validation_result,
    validation_name: str,
    dataset_name: str,
) -> dict:
    """Transform a Great Expectations result into a compact JSON report."""
    failed_results = [
        result for result in validation_result.results if not result.success
    ]
    failed_expectation_types = [
        result.expectation_config.type for result in failed_results
    ]
    failed_expectation_details = [
        {
            "expectation_type": result.expectation_config.type,
            "column": result.expectation_config.kwargs.get("column"),
        }
        for result in failed_results
    ]
    total_expectations = len(validation_result.results)
    successful_expectations = total_expectations - len(failed_expectation_types)
    report = {
        "validation_name": validation_name,
        "success": bool(validation_result.success),
        "total_expectations": total_expectations,
        "successful_expectations": successful_expectations,
        "failed_expectations": len(failed_expectation_types),
        "failed_expectation_types": failed_expectation_types,
        "failed_expectation_details": failed_expectation_details,
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
    }

    return report


def enforce_validation_contracts(
    raw_report: dict,
    feature_report: dict,
) -> None:
    """Fail the pipeline after both validation artefacts have been saved."""
    failed_reports = [
        report for report in [raw_report, feature_report] if not report["success"]
    ]
    if not failed_reports:
        return

    failure_details = []
    for report in failed_reports:
        failed_rules = ", ".join(
            (
                f"{detail['expectation_type']} "
                f"(column={detail['column']})"
                if detail["column"]
                else detail["expectation_type"]
            )
            for detail in report["failed_expectation_details"]
        )
        failure_details.append(f"{report['dataset_name']}: {failed_rules}")

    raise ValueError(
        "One or more datasets do not satisfy the quality contract. "
        f"Failed expectations: {'; '.join(failure_details)}"
    )


def _run_in_memory_validation(
    data: pd.DataFrame,
    validation_name: str,
    expectations: list,
):
    """Execute a validation suite using an ephemeral in-memory context."""
    unique_suffix = uuid4().hex
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(
        name=f"{validation_name}_source_{unique_suffix}"
    )
    data_asset = data_source.add_dataframe_asset(
        name=f"{validation_name}_asset_{unique_suffix}"
    )
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name=f"{validation_name}_batch_{unique_suffix}"
    )

    suite = ExpectationSuite(name=f"{validation_name}_suite_{unique_suffix}")
    for expectation in expectations:
        suite.add_expectation(expectation)
    context.suites.add(suite)

    validation_definition = ValidationDefinition(
        name=f"{validation_name}_definition_{unique_suffix}",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add(validation_definition)

    return validation_definition.run(
        batch_parameters={"dataframe": data},
        result_format="SUMMARY",
    )
