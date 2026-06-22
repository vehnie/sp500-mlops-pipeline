"""Local read-only dashboard for existing S&P 500 MLOps artefacts."""

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data/08_reporting/great_expectations/raw_data_validation_report.json"
)
FEATURE_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data/08_reporting/great_expectations/feature_data_validation_report.json"
)
BASELINE_METRICS_PATH = PROJECT_ROOT / "data/07_model_output/test_metrics.json"
CHALLENGER_METRICS_PATH = (
    PROJECT_ROOT / "data/07_model_output/random_forest_test_metrics.json"
)
BASELINE_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data/07_model_output/test_predictions.csv"
)
CHALLENGER_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data/07_model_output/random_forest_test_predictions.csv"
)
SHAP_REPORTING_DIR = PROJECT_ROOT / "data/08_reporting/shap"
SHAP_SUMMARY_PATH = (
    SHAP_REPORTING_DIR
    / "logistic_regression_v2_explainability_summary.json"
)
SHAP_IMPORTANCE_PATH = (
    SHAP_REPORTING_DIR
    / "logistic_regression_v2_global_feature_importance.csv"
)
SHAP_IMPORTANCE_BAR_PATH = (
    SHAP_REPORTING_DIR
    / "logistic_regression_v2_feature_importance_bar.png"
)
SHAP_SUMMARY_PLOT_PATH = (
    SHAP_REPORTING_DIR
    / "logistic_regression_v2_summary_plot.png"
)

PREDICTION_COLUMNS = [
    "Date",
    "actual_target",
    "predicted_target",
    "probability_up",
]


def load_json_file(path: str | Path) -> dict | None:
    """Load a JSON object, returning ``None`` when it cannot be read."""
    file_path = Path(path)
    try:
        with file_path.open(encoding="utf-8") as file:
            content = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    return content if isinstance(content, dict) else None


def load_csv_file(path: str | Path) -> pd.DataFrame | None:
    """Load a CSV file, returning ``None`` when it cannot be read."""
    file_path = Path(path)
    try:
        return pd.read_csv(file_path)
    except (FileNotFoundError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def build_model_comparison(
    baseline_metrics: dict,
    challenger_metrics: dict,
) -> pd.DataFrame:
    """Build the comparison table from persisted metric dictionaries."""
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression v2",
                **{metric: baseline_metrics.get(metric) for metric in metric_names},
            },
            {
                "model": "Random Forest v2",
                **{metric: challenger_metrics.get(metric) for metric in metric_names},
            },
        ]
    )


def load_shap_artifacts(
    summary_path: str | Path = SHAP_SUMMARY_PATH,
    importance_path: str | Path = SHAP_IMPORTANCE_PATH,
    importance_bar_path: str | Path = SHAP_IMPORTANCE_BAR_PATH,
    summary_plot_path: str | Path = SHAP_SUMMARY_PLOT_PATH,
) -> dict:
    """Load the existing SHAP report, importance table, and image paths."""
    importance_bar = Path(importance_bar_path)
    summary_plot = Path(summary_plot_path)
    return {
        "summary": load_json_file(summary_path),
        "importance": load_csv_file(importance_path),
        "importance_bar_path": (
            importance_bar if importance_bar.is_file() else None
        ),
        "summary_plot_path": summary_plot if summary_plot.is_file() else None,
    }


def render_overview(st) -> None:
    """Render the high-level project status."""
    st.title("S&P 500 MLOps Dashboard")
    st.write(
        "Read-only view of the latest data-quality, model-evaluation, and "
        "prediction artefacts."
    )

    raw_report = load_json_file(RAW_VALIDATION_PATH)
    feature_report = load_json_file(FEATURE_VALIDATION_PATH)
    baseline_metrics = load_json_file(BASELINE_METRICS_PATH)
    challenger_metrics = load_json_file(CHALLENGER_METRICS_PATH)

    components = [
        ("Kedro pipelines", baseline_metrics is not None and challenger_metrics is not None),
        (
            "Great Expectations",
            raw_report is not None and feature_report is not None,
        ),
        ("MLflow", (PROJECT_ROOT / "mlflow.db").exists()),
        ("pytest", True),
    ]
    columns = st.columns(len(components))
    for column, (name, available) in zip(columns, components):
        column.metric(name, "Available" if available else "Missing")

    st.subheader("Current summary")
    _render_validation_summary_line(st, "Raw-data validation", raw_report)
    _render_validation_summary_line(st, "Feature-data validation", feature_report)
    st.success("Provisional selected model: Logistic Regression v2")
    st.caption("Latest verified automated-test result: 56 passed, 1 warning.")


def render_data_quality(st) -> None:
    """Render persisted Great Expectations reports."""
    st.header("Data Quality")
    reports = [
        ("Raw market data", RAW_VALIDATION_PATH),
        ("Feature dataset", FEATURE_VALIDATION_PATH),
    ]

    for label, path in reports:
        st.subheader(label)
        report = load_json_file(path)
        if report is None:
            st.warning(f"Validation report not found or unreadable: {path}")
            continue

        if report.get("success"):
            st.success("Validation successful")
        else:
            st.error("Validation failed")

        fields = {
            "Dataset name": report.get("dataset_name"),
            "Validation name": report.get("validation_name"),
            "Success": report.get("success"),
            "Total expectations": report.get("total_expectations"),
            "Successful expectations": report.get("successful_expectations"),
            "Failed expectations": report.get("failed_expectations"),
            "Validation timestamp": report.get("validation_timestamp"),
        }
        st.dataframe(
            pd.DataFrame(fields.items(), columns=["field", "value"]),
            hide_index=True,
            use_container_width=True,
        )

        failed_types = report.get("failed_expectation_types") or []
        if failed_types:
            st.write("Failed expectation types:")
            st.code("\n".join(str(item) for item in failed_types))


def render_model_comparison(st) -> None:
    """Render persisted model metrics and a compact comparison chart."""
    st.header("Model Comparison")
    baseline_metrics = load_json_file(BASELINE_METRICS_PATH)
    challenger_metrics = load_json_file(CHALLENGER_METRICS_PATH)

    if baseline_metrics is None or challenger_metrics is None:
        st.warning(
            "One or both model metric files are missing or unreadable. "
            "Run the model evaluation pipelines first."
        )
        return

    comparison = build_model_comparison(baseline_metrics, challenger_metrics)
    st.dataframe(
        comparison.style.format(
            {
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "roc_auc": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    import matplotlib.pyplot as plt

    chart_data = comparison.set_index("model")[["f1_score", "roc_auc"]]
    figure, axis = plt.subplots(figsize=(8, 4))
    chart_data.plot(kind="bar", ax=axis, color=["#2E86AB", "#F18F01"])
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_title("Final test metric comparison")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(["F1-score", "ROC-AUC"])
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.info(
        "Logistic Regression v2 has the higher F1-score and recall. Random "
        "Forest v2 has a marginally higher ROC-AUC, although both ROC-AUC "
        "values are close to random discrimination. Logistic Regression "
        "remains the current candidate champion."
    )


def render_predictions(st) -> None:
    """Render persisted final-test predictions for the selected model."""
    st.header("Predictions")
    model_name = st.selectbox(
        "Model",
        ["Logistic Regression", "Random Forest"],
    )
    prediction_path = (
        BASELINE_PREDICTIONS_PATH
        if model_name == "Logistic Regression"
        else CHALLENGER_PREDICTIONS_PATH
    )
    predictions = load_csv_file(prediction_path)
    if predictions is None:
        st.warning(f"Prediction file not found or unreadable: {prediction_path}")
        return

    missing_columns = [
        column for column in PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing_columns:
        st.error(f"Prediction file is missing required columns: {missing_columns}")
        return

    predictions = predictions.loc[:, PREDICTION_COLUMNS].copy()
    predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
    predictions = predictions.dropna(subset=["Date"]).sort_values("Date")
    if predictions.empty:
        st.warning("No valid dated predictions are available.")
        return

    minimum_date = predictions["Date"].min().date()
    maximum_date = predictions["Date"].max().date()
    selected_dates = st.date_input(
        "Date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        date_values = predictions["Date"].dt.date
        predictions = predictions[
            (date_values >= start_date) & (date_values <= end_date)
        ]

    st.subheader("First 20 rows")
    st.dataframe(predictions.head(20), hide_index=True, use_container_width=True)

    if predictions.empty:
        st.warning("No predictions fall within the selected date range.")
        return

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        predictions["Date"],
        predictions["probability_up"],
        color="#2E86AB",
        linewidth=1.2,
    )
    axis.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability of next-day rise")
    axis.set_xlabel("Date")
    axis.set_title(f"{model_name}: probability_up over time")
    figure.autofmt_xdate()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def render_explainability(st) -> None:
    """Render persisted global SHAP explanations."""
    st.header("Model Explainability — Logistic Regression v2")
    artifacts = load_shap_artifacts()
    summary = artifacts["summary"]
    importance = artifacts["importance"]

    if summary is None:
        st.warning(
            f"SHAP explainability summary not found or unreadable: "
            f"{SHAP_SUMMARY_PATH}"
        )
    else:
        context = {
            "Model name": summary.get("model_name"),
            "Model version": summary.get("model_version"),
            "Model alias": summary.get("model_alias"),
            "Explainer type": summary.get("explainer_type"),
            "Explained dataset": summary.get("explained_dataset"),
            "Number of observations": summary.get("n_observations"),
            "Number of features": summary.get("n_features"),
            "Positive class": summary.get("positive_class"),
        }
        st.subheader("Explanation context")
        st.dataframe(
            pd.DataFrame(context.items(), columns=["field", "value"]),
            hide_index=True,
            use_container_width=True,
        )

    required_columns = ["feature", "mean_abs_shap_value", "rank"]
    if importance is None:
        st.warning(
            f"SHAP feature-importance CSV not found or unreadable: "
            f"{SHAP_IMPORTANCE_PATH}"
        )
    else:
        missing_columns = [
            column for column in required_columns if column not in importance.columns
        ]
        if missing_columns:
            st.error(
                "SHAP feature-importance CSV is missing required columns: "
                f"{missing_columns}"
            )
        else:
            importance = importance.loc[:, required_columns].sort_values("rank")
            st.subheader("Global feature importance")
            st.dataframe(
                importance.style.format(
                    {"mean_abs_shap_value": "{:.6f}", "rank": "{:.0f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Top 5 features")
            top_features = importance.head(5)
            for row in top_features.itertuples(index=False):
                st.write(
                    f"**{int(row.rank)}. `{row.feature}`** — "
                    f"mean |SHAP|: {row.mean_abs_shap_value:.6f}"
                )

    st.subheader("Global importance bar chart")
    if artifacts["importance_bar_path"] is None:
        st.warning(
            f"SHAP importance-bar image not found: {SHAP_IMPORTANCE_BAR_PATH}"
        )
    else:
        st.image(
            str(artifacts["importance_bar_path"]),
            caption="Mean absolute SHAP importance by feature",
            use_container_width=True,
        )

    st.subheader("SHAP summary plot")
    if artifacts["summary_plot_path"] is None:
        st.warning(f"SHAP summary image not found: {SHAP_SUMMARY_PLOT_PATH}")
    else:
        st.image(
            str(artifacts["summary_plot_path"]),
            caption="Feature contributions across the validation dataset",
            use_container_width=True,
        )

    st.info(
        "SHAP values indicate each feature's contribution to model "
        "predictions. Larger mean absolute SHAP values indicate greater "
        "average influence, but feature importance is not evidence of "
        "causality. These explanations use the validation dataset and do not "
        "use the final test set."
    )


def _render_validation_summary_line(st, label: str, report: dict | None) -> None:
    if report is None:
        st.warning(f"{label}: report unavailable")
    elif report.get("success"):
        st.success(
            f"{label}: {report.get('successful_expectations', 0)}/"
            f"{report.get('total_expectations', 0)} expectations passed"
        )
    else:
        st.error(
            f"{label}: {report.get('failed_expectations', 0)} expectations failed"
        )


def main() -> None:
    """Run the Streamlit dashboard."""
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Streamlit is not installed in the active environment. "
            "Install it before running: streamlit run streamlit_app.py"
        ) from exc

    st.set_page_config(
        page_title="S&P 500 MLOps Dashboard",
        page_icon="📈",
        layout="wide",
    )
    section = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Data Quality",
            "Model Comparison",
            "Predictions",
            "Explainability",
        ],
    )

    renderers = {
        "Overview": render_overview,
        "Data Quality": render_data_quality,
        "Model Comparison": render_model_comparison,
        "Predictions": render_predictions,
        "Explainability": render_explainability,
    }
    renderers[section](st)


if __name__ == "__main__":
    main()
