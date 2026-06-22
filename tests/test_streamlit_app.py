import importlib
from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_streamlit_app_module_can_be_imported() -> None:
    module = importlib.import_module("streamlit_app")

    assert callable(module.load_json_file)
    assert callable(module.load_csv_file)


def test_json_and_csv_helpers_load_valid_files(tmp_path: Path) -> None:
    from streamlit_app import load_csv_file, load_json_file

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "predictions.csv"
    json_path.write_text(json.dumps({"success": True}), encoding="utf-8")
    pd.DataFrame({"value": [1, 2]}).to_csv(csv_path, index=False)

    assert load_json_file(json_path) == {"success": True}
    loaded_csv = load_csv_file(csv_path)
    assert loaded_csv is not None
    pd.testing.assert_frame_equal(loaded_csv, pd.DataFrame({"value": [1, 2]}))


def test_json_and_csv_helpers_handle_missing_files(tmp_path: Path) -> None:
    from streamlit_app import load_csv_file, load_json_file

    assert load_json_file(tmp_path / "missing.json") is None
    assert load_csv_file(tmp_path / "missing.csv") is None


def test_shap_artifact_helper_loads_existing_files(tmp_path: Path) -> None:
    from streamlit_app import load_shap_artifacts

    summary_path = tmp_path / "summary.json"
    importance_path = tmp_path / "importance.csv"
    importance_bar_path = tmp_path / "importance.png"
    summary_plot_path = tmp_path / "summary.png"
    summary_path.write_text(
        json.dumps({"model_name": "sp500_direction_model"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature": ["volatility_10"],
            "mean_abs_shap_value": [0.03],
            "rank": [1],
        }
    ).to_csv(importance_path, index=False)
    importance_bar_path.write_bytes(b"image")
    summary_plot_path.write_bytes(b"image")

    artifacts = load_shap_artifacts(
        summary_path,
        importance_path,
        importance_bar_path,
        summary_plot_path,
    )

    assert artifacts["summary"]["model_name"] == "sp500_direction_model"
    assert artifacts["importance"] is not None
    assert artifacts["importance"]["feature"].tolist() == ["volatility_10"]
    assert artifacts["importance_bar_path"] == importance_bar_path
    assert artifacts["summary_plot_path"] == summary_plot_path


def test_shap_artifact_helper_handles_missing_files(tmp_path: Path) -> None:
    from streamlit_app import load_shap_artifacts

    artifacts = load_shap_artifacts(
        tmp_path / "missing-summary.json",
        tmp_path / "missing-importance.csv",
        tmp_path / "missing-importance.png",
        tmp_path / "missing-summary.png",
    )

    assert artifacts == {
        "summary": None,
        "importance": None,
        "importance_bar_path": None,
        "summary_plot_path": None,
    }
