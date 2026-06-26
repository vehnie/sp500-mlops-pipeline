from pathlib import Path
import sys
import types

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from sp500_mlops_pipeline.pipelines.feature_store.nodes import (
    DATE_COLUMN,
    connect_to_hopsworks,
    create_or_get_sp500_feature_group,
    create_or_get_sp500_feature_view,
    insert_sp500_feature_data,
    upload_sp500_features_to_hopsworks,
    validate_feature_store_dataframe,
    _normalize_hopsworks_host,
)


def make_feature_data() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    data = pd.DataFrame(
        {
            DATE_COLUMN: dates[::-1],
            TARGET_COLUMN: [1, 0, 1],
        }
    )
    for index, column in enumerate(MARKET_FEATURE_COLUMNS, start=1):
        data[column] = [float(index), float(index + 1), float(index + 2)]
    return data


def test_validate_feature_store_dataframe_requires_date_column() -> None:
    feature_data = make_feature_data().drop(columns=[DATE_COLUMN])

    with pytest.raises(ValueError, match="missing required canonical columns"):
        validate_feature_store_dataframe(feature_data)


def test_validate_feature_store_dataframe_converts_sorts_and_requires_unique_dates() -> None:
    feature_data = make_feature_data()

    result = validate_feature_store_dataframe(feature_data)

    assert pd.api.types.is_datetime64_any_dtype(result[DATE_COLUMN])
    assert result[DATE_COLUMN].is_monotonic_increasing
    assert result[DATE_COLUMN].is_unique


def test_validate_feature_store_dataframe_rejects_duplicate_dates() -> None:
    feature_data = make_feature_data()
    feature_data.loc[1, DATE_COLUMN] = feature_data.loc[0, DATE_COLUMN]

    with pytest.raises(ValueError, match="unique Date values"):
        validate_feature_store_dataframe(feature_data)


def test_validate_feature_store_dataframe_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty feature dataset"):
        validate_feature_store_dataframe(pd.DataFrame())


def test_validate_feature_store_dataframe_requires_canonical_features() -> None:
    feature_data = make_feature_data().drop(columns=[MARKET_FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing required canonical columns"):
        validate_feature_store_dataframe(feature_data)


def test_validate_feature_store_dataframe_requires_target() -> None:
    feature_data = make_feature_data().drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="missing required canonical columns"):
        validate_feature_store_dataframe(feature_data)


def test_target_is_excluded_from_model_feature_list() -> None:
    assert TARGET_COLUMN not in MARKET_FEATURE_COLUMNS


def test_connect_to_hopsworks_uses_environment_without_printing_secret(
    monkeypatch,
    capsys,
) -> None:
    login_calls = []
    fake_project = object()

    def fake_login(**kwargs):
        login_calls.append(kwargs)
        return fake_project

    fake_hopsworks = types.SimpleNamespace(login=fake_login)
    monkeypatch.setitem(sys.modules, "hopsworks", fake_hopsworks)
    monkeypatch.setenv("HOPSWORKS_API_KEY", "secret-value")
    monkeypatch.setenv("HOPSWORKS_PROJECT", "sp500_mlops_feature_store")
    monkeypatch.setenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    result = connect_to_hopsworks(env_path="missing-test-env-file")

    captured = capsys.readouterr()
    assert result is fake_project
    assert login_calls == [
        {
            "api_key_value": "secret-value",
            "project": "sp500_mlops_feature_store",
            "host": "eu-west.cloud.hopsworks.ai",
        }
    ]
    assert "secret-value" not in captured.out
    assert "secret-value" not in captured.err


def test_connect_to_hopsworks_normalizes_url_host(monkeypatch) -> None:
    login_calls = []
    fake_project = object()

    def fake_login(**kwargs):
        login_calls.append(kwargs)
        return fake_project

    fake_hopsworks = types.SimpleNamespace(login=fake_login)
    monkeypatch.setitem(sys.modules, "hopsworks", fake_hopsworks)
    monkeypatch.setenv("HOPSWORKS_API_KEY", "secret-value")
    monkeypatch.setenv("HOPSWORKS_PROJECT", "sp500_mlops_feature_store")
    monkeypatch.setenv("HOPSWORKS_HOST", "https://eu-west.cloud.hopsworks.ai")

    result = connect_to_hopsworks(env_path="missing-test-env-file")

    assert result is fake_project
    assert login_calls[0]["host"] == "eu-west.cloud.hopsworks.ai"
    assert "/" not in login_calls[0]["host"]


def test_normalize_hopsworks_host_removes_scheme_and_trailing_slash() -> None:
    assert (
        _normalize_hopsworks_host("https://eu-west.cloud.hopsworks.ai/")
        == "eu-west.cloud.hopsworks.ai"
    )


def test_create_or_get_sp500_feature_group_uses_expected_metadata() -> None:
    class FakeFeatureStore:
        def __init__(self):
            self.calls = []

        def get_or_create_feature_group(self, **kwargs):
            self.calls.append(kwargs)
            return "feature-group"

    feature_store = FakeFeatureStore()

    result = create_or_get_sp500_feature_group(
        feature_store=feature_store,
        feature_group_name="sp500_market_features",
        feature_group_version=1,
    )

    assert result == "feature-group"
    assert feature_store.calls[0]["name"] == "sp500_market_features"
    assert feature_store.calls[0]["version"] == 1
    assert feature_store.calls[0]["primary_key"] == [DATE_COLUMN]
    assert feature_store.calls[0]["event_time"] == DATE_COLUMN
    assert feature_store.calls[0]["time_travel_format"] == "HUDI"
    assert feature_store.calls[0]["hudi_precombine_key"] == DATE_COLUMN
    assert feature_store.calls[0]["online_enabled"] is False


def test_insert_sp500_feature_data_calls_feature_group_insert() -> None:
    class FakeFeatureGroup:
        def __init__(self):
            self.insert_calls = []

        def insert(self, dataframe, write_options):
            self.insert_calls.append((dataframe, write_options))

    feature_group = FakeFeatureGroup()
    feature_data = validate_feature_store_dataframe(make_feature_data())

    insert_sp500_feature_data(feature_group, feature_data, wait_for_job=False)

    inserted_dataframe, write_options = feature_group.insert_calls[0]
    pd.testing.assert_frame_equal(inserted_dataframe, feature_data)
    assert write_options == {"wait_for_job": False}


def test_create_or_get_sp500_feature_view_uses_features_and_target() -> None:
    class FakeFeatureGroup:
        def __init__(self):
            self.selected_columns = None

        def select(self, columns):
            self.selected_columns = columns
            return "query"

    class FakeFeatureStore:
        def __init__(self):
            self.calls = []

        def get_or_create_feature_view(self, **kwargs):
            self.calls.append(kwargs)
            return "feature-view"

    feature_group = FakeFeatureGroup()
    feature_store = FakeFeatureStore()

    result = create_or_get_sp500_feature_view(
        feature_store=feature_store,
        feature_group=feature_group,
        feature_view_name="sp500_direction_training_view",
        feature_view_version=1,
    )

    assert result == "feature-view"
    assert feature_group.selected_columns == [*MARKET_FEATURE_COLUMNS, TARGET_COLUMN]
    assert feature_store.calls[0]["name"] == "sp500_direction_training_view"
    assert feature_store.calls[0]["version"] == 1
    assert feature_store.calls[0]["query"] == "query"
    assert feature_store.calls[0]["labels"] == [TARGET_COLUMN]


def test_upload_sp500_features_to_hopsworks_returns_upload_summary(monkeypatch) -> None:
    class FakeFeatureGroup:
        def __init__(self):
            self.inserted_dataframe = None

        def insert(self, dataframe, write_options):
            self.inserted_dataframe = dataframe

        def select(self, columns):
            return {"columns": columns}

    class FakeFeatureStore:
        def __init__(self):
            self.feature_group = FakeFeatureGroup()

        def get_or_create_feature_group(self, **kwargs):
            return self.feature_group

        def get_or_create_feature_view(self, **kwargs):
            return {"feature_view": kwargs}

    class FakeProject:
        def __init__(self):
            self.feature_store = FakeFeatureStore()

        def get_feature_store(self):
            return self.feature_store

    fake_project = FakeProject()
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.feature_store.nodes.connect_to_hopsworks",
        lambda env_path: fake_project,
    )

    summary = upload_sp500_features_to_hopsworks(
        feature_data=make_feature_data(),
        feature_group_name="sp500_market_features",
        feature_group_version=1,
        feature_view_name="sp500_direction_training_view",
        feature_view_version=1,
        env_path=".env",
        wait_for_job=False,
    )

    assert summary["feature_group_name"] == "sp500_market_features"
    assert summary["feature_group_version"] == 1
    assert summary["feature_view_name"] == "sp500_direction_training_view"
    assert summary["feature_view_version"] == 1
    assert summary["feature_columns"] == MARKET_FEATURE_COLUMNS
    assert summary["target_column"] == TARGET_COLUMN
    assert summary["rows_uploaded"] == 3
    assert TARGET_COLUMN not in summary["feature_columns"]
