"""Pydantic request and response schemas for model serving."""

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    """Eight required finite features expected by the registered model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    simple_return: float
    log_return: float
    sma_10: float
    sma_20: float
    sma_ratio_10: float
    rsi_14: float
    volatility_10: float
    volume_change: float


class PredictionResponse(BaseModel):
    """Binary prediction and class probabilities."""

    prediction: int
    prediction_label: str
    probability_down: float
    probability_up: float
    model_name: str
    model_alias: str


class HealthResponse(BaseModel):
    """API and loaded-model health status."""

    status: str
    model_name: str
    model_alias: str


class ModelInfoResponse(BaseModel):
    """Metadata describing the model currently loaded by the API."""

    model_name: str
    model_alias: str
    model_version: str | None
    model_family: str
    feature_names: list[str]
    expected_number_of_features: int
