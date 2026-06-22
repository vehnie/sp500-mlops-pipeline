"""FastAPI application serving the MLflow candidate-champion model."""

from collections.abc import Callable
from contextlib import asynccontextmanager
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .model_loader import ModelService, load_candidate_champion_model
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


def create_app(
    model_loader: Callable[[], ModelService] = load_candidate_champion_model,
) -> FastAPI:
    """Create an API whose model is loaded once during application startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model_service = model_loader()
        yield

    application = FastAPI(
        title="S&P 500 Direction Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=422,
            content={"detail": _sanitize_non_finite_values(exc.errors())},
        )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        service = _get_model_service(request)
        return HealthResponse(
            status="ok",
            model_name=service.model_name,
            model_alias=service.model_alias,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        service = _get_model_service(request)
        return ModelInfoResponse(
            model_name=service.model_name,
            model_alias=service.model_alias,
            model_version=service.model_version,
            model_family=service.model_family,
            feature_names=service.feature_names,
            expected_number_of_features=len(service.feature_names),
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        service = _get_model_service(request)
        result = service.predict(payload.model_dump())
        return PredictionResponse(**result)

    return application


def _get_model_service(request: Request) -> ModelService:
    """Return the model loaded by the application lifespan."""
    return request.app.state.model_service


def _sanitize_non_finite_values(value):
    """Make validation errors JSON serializable without hiding their cause."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {
            key: _sanitize_non_finite_values(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite_values(item) for item in value]
    return value


app = create_app()
