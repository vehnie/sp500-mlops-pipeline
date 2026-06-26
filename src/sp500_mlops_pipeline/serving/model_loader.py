"""Load and serve the MLflow candidate-champion model."""

from dataclasses import dataclass
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "sp500_direction_model"
MODEL_ALIAS = "candidate_champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

FEATURE_NAMES = [
    "simple_return",
    "log_return",
    "sma_10",
    "sma_20",
    "sma_ratio_10",
    "rsi_14",
    "volatility_10",
    "volume_change",
]


@dataclass(frozen=True)
class ModelService:
    """Loaded sklearn model and immutable registry metadata."""

    model: object
    model_name: str
    model_alias: str
    model_version: str | None
    model_family: str
    feature_names: list[str]

    def predict(self, feature_values: dict[str, float]) -> dict:
        """Predict one observation using the trained feature order."""
        model_input = pd.DataFrame(
            [[feature_values[name] for name in self.feature_names]],
            columns=self.feature_names,
        )

        prediction = int(self.model.predict(model_input)[0])
        probabilities = np.asarray(self.model.predict_proba(model_input))[0]
        classes = list(self.model.classes_)

        probability_by_class = {
            int(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

        return {
            "prediction": prediction,
            "prediction_label": "up" if prediction == 1 else "down",
            "probability_down": probability_by_class.get(0, 0.0),
            "probability_up": probability_by_class.get(1, 0.0),
            "model_name": self.model_name,
            "model_alias": self.model_alias,
        }


def load_candidate_champion_model() -> ModelService:
    """Load the candidate champion once from the MLflow Model Registry."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    local_model_path = os.getenv("LOCAL_MODEL_PATH")

    if local_model_path:
        model = mlflow.sklearn.load_model(local_model_path)
    else:
        model = mlflow.sklearn.load_model(MODEL_URI)

    trained_feature_names = list(
        getattr(model, "feature_names_in_", FEATURE_NAMES)
    )

    if trained_feature_names != FEATURE_NAMES:
        raise ValueError(
            "Registered model feature order does not match the serving contract"
        )

    return ModelService(
        model=model,
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=model_version.version,
        model_family=model_version.tags.get(
            "model_family",
            "logistic_regression",
        ),
        feature_names=trained_feature_names,
    )
