FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.137.1 \
    uvicorn==0.49.0 \
    mlflow==3.14.0 \
    numpy==2.4.6 \
    pandas==2.3.3 \
    scikit-learn==1.9.0 \
    skops==0.14.0

COPY src/sp500_mlops_pipeline/__init__.py src/sp500_mlops_pipeline/__init__.py
COPY src/sp500_mlops_pipeline/serving src/sp500_mlops_pipeline/serving

EXPOSE 8000

CMD ["uvicorn", "sp500_mlops_pipeline.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
