"""Manual Yahoo Finance data-ingestion pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_sp500_market_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the standalone manual market-data refresh pipeline.

    The default Kedro pipeline intentionally consumes the persisted local raw CSV
    snapshot and does not call Yahoo Finance.
    """
    return pipeline(
        [
            node(
                func=download_sp500_market_data,
                inputs={
                    "ticker": "params:data_ingestion.ticker",
                    "start_date": "params:data_ingestion.start_date",
                    "yfinance_cache_dir": "params:data_ingestion.yfinance_cache_dir",
                },
                outputs="sp500_raw_data",
                name="download_sp500_market_data_node",
            )
        ]
    )
