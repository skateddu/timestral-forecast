"""Statistical benchmarks with statsforecast: moving average, naive methods, Croston, AutoARIMA."""

import logging
import os
from collections.abc import Sequence

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonSBA,
    Naive,
    SeasonalNaive,
    WindowAverage,
)

from timestral_forecast.preprocessing import split_at, trim_leading_zeros

logger = logging.getLogger(__name__)


def baseline_models(season_length: int, window: int) -> list[object]:
    """Build the cheap reference models: moving average, naive, seasonal naive and Croston SBA.

    Parameters
    ----------
    season_length : int
        Seasonal period of SeasonalNaive.
    window : int
        Window of the moving average, which is aliased `MA<window>`.

    Returns
    -------
    list[object]
        statsforecast model instances.
    """
    return [
        WindowAverage(window_size=window, alias=f"MA{window}"),
        Naive(),
        SeasonalNaive(season_length=season_length),
        CrostonSBA(),
    ]


def arima_models(season_length: int) -> list[object]:
    """Build AutoARIMA with weekly seasonality, the classic time-series benchmark.

    Parameters
    ----------
    season_length : int
        Seasonal period.

    Returns
    -------
    list[object]
        statsforecast model instances.
    """
    return [AutoARIMA(season_length=season_length)]


def exponential_smoothing_models(season_length: int) -> list[object]:
    """Build AutoETS and AutoCES (complex exponential smoothing), the other M-competition classics.

    Parameters
    ----------
    season_length : int
        Seasonal period.

    Returns
    -------
    list[object]
        statsforecast model instances.
    """
    return [
        AutoETS(season_length=season_length),
        AutoCES(season_length=season_length, alias="AutoCES"),
    ]


def backtest_statsforecast(
    y_df: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    horizon: int,
    models: list[object],
    n_jobs: int,
    train_days: int | None = None,
) -> pd.DataFrame:
    """Fit and forecast statsforecast models from each origin.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    cutoffs : Sequence[pd.Timestamp]
        Backtest origins.
    horizon : int
        Days to forecast.
    models : list[object]
        statsforecast model instances.
    n_jobs : int
        Parallel workers.
    train_days : int | None
        If set, fit on the last `train_days` before each origin only.

    Returns
    -------
    pd.DataFrame
        `unique_id, ds, cutoff` plus one column per model alias.
    """
    # Spawned workers read this at start-up: one BLAS thread each, or many workers exhaust memory
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    out = []
    for cutoff in cutoffs:
        train, _ = split_at(y_df, cutoff=cutoff, horizon=horizon)
        if train_days is not None:
            train = train[train["ds"] > cutoff - pd.Timedelta(days=train_days)]
        train = trim_leading_zeros(train)
        sf = StatsForecast(models=models, freq="D", n_jobs=n_jobs, fallback_model=Naive())
        fc = sf.forecast(df=train, h=horizon)
        out.append(fc.assign(cutoff=cutoff))
        logger.info(
            "statsforecast fold done",
            extra={"cutoff": str(cutoff.date()), "n_series": fc["unique_id"].nunique()},
        )
    return pd.concat(out, ignore_index=True)
