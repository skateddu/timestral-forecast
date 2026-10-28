"""Global LightGBM benchmark with mlforecast: one model trained on all series."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean, RollingStd

from timestral_forecast.preprocessing import (
    EVENT_CATEGORIES,
    KNOWN_COVARIATES,
    calendar_features,
    split_at,
    trim_leading_zeros,
)

logger = logging.getLogger(__name__)

LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
LAG_TRANSFORMS = {
    1: [
        RollingMean(window_size=7),
        RollingMean(window_size=28),
        RollingStd(window_size=28),
        ExponentiallyWeightedMean(alpha=0.1),
    ],
    7: [RollingMean(window_size=4 * 7)],
    28: [RollingMean(window_size=28)],
}
DYNAMIC_FEATURES = ("sell_price", "price_ratio", "snap", "is_event")
STATIC_FEATURES = ["item_id", "store_id"]
DATE_FEATURES = ["dayofweek", "day", "month", "year"]
LEVEL_DAYS = 56
# keeps the scale positive for series that sold nothing in the last LEVEL_DAYS
LEVEL_FLOOR = 0.1


def dow_sin(dates: pd.DatetimeIndex) -> np.ndarray:
    """Sine of the day of week, as fed to Timestral."""
    return calendar_features(pd.DatetimeIndex(dates))[0]


def dow_cos(dates: pd.DatetimeIndex) -> np.ndarray:
    """Cosine of the day of week, as fed to Timestral."""
    return calendar_features(pd.DatetimeIndex(dates))[1]


def doy_sin(dates: pd.DatetimeIndex) -> np.ndarray:
    """Sine of the day of year, as fed to Timestral."""
    return calendar_features(pd.DatetimeIndex(dates))[2]


def doy_cos(dates: pd.DatetimeIndex) -> np.ndarray:
    """Cosine of the day of year, as fed to Timestral."""
    return calendar_features(pd.DatetimeIndex(dates))[3]


CYCLIC_DATE_FEATURES = [dow_sin, dow_cos, doy_sin, doy_cos]


@dataclass(frozen=True)
class LGBMVariant:
    """One LightGBM configuration of the benchmark.

    Attributes
    ----------
    alias : str
        Output column name.
    normalize : bool
        Divide each series by its recent level and weight its rows by that level (as in
        VN1): series become comparable while each one still counts by its volume.
    direct : bool
        One model per horizon step (`max_horizon`) instead of recursive forecasting.
    train_days : int | None
        Train on the last `train_days` before the origin only (bounds the direct cost).
    covariates : tuple[str, ...]
        Known-future columns of `x_df` used as features; `price_ratio` is derived from
        `sell_price`, which must then be listed too. Categorical columns (event name and
        type) are passed to LightGBM as categories.
    cyclic_calendar : bool
        Sine/cosine of day of week and day of year, exactly as fed to Timestral, instead of
        day of week, day of month, month and year (which together single out each holiday).
    static_features : bool
        Use item and store ids as categorical features.
    """

    alias: str
    normalize: bool = False
    direct: bool = False
    train_days: int | None = None
    covariates: tuple[str, ...] = DYNAMIC_FEATURES
    cyclic_calendar: bool = False
    static_features: bool = True


# Same information as Timestral-28M+cov and +cov+cat: no ids, no day/month/year, raw price
# only, and the 1,024 days of Timestral's context (its `max_context`).
_MATCHED = {
    "normalize": True,
    "train_days": 1024,
    "cyclic_calendar": True,
    "static_features": False,
}

LGBM_VARIANTS = (
    LGBMVariant(alias="LightGBM"),
    LGBMVariant(alias="LightGBM-norm", normalize=True),
    LGBMVariant(alias="LightGBM-norm-direct", normalize=True, direct=True, train_days=730),
    LGBMVariant(alias="LightGBM-norm-matched", covariates=KNOWN_COVARIATES, **_MATCHED),
    LGBMVariant(
        alias="LightGBM-norm-matched+cat",
        covariates=("sell_price", "snap", *EVENT_CATEGORIES),
        **_MATCHED,
    ),
)


def build_mlforecast(
    alias: str, tweedie_power: float, n_jobs: int, seed: int = 0, cyclic_calendar: bool = False
) -> MLForecast:
    """Build a global LightGBM with Tweedie loss (log link, suited to counts with many zeros).

    Parameters
    ----------
    alias : str
        Output column name.
    tweedie_power : float
        Tweedie variance power, between 1 (Poisson) and 2 (Gamma).
    n_jobs : int
        LightGBM threads.
    seed : int
        Random seed of row and column subsampling.
    cyclic_calendar : bool
        Sine/cosine date features instead of day of week, day, month and year.

    Returns
    -------
    MLForecast
        Unfitted forecaster with lags, rolling statistics and date features.
    """
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=tweedie_power,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    return MLForecast(
        models={alias: model},
        freq="D",
        lags=LAGS,
        lag_transforms=LAG_TRANSFORMS,
        date_features=CYCLIC_DATE_FEATURES if cyclic_calendar else DATE_FEATURES,
    )


def _training_frame(
    y_df: pd.DataFrame, x_df: pd.DataFrame, s_df: pd.DataFrame, variant: LGBMVariant
) -> pd.DataFrame:
    """Join target, the variant's covariates and, if used, categorical static features."""
    cols = ["unique_id", "ds", *(c for c in variant.covariates if c != "price_ratio")]
    df = y_df.merge(x_df[cols], on=["unique_id", "ds"], how="left")
    if not variant.static_features:
        return df
    statics = s_df[["unique_id", *STATIC_FEATURES]].astype(
        {c: "category" for c in STATIC_FEATURES}
    )
    return df.merge(statics, on="unique_id", how="left")


def add_price_ratio(
    train: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the price relative to the series' median price before the origin (0.8 = 20% discount).

    Parameters
    ----------
    train : pd.DataFrame
        Training rows (before the origin) with `unique_id, sell_price`.
    future : pd.DataFrame
        Horizon rows with the same columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Both frames with a `price_ratio` column, scaled by the training median only.
    """
    median = train.groupby("unique_id", observed=True)["sell_price"].median()
    out = []
    for frame in (train, future):
        out.append(frame.assign(price_ratio=frame["sell_price"] / frame["unique_id"].map(median)))
    return out[0], out[1]


def recent_level(train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Mean demand of each series over the last `LEVEL_DAYS` before `cutoff`, plus a floor.

    Parameters
    ----------
    train : pd.DataFrame
        Training rows `unique_id, ds, y`.
    cutoff : pd.Timestamp
        Forecast origin.

    Returns
    -------
    pd.Series
        Level by `unique_id`, always at least `LEVEL_FLOOR`.
    """
    recent = train[train["ds"] > cutoff - pd.Timedelta(days=LEVEL_DAYS)]
    level = recent.groupby("unique_id", observed=True)["y"].mean()
    return level.reindex(train["unique_id"].unique()).fillna(0) + LEVEL_FLOOR


def _fit_predict(
    train: pd.DataFrame,
    future: pd.DataFrame,
    variant: LGBMVariant,
    horizon: int,
    tweedie_power: float,
    n_jobs: int,
) -> pd.DataFrame:
    """Fit one variant on `train` and forecast `horizon` days with the known future features."""
    mlf = build_mlforecast(
        alias=variant.alias,
        tweedie_power=tweedie_power,
        n_jobs=n_jobs,
        cyclic_calendar=variant.cyclic_calendar,
    )
    mlf.fit(
        train,
        static_features=STATIC_FEATURES if variant.static_features else [],
        weight_col="weight" if variant.normalize else None,
        max_horizon=horizon if variant.direct else None,
    )
    return mlf.predict(h=horizon, X_df=future[["unique_id", "ds", *variant.covariates]])


def backtest_lightgbm(
    y_df: pd.DataFrame,
    x_df: pd.DataFrame,
    s_df: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    horizon: int,
    tweedie_power: float,
    n_jobs: int,
    variant: LGBMVariant = LGBM_VARIANTS[0],
) -> pd.DataFrame:
    """Train one global LightGBM per origin and forecast `horizon` days.

    Prices, SNAP days and events are planned in advance, so their future values are
    passed to `predict` as known exogenous features.

    Parameters
    ----------
    y_df, x_df, s_df : pd.DataFrame
        Target, known covariates and static attributes (see `M5Data`).
    cutoffs : Sequence[pd.Timestamp]
        Backtest origins.
    horizon : int
        Days to forecast.
    tweedie_power : float
        Tweedie variance power.
    n_jobs : int
        LightGBM threads.
    variant : LGBMVariant
        Normalisation, strategy and training window.

    Returns
    -------
    pd.DataFrame
        `unique_id, ds, cutoff, <variant.alias>`.
    """
    full = _training_frame(y_df, x_df=x_df, s_df=s_df, variant=variant)
    out = []
    for cutoff in cutoffs:
        train, future = split_at(full, cutoff=cutoff, horizon=horizon)
        if variant.train_days is not None:
            train = train[train["ds"] > cutoff - pd.Timedelta(days=variant.train_days)]
        train = trim_leading_zeros(train)
        future = future[future["unique_id"].isin(train["unique_id"].unique())]
        if "price_ratio" in variant.covariates:
            train, future = add_price_ratio(train, future)
        if variant.normalize:
            level = recent_level(train, cutoff=cutoff)
            train = train.assign(weight=train["unique_id"].map(level))
            train["y"] = train["y"] / train["weight"]
        fc = _fit_predict(
            train,
            future,
            variant=variant,
            horizon=horizon,
            tweedie_power=tweedie_power,
            n_jobs=n_jobs,
        )
        if variant.normalize:
            fc[variant.alias] = fc[variant.alias] * fc["unique_id"].map(level).to_numpy()
        out.append(fc.assign(cutoff=cutoff))
        logger.info(
            "LightGBM fold done", extra={"alias": variant.alias, "cutoff": str(cutoff.date())}
        )
    return pd.concat(out, ignore_index=True)
