from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from timestral_forecast.models.baselines import backtest_statsforecast, baseline_models
from timestral_forecast.models.lgbm import (
    LEVEL_DAYS,
    LEVEL_FLOOR,
    LGBM_VARIANTS,
    LGBMVariant,
    add_price_ratio,
    backtest_lightgbm,
    recent_level,
)


@pytest.fixture
def weekly_series(make_long: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    pattern = np.tile([1, 2, 3, 4, 5, 8, 9], 20)
    return make_long(
        {f"s{i}": (pattern + rng.poisson(lam=1, size=len(pattern))).tolist() for i in range(3)}
    )


def test_backtest_statsforecast_returns_horizon_per_series_and_cutoff(
    weekly_series: pd.DataFrame,
) -> None:
    cutoffs = [pd.Timestamp("2020-04-15"), pd.Timestamp("2020-04-22")]

    fc = backtest_statsforecast(
        weekly_series,
        cutoffs=cutoffs,
        horizon=7,
        models=baseline_models(season_length=7, window=14),
        n_jobs=1,
        train_days=60,
    )

    assert fc.groupby("cutoff").size().tolist() == [3 * 7, 3 * 7]
    assert {"MA14", "Naive", "SeasonalNaive", "CrostonSBA"} <= set(fc.columns)


def test_backtest_lightgbm_uses_known_covariates(
    weekly_series: pd.DataFrame, make_covariates: Callable[..., pd.DataFrame]
) -> None:
    x_df = make_covariates(weekly_series)
    s_df = pd.DataFrame(
        {"unique_id": ["s0", "s1", "s2"], "item_id": ["a", "b", "c"], "store_id": "CA_1"}
    )

    fc = backtest_lightgbm(
        weekly_series,
        x_df=x_df,
        s_df=s_df,
        cutoffs=[pd.Timestamp("2020-04-22")],
        horizon=7,
        tweedie_power=1.1,
        n_jobs=1,
    )

    assert len(fc) == 3 * 7
    assert (fc["LightGBM"] >= 0).all()  # Tweedie log link: non-negative forecasts


def test_recent_level_is_mean_of_last_days_plus_floor(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    df = make_long(
        {"a": [100.0] * 10 + [2.0] * LEVEL_DAYS, "dead": [5.0] * 10 + [0.0] * LEVEL_DAYS}
    )

    level = recent_level(df, cutoff=df["ds"].max())

    assert level["a"] == pytest.approx(2.0 + LEVEL_FLOOR)
    assert level["dead"] == pytest.approx(LEVEL_FLOOR)


def test_add_price_ratio_uses_training_median() -> None:
    train = pd.DataFrame({"unique_id": ["a"] * 3, "sell_price": [2.0, 2.0, 4.0]})
    future = pd.DataFrame({"unique_id": ["a"], "sell_price": [1.6]})

    train, future = add_price_ratio(train, future)

    assert future["price_ratio"].iloc[0] == pytest.approx(0.8)


@pytest.mark.parametrize("variant", LGBM_VARIANTS[1:], ids=lambda v: v.alias)
def test_backtest_lightgbm_variants_forecast_every_step(
    weekly_series: pd.DataFrame, make_covariates: Callable[..., pd.DataFrame], variant: LGBMVariant
) -> None:
    s_df = pd.DataFrame(
        {"unique_id": ["s0", "s1", "s2"], "item_id": ["a", "b", "c"], "store_id": "CA_1"}
    )

    fc = backtest_lightgbm(
        weekly_series,
        x_df=make_covariates(weekly_series),
        s_df=s_df,
        cutoffs=[pd.Timestamp("2020-04-22")],
        horizon=7,
        tweedie_power=1.1,
        n_jobs=1,
        variant=variant,
    )

    assert len(fc) == 3 * 7
    assert fc[variant.alias].between(0, 100).all()  # rescaled back to units
