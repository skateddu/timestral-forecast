from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from timestral_forecast.models.timestral_model import (
    TimestralVariant,
    backtest_timestral,
    predict_quantiles,
    quantile_column,
    resolve_device,
)
from timestral_forecast.preprocessing import (
    KNOWN_COVARIATES,
    build_timestral_arrays,
    store_of,
    store_totals,
)

LEVELS = (0.1, 0.5, 0.9)


@dataclass
class FakeForecast:
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]


class FakeForecaster:
    """Returns last value ± 1 as quantiles and records every call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forecast(self, context: np.ndarray, prediction_length: int, **kwargs: Any) -> FakeForecast:
        self.calls.append({"batch": context.shape[0], **kwargs})
        last = np.nan_to_num(context[..., -1:])
        base = np.repeat(last, prediction_length, axis=-1)[..., None]
        return FakeForecast(quantiles=base + np.array([-1.0, 0.0, 1.0]), quantile_levels=LEVELS)


def test_predict_quantiles_batches_and_concatenates(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    df = make_long({f"s{i}": [i + 1.0] * 5 for i in range(5)})
    inputs = build_timestral_arrays(df, cutoff=pd.Timestamp("2020-01-05"), horizon=2)
    fake = FakeForecaster()

    quantiles, levels = predict_quantiles(fake, inputs=inputs, horizon=2, batch_size=2)

    assert [c["batch"] for c in fake.calls] == [2, 2, 1]
    assert quantiles.shape == (5, 2, 3)
    assert levels == LEVELS


def test_predict_quantiles_passes_known_covariates(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1.0] * 6})
    inputs = build_timestral_arrays(
        df, cutoff=pd.Timestamp("2020-01-04"), horizon=2, x_df=make_covariates(df)
    )
    fake = FakeForecaster()

    predict_quantiles(fake, inputs=inputs, horizon=2, batch_size=8)

    assert fake.calls[0]["known_covariates"].shape[-1] == 4 + 2


@pytest.mark.parametrize("symmetric", [True, False])
def test_backtest_timestral_forwards_symmetric_averaging(
    make_long: Callable[..., pd.DataFrame], symmetric: bool
) -> None:
    fake = FakeForecaster()

    backtest_timestral(
        fake,
        variant=TimestralVariant(alias="TS", model_id="m", symmetric_averaging=symmetric),
        y_df=make_long({"a": [1.0] * 6}),
        cutoffs=[pd.Timestamp("2020-01-04")],
        horizon=2,
    )

    assert fake.calls[0]["symmetric_averaging"] is symmetric


def test_backtest_timestral_frame_has_median_mean_and_quantiles(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    df = make_long({"a": [3.0] * 10, "b": [5.0] * 10})
    cutoffs = [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-08")]

    fc, seconds = backtest_timestral(
        FakeForecaster(),
        variant=TimestralVariant(alias="TS", model_id="m"),
        y_df=df,
        cutoffs=cutoffs,
        horizon=2,
    )

    assert len(fc) == 2 * 2 * 2
    assert {"TS", "TS-mean", quantile_column("TS", 0.1), "cutoff"} <= set(fc.columns)
    assert fc.loc[fc["unique_id"] == "b", "TS"].eq(5.0).all()
    assert fc["TS-mean"].equals(fc["TS"])  # symmetric fake quantiles: mean = median
    assert seconds >= 0


def test_quantile_column_zero_pads_levels() -> None:
    assert quantile_column("TS", 0.05) == "TS_q05"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resolve_device_explicit_value_is_kept(device: str) -> None:
    assert resolve_device(device) == device


def test_backtest_timestral_multivariate_covers_every_series_once(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({f"s{i}": [i + 1.0] * 8 for i in range(5)})
    s_df = pd.DataFrame(
        {"unique_id": [f"s{i}" for i in range(5)], "store_id": ["A"] * 3 + ["B"] * 2}
    )
    fake = FakeForecaster()
    variant = TimestralVariant(
        alias="TS", model_id="m", known_covariates=("snap",), past_covariates=True, group_size=2
    )

    fc, _ = backtest_timestral(
        fake,
        variant=variant,
        y_df=df,
        cutoffs=[pd.Timestamp("2020-01-06")],
        horizon=2,
        x_df=make_covariates(df),
        past_df=store_totals(df, s_df=s_df),
        store_of=store_of(s_df),
    )

    assert fc.groupby("unique_id")["TS"].first().to_dict() == {f"s{i}": i + 1.0 for i in range(5)}
    assert len(fc) == 5 * 2
    assert {c["past_covariates"].shape[1] for c in fake.calls} == {1}


def test_backtest_timestral_masks_constant_price(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1.0] * 6})
    fake = FakeForecaster()
    variant = TimestralVariant(
        alias="TS", model_id="m", known_covariates=KNOWN_COVARIATES, mask_constant_price=True
    )

    backtest_timestral(
        fake,
        variant=variant,
        y_df=df,
        cutoffs=[pd.Timestamp("2020-01-04")],
        horizon=2,
        x_df=make_covariates(df, extra_days=2),
    )

    assert np.isnan(fake.calls[0]["known_covariates"][0, 0]).all()
