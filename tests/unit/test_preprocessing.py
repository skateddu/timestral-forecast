from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from timestral_forecast.preprocessing import (
    CALENDAR_COVARIATES,
    EVENT_CATEGORIES,
    KNOWN_COVARIATES,
    backtest_cutoffs,
    build_timestral_arrays,
    calendar_features,
    mask_constant_covariate,
    split_at,
    store_of,
    trim_leading_zeros,
)


def test_trim_leading_zeros_drops_rows_before_first_sale(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    df = make_long({"a": [0, 0, 3, 0, 1], "b": [1, 0, 0, 0, 0], "never": [0, 0, 0, 0, 0]})

    out = trim_leading_zeros(df)

    assert out.groupby("unique_id").size().to_dict() == {"a": 3, "b": 5}


def test_backtest_cutoffs_are_non_overlapping_and_oldest_first() -> None:
    cutoffs = backtest_cutoffs(pd.Timestamp("2020-12-31"), horizon=28, n_windows=3)

    assert cutoffs == [pd.Timestamp("2020-12-31") - pd.Timedelta(days=28 * k) for k in (3, 2, 1)]


def test_split_at_test_window_has_horizon_days(make_long: Callable[..., pd.DataFrame]) -> None:
    df = make_long({"a": list(range(20))})
    cutoff = pd.Timestamp("2020-01-10")

    train, test = split_at(df, cutoff=cutoff, horizon=5)

    assert train["ds"].max() == cutoff
    assert len(test) == 5


def test_calendar_features_shape_and_range() -> None:
    feats = calendar_features(pd.date_range("2020-01-01", periods=10, freq="D"))

    assert feats.shape == (4, 10)
    assert np.abs(feats).max() <= 1.0


def test_build_timestral_arrays_history_only_shapes_and_prelaunch_nan(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    df = make_long({"a": [1, 2, 3, 4, 5, 6], "late": [0, 0, 0, 2, 0, 1], "never": [0] * 6})

    inputs = build_timestral_arrays(df, cutoff=pd.Timestamp("2020-01-06"), horizon=3)

    assert list(inputs.unique_ids) == ["a", "late"]
    assert inputs.context.shape == (2, 1, 6)
    assert np.isnan(inputs.context[1, 0, :3]).all()
    assert inputs.context[1, 0, 4] == 0  # zero after launch is real demand
    assert inputs.known_covariates is None
    assert len(inputs.future_ds) == 3


def test_build_timestral_arrays_with_covariates_spans_context_plus_horizon(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1, 2, 3, 4, 5, 6, 7, 8]})
    x = make_covariates(df)
    cutoff = pd.Timestamp("2020-01-05")

    inputs = build_timestral_arrays(df, cutoff=cutoff, horizon=3, x_df=x)

    assert inputs.known_covariates.shape == (
        1,
        len(KNOWN_COVARIATES) + len(CALENDAR_COVARIATES),
        5 + 3,
    )
    assert inputs.covariate_names == (*KNOWN_COVARIATES, *CALENDAR_COVARIATES)
    assert not np.isnan(inputs.known_covariates).any()


def test_build_timestral_arrays_missing_future_covariates_are_nan(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1, 2, 3, 4, 5]})
    x = make_covariates(df)  # ends at the cutoff: future unknown

    inputs = build_timestral_arrays(df, cutoff=pd.Timestamp("2020-01-05"), horizon=2, x_df=x)

    assert np.isnan(inputs.known_covariates[0, 0, -2:]).all()


@pytest.mark.parametrize("horizon", [1, 28])
def test_build_timestral_arrays_future_dates_follow_cutoff(
    make_long: Callable[..., pd.DataFrame], horizon: int
) -> None:
    df = make_long({"a": [1.0] * 10})
    cutoff = pd.Timestamp("2020-01-10")

    inputs = build_timestral_arrays(df, cutoff=cutoff, horizon=horizon)

    assert inputs.future_ds[0] == cutoff + pd.Timedelta(days=1)
    assert len(inputs.future_ds) == horizon


def test_mask_constant_covariate_blanks_only_flat_series(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"flat": [1.0] * 8, "moving": [1.0] * 8})
    x = make_covariates(df)
    x.loc[(x["unique_id"] == "moving") & (x["ds"] == "2020-01-03"), "sell_price"] = 1.5
    x.loc[(x["unique_id"] == "flat") & (x["ds"] == "2020-01-07"), "sell_price"] = (
        1.5  # change in horizon only
    )
    inputs = build_timestral_arrays(df, cutoff=pd.Timestamp("2020-01-05"), horizon=3, x_df=x)

    masked, n = mask_constant_covariate(inputs, name="sell_price", window=1024)

    flat, moving = list(masked.unique_ids).index("flat"), list(masked.unique_ids).index("moving")
    assert n == 1
    assert np.isnan(masked.known_covariates[flat, 0]).all()
    assert not np.isnan(masked.known_covariates[moving, 0]).any()
    assert not np.isnan(masked.known_covariates[flat, 1:]).any()  # other covariates untouched


def test_mask_constant_covariate_without_covariates_is_noop(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    inputs = build_timestral_arrays(
        make_long({"a": [1.0] * 5}), cutoff=pd.Timestamp("2020-01-03"), horizon=2
    )

    masked, n = mask_constant_covariate(inputs, name="sell_price", window=1024)

    assert masked is inputs
    assert n == 0


def test_timestral_inputs_context_ds_ends_at_cutoff(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    y_df = make_long({"a": [1.0] * 10, "b": [0.0, 2.0] * 5})
    cutoff = pd.Timestamp("2020-01-08")

    inputs = build_timestral_arrays(y_df, cutoff=cutoff, horizon=3)

    assert inputs.context_ds[-1] == cutoff
    assert len(inputs.context_ds) == inputs.context.shape[-1]


def test_timestral_inputs_index_of_finds_series_and_rejects_unknown(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    inputs = build_timestral_arrays(
        make_long({"a": [1.0] * 5, "b": [2.0] * 5}), cutoff=pd.Timestamp("2020-01-04"), horizon=1
    )

    assert inputs.unique_ids[inputs.index_of("b")] == "b"
    with pytest.raises(KeyError):
        inputs.index_of("missing")


def test_store_of_maps_series_to_store() -> None:
    s_df = pd.DataFrame({"unique_id": ["a", "b"], "store_id": ["CA_1", "TX_1"]})

    assert store_of(s_df).to_dict() == {"a": "CA_1", "b": "TX_1"}


def test_build_timestral_arrays_categorical_codes_span_context_plus_horizon(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1.0] * 5, "b": [2.0] * 5})
    x = make_covariates(df, extra_days=2)

    inputs = build_timestral_arrays(
        df,
        cutoff=pd.Timestamp("2020-01-05"),
        horizon=2,
        x_df=x,
        covariates=("snap",),
        categorical=EVENT_CATEGORIES,
    )

    codes = inputs.categorical_covariates
    assert codes.shape == (2, len(EVENT_CATEGORIES), 5 + 2)
    assert inputs.categorical_names == EVENT_CATEGORIES
    assert inputs.covariate_names == ("snap", *CALENDAR_COVARIATES)
    np.testing.assert_array_equal(codes[0], codes[1])  # same code for the same category
    event_day = codes[0, 0, 2]  # 2020-01-03
    assert np.isfinite(codes).all()
    assert (codes[0, 0, np.arange(7) != 2] != event_day).all()


def test_build_timestral_arrays_without_categorical_has_none(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1.0] * 5})

    inputs = build_timestral_arrays(
        df, cutoff=pd.Timestamp("2020-01-05"), horizon=2, x_df=make_covariates(df)
    )

    assert inputs.categorical_covariates is None
    assert inputs.categorical_names == ()


def test_build_timestral_arrays_categorical_only_keeps_calendar_rows(
    make_long: Callable[..., pd.DataFrame], make_covariates: Callable[..., pd.DataFrame]
) -> None:
    df = make_long({"a": [1.0] * 5})

    inputs = build_timestral_arrays(
        df,
        cutoff=pd.Timestamp("2020-01-05"),
        horizon=2,
        x_df=make_covariates(df, extra_days=2),
        covariates=(),
        categorical=("event_name",),
    )

    assert inputs.covariate_names == CALENDAR_COVARIATES
    assert inputs.known_covariates.shape == (1, len(CALENDAR_COVARIATES), 5 + 2)
    assert inputs.categorical_covariates.shape == (1, 1, 5 + 2)
