from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from timestral_forecast.exceptions import ShapeMismatchError
from timestral_forecast.inventory import (
    calibration_error,
    correlation_sweep,
    cross_validate_correlation,
    demand_std,
    normal_order_up_to,
    quantile_levels,
    quantile_order_up_to,
    service_metrics,
)

CUTOFF = pd.Timestamp("2020-01-31")
GRID = (0.1, 0.5, 0.9)


def _forecast_frame(days: int = 14, spread: float = 2.0, y: float = 5.0) -> pd.DataFrame:
    """One series after `CUTOFF`, daily quantiles 5 - spread, 5, 5 + spread."""
    ds = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=days, freq="D")
    return pd.DataFrame(
        {
            "unique_id": "a",
            "ds": ds,
            "cutoff": CUTOFF,
            "y": y,
            "M": 5.0,
            "TS_q10": 5.0 - spread,
            "TS_q50": 5.0,
            "TS_q90": 5.0 + spread,
        }
    )


def test_quantile_levels_reads_only_the_alias_columns() -> None:
    df = _forecast_frame().assign(**{"TS+cov_q50": 1.0})

    assert quantile_levels(df, alias="TS") == GRID


def test_comonotonic_sums_the_daily_quantiles() -> None:
    levels_df = quantile_order_up_to(
        _forecast_frame(), alias="TS", days=7, service_levels=(0.5, 0.9), integer=False
    )

    assert levels_df["period"].unique().tolist() == [0, 1]
    by_level = levels_df.groupby("service_level")["order_up_to"].first()
    assert by_level[0.5] == pytest.approx(35.0)
    assert by_level[0.9] == pytest.approx(49.0)
    assert levels_df["y"].eq(35.0).all()
    assert levels_df["policy"].eq("TS (comonotonic)").all()


def test_independent_days_need_less_stock_than_comonotonic() -> None:
    kwargs = {"alias": "TS", "days": 7, "service_levels": (0.9,), "integer": False}

    co = quantile_order_up_to(_forecast_frame(), method="comonotonic", **kwargs)
    ind = quantile_order_up_to(_forecast_frame(), method="independent", n_samples=4000, **kwargs)

    assert (ind["order_up_to"] < co["order_up_to"]).all()
    assert (ind["order_up_to"] > 35.0).all()  # still above the median of the sum


def test_independent_with_degenerate_quantiles_is_exact() -> None:
    levels_df = quantile_order_up_to(
        _forecast_frame(spread=0.0), alias="TS", days=7, method="independent", n_samples=10
    )

    assert levels_df["order_up_to"].eq(35.0).all()


def test_order_up_to_is_rounded_up_to_whole_units() -> None:
    levels_df = quantile_order_up_to(
        _forecast_frame(spread=0.1), alias="TS", days=7, service_levels=(0.9,)
    )

    assert levels_df["order_up_to"].eq(np.ceil(7 * 5.1)).all()


def test_quantile_order_up_to_rejects_unknown_method_and_levels_off_grid() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        quantile_order_up_to(_forecast_frame(), alias="TS", days=7, method="gaussian")
    with pytest.raises(ValueError, match="within"):
        quantile_order_up_to(_forecast_frame(), alias="TS", days=7, service_levels=(0.99,))


def test_correlated_with_zero_correlation_is_independent() -> None:
    kwargs = {"alias": "TS", "days": 7, "service_levels": (0.9,), "integer": False, "seed": 3}

    ind = quantile_order_up_to(_forecast_frame(), method="independent", **kwargs)
    cor = quantile_order_up_to(_forecast_frame(), method="correlated", correlation=0.0, **kwargs)

    np.testing.assert_allclose(cor["order_up_to"], ind["order_up_to"])
    assert cor["policy"].eq("TS (correlated)").all()


def test_correlated_with_full_correlation_sums_the_daily_quantiles() -> None:
    levels_df = quantile_order_up_to(
        _forecast_frame(),
        alias="TS",
        days=7,
        service_levels=(0.5, 0.9),
        method="correlated",
        correlation=1.0,
        n_samples=4000,
        integer=False,
    )

    by_level = levels_df.groupby("service_level")["order_up_to"].mean()
    assert by_level[0.5] == pytest.approx(35.0, abs=0.5)
    assert by_level[0.9] == pytest.approx(49.0, abs=0.5)


def test_stock_grows_with_the_correlation_between_days() -> None:
    sweep_df = correlation_sweep(
        _forecast_frame(), alias="TS", days=7, correlations=(0.0, 0.5, 1.0), service_levels=(0.9,)
    )

    by_rho = sweep_df.groupby("correlation")["order_up_to"].mean()
    assert by_rho.is_monotonic_increasing
    assert by_rho[0.0] < by_rho[1.0]


def test_quantile_order_up_to_rejects_correlation_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="correlation"):
        quantile_order_up_to(
            _forecast_frame(), alias="TS", days=7, method="correlated", correlation=1.5
        )


def test_calibration_error_averages_the_gap_over_target_levels() -> None:
    levels_df = pd.DataFrame(
        {
            "correlation": 0.0,
            "service_level": [0.5, 0.5, 0.9, 0.9],
            "y": [1.0, 3.0, 1.0, 3.0],
            "order_up_to": [2.0, 2.0, 2.0, 2.0],
        }
    )

    errors = calibration_error(levels_df, by=["correlation"])

    # service 50% at both targets: gaps 0 and 0.4
    assert errors[0.0] == pytest.approx(0.2)


def _sweep(cutoff: str, rho: float, in_stock: bool) -> pd.DataFrame:
    """Ten periods of one origin, all in stock or all out of stock at a 0.9 target."""
    return pd.DataFrame(
        {
            "unique_id": "a",
            "cutoff": pd.Timestamp(cutoff),
            "period": range(10),
            "y": 1.0 if in_stock else 3.0,
            "service_level": 0.9,
            "order_up_to": 2.0,
            "policy": "TS (correlated)",
            "correlation": rho,
        }
    )


def test_cross_validation_picks_the_correlation_calibrated_on_the_other_origins() -> None:
    # correlation 1 is calibrated on origin A only, correlation 0 on origin B only
    sweep_df = pd.concat(
        [
            _sweep("2020-01-01", rho=1.0, in_stock=True),
            _sweep("2020-01-01", rho=0.0, in_stock=False),
            _sweep("2020-02-01", rho=1.0, in_stock=False),
            _sweep("2020-02-01", rho=0.0, in_stock=True),
        ]
    )

    levels_df, chosen = cross_validate_correlation(sweep_df)

    assert chosen.to_dict() == {pd.Timestamp("2020-01-01"): 0.0, pd.Timestamp("2020-02-01"): 1.0}
    assert len(levels_df) == 20
    assert levels_df["policy"].eq("TS (fitted correlation)").all()
    assert "correlation" not in levels_df.columns


def test_cross_validation_needs_two_origins() -> None:
    with pytest.raises(ValueError, match="two origins"):
        cross_validate_correlation(_sweep("2020-01-01", rho=0.0, in_stock=True))


def test_periods_must_divide_the_horizon() -> None:
    with pytest.raises(ShapeMismatchError):
        quantile_order_up_to(_forecast_frame(days=10), alias="TS", days=7)


def test_demand_std_uses_only_days_before_the_origin(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    values = [0.0, 10.0] * 28 + [1000.0] * 5  # the spike comes after the origin
    y_df = make_long({"a": values}, start="2019-12-07")

    sigma = demand_std(y_df, cutoffs=[CUTOFF], history_days=56)

    assert sigma[("a", CUTOFF)] == pytest.approx(np.std([0.0, 10.0] * 28, ddof=1))


def test_normal_order_up_to_adds_z_sigma_sqrt_days(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    y_df = make_long({"a": [0.0, 10.0] * 28}, start="2019-12-07")
    sigma = np.std([0.0, 10.0] * 28, ddof=1)

    levels_df = normal_order_up_to(
        _forecast_frame(), model="M", y_df=y_df, days=7, service_levels=(0.5, 0.9), integer=False
    )

    by_level = levels_df.groupby("service_level")["order_up_to"].first()
    assert by_level[0.5] == pytest.approx(35.0)
    assert by_level[0.9] == pytest.approx(35.0 + 1.2815516 * sigma * np.sqrt(7))
    assert levels_df["policy"].eq("M + normal SS").all()


def test_service_metrics_counts_stockouts_served_units_and_leftover() -> None:
    levels_df = pd.DataFrame(
        {
            "policy": "p",
            "service_level": 0.9,
            "y": [4.0, 10.0, 0.0],
            "order_up_to": [5.0, 8.0, 2.0],
        }
    )

    metrics = service_metrics(levels_df).loc[("p", 0.9)]

    assert metrics["cycle service"] == pytest.approx(2 / 3)
    assert metrics["fill rate"] == pytest.approx((4 + 8) / 14)
    assert metrics["leftover per unit sold"] == pytest.approx((1 + 2) / 14)
