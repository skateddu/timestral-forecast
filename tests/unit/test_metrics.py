import numpy as np
import pandas as pd
import pytest

from timestral_forecast.metrics import (
    aggregate_periods,
    bias,
    bootstrap_difference,
    bootstrap_scores,
    coverage,
    evaluate,
    fold_table,
    forecast_value_added,
    score,
    weighted_quantile_loss,
    wmape,
)

ACTUAL = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 12.0, 0.0])


@pytest.mark.parametrize(
    ("forecast", "expected_wmape", "expected_bias"),
    [(np.zeros(8), 1.0, -1.0), (np.full(8, 2.75), 1.5, 0.0), (ACTUAL, 0.0, 0.0)],
    ids=["always-zero", "mean", "perfect"],
)
def test_wmape_and_bias_known_values_match(
    forecast: np.ndarray, expected_wmape: float, expected_bias: float
) -> None:
    assert wmape(ACTUAL, forecast) == pytest.approx(expected_wmape)
    assert bias(ACTUAL, forecast) == pytest.approx(expected_bias)


def test_score_penalises_zero_forecast_more_than_mean() -> None:
    assert score(ACTUAL, np.zeros(8)) > score(ACTUAL, np.full(8, ACTUAL.mean()))


def test_weighted_quantile_loss_perfect_quantiles_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    quantiles = np.repeat(y[:, None], 3, axis=1)

    assert weighted_quantile_loss(y, quantiles, levels=[0.1, 0.5, 0.9]) == pytest.approx(0.0)


def test_weighted_quantile_loss_median_equals_wmape() -> None:
    y = np.array([1.0, 4.0, 0.0, 5.0])
    median = np.array([[2.0], [3.0], [1.0], [5.0]])

    # pinball at q=0.5 is |e|/2, scaled by 2 / sum|y| -> WMAPE
    assert weighted_quantile_loss(y, median, levels=[0.5]) == pytest.approx(wmape(y, median[:, 0]))


def test_coverage_counts_inclusive_bounds() -> None:
    y = np.array([1.0, 5.0, 10.0, 20.0])

    assert coverage(y, lower=np.full(4, 1.0), upper=np.full(4, 10.0)) == pytest.approx(0.75)


def test_evaluate_negative_forecasts_are_clipped() -> None:
    df = pd.DataFrame({"y": [1.0, 1.0], "neg": [-5.0, 1.0], "zero": [0.0, 1.0]})

    out = evaluate(df, models=["neg", "zero"]).set_index("model")

    assert out.loc["neg", "wmape"] == pytest.approx(out.loc["zero", "wmape"])


def test_evaluate_by_group_returns_one_row_per_group_and_model() -> None:
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "y": [1.0, 2.0, 3.0], "m1": [1.0, 2.0, 3.0], "m2": [0.0, 0.0, 0.0]}
    )

    out = evaluate(df, models=["m1", "m2"], by="g")

    assert set(zip(out["g"], out["model"], strict=True)) == {
        ("a", "m1"),
        ("a", "m2"),
        ("b", "m1"),
        ("b", "m2"),
    }


def test_fold_table_sorted_by_mean_with_fva() -> None:
    cutoffs = pd.to_datetime(["2020-01-01", "2020-02-01"])
    df = pd.DataFrame(
        {
            "cutoff": np.repeat(cutoffs, 2),
            "y": [1.0, 1.0, 2.0, 2.0],
            "good": [1.0, 1.0, 2.0, 2.0],
            "bad": 0.0,
        }
    )

    table = fold_table(df, models=["bad", "good"])
    fva = forecast_value_added(table, reference="bad")

    assert list(table.index) == ["good", "bad"]
    assert fva["good"] == pytest.approx(1.0)


def test_aggregate_periods_sums_weekly_blocks_per_series_and_origin() -> None:
    cutoff = pd.Timestamp("2020-01-01")
    ds = pd.date_range("2020-01-02", periods=14, freq="D")
    df = pd.DataFrame(
        {"unique_id": "a", "ds": ds, "cutoff": cutoff, "y": 1.0, "M": [0.0] + [2.0] * 13}
    )

    out = aggregate_periods(df, models=["M"], days=7)

    assert out["period"].tolist() == [0, 1]
    assert out["y"].tolist() == [7.0, 7.0]
    assert out["M"].tolist() == [12.0, 14.0]


def test_aggregate_periods_daily_errors_cancel_at_weekly_level() -> None:
    cutoff = pd.Timestamp("2020-01-01")
    ds = pd.date_range("2020-01-02", periods=7, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": ds,
            "cutoff": cutoff,
            "y": [2.0, 0.0] * 3 + [1.0],
            "M": [0.0, 2.0] * 3 + [1.0],
        }
    )

    weekly = aggregate_periods(df, models=["M"], days=7)

    assert wmape(df["y"].to_numpy(), df["M"].to_numpy()) > 0
    assert wmape(weekly["y"].to_numpy(), weekly["M"].to_numpy()) == 0


def _portfolio(n_series: int = 30, n_days: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for cutoff in pd.to_datetime(["2020-01-01", "2020-02-01"]):
        for i in range(n_series):
            y = rng.poisson(3, size=n_days).astype(float)
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": f"s{i}",
                        "cutoff": cutoff,
                        "y": y,
                        "perfect": y,
                        "double": 2 * y,
                        "noisy": y + rng.normal(0, 1, size=n_days),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_scores_are_exact_where_the_score_does_not_depend_on_the_sample() -> None:
    boot = bootstrap_scores(_portfolio(), models=["perfect", "double"], n_boot=50)

    assert boot.shape == (50, 2)
    np.testing.assert_allclose(boot["perfect"], 0.0)
    np.testing.assert_allclose(boot["double"], 2.0)  # WMAPE 1 + bias 1 in every resample


def test_bootstrap_scores_centre_on_the_fold_table_score() -> None:
    df = _portfolio()

    boot = bootstrap_scores(df, models=["noisy"], n_boot=400)

    expected = fold_table(df, models=["noisy"]).loc["noisy", "mean"]
    assert boot["noisy"].median() == pytest.approx(expected, rel=0.05)
    assert boot["noisy"].std() > 0


def test_bootstrap_difference_ranks_models_against_the_reference() -> None:
    boot = bootstrap_scores(_portfolio(), models=["perfect", "double", "noisy"], n_boot=200)

    diff = bootstrap_difference(boot, reference="noisy")

    assert list(diff.index) == ["perfect", "double"]
    assert diff.loc["perfect", "share better"] == 1.0
    assert diff.loc["double", "share better"] == 0.0
    assert (diff["lower"] <= diff["difference"]).all()
    assert (diff["difference"] <= diff["upper"]).all()
