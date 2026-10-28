from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from timestral_forecast import backtest
from timestral_forecast.backtest import (
    actuals,
    cached,
    merge_forecasts,
    timestral_alias,
    timestral_variants,
)
from timestral_forecast.config import Settings
from timestral_forecast.data import M5Data
from timestral_forecast.models.timestral_model import TimestralVariant


def test_actuals_stacks_one_window_per_cutoff(make_long: Callable[..., pd.DataFrame]) -> None:
    df = make_long({"a": list(range(10)), "b": list(range(10))})
    cutoffs = [pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-07")]

    out = actuals(df, cutoffs=cutoffs, horizon=3)

    assert out.groupby("cutoff").size().tolist() == [6, 6]


def test_merge_forecasts_missing_series_get_zero(make_long: Callable[..., pd.DataFrame]) -> None:
    truth = actuals(
        make_long({"a": [1.0] * 4, "b": [2.0] * 4}),
        cutoffs=[pd.Timestamp("2020-01-02")],
        horizon=2,
    )
    fc = truth[truth["unique_id"] == "a"][["unique_id", "ds", "cutoff"]].assign(M=7.0)

    out = merge_forecasts(truth, forecasts=[fc])

    assert out.groupby("unique_id")["M"].first().to_dict() == {"a": 7.0, "b": 0.0}


def test_merge_forecasts_negative_values_are_clipped_to_zero(
    make_long: Callable[..., pd.DataFrame],
) -> None:
    truth = actuals(make_long({"a": [1.0] * 4}), cutoffs=[pd.Timestamp("2020-01-02")], horizon=2)
    fc = truth[["unique_id", "ds", "cutoff"]].assign(M=[-3.0, 2.0], M_q10=[-1.0, 0.5])

    out = merge_forecasts(truth, forecasts=[fc])

    assert out["M"].tolist() == [0.0, 2.0]
    assert out["M_q10"].tolist() == [0.0, 0.5]


def test_cached_reads_existing_file_without_computing(tmp_path: Path) -> None:
    path = tmp_path / "fc.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)

    def fail() -> pd.DataFrame:
        raise AssertionError("should not compute")

    frame, seconds = cached(path, compute=fail)

    assert frame["x"].tolist() == [1]
    assert seconds is None


def test_cached_computes_and_saves(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "fc.parquet"

    frame, seconds = cached(path, compute=lambda: pd.DataFrame({"x": [2]}))

    assert path.exists()
    assert seconds >= 0


def test_timestral_alias_strips_org_and_marks_covariates() -> None:
    assert timestral_alias("mistralai/Timestral-28M", with_covariates=True) == "Timestral-28M+cov"


def test_run_benchmark_merges_jobs_and_records_timings(
    tmp_path: Path, make_long: Callable[..., pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    y_df = make_long({"a": [1.0] * 10})
    data = M5Data(y_df=y_df, x_df=pd.DataFrame(), s_df=pd.DataFrame())
    settings = Settings(data_dir=tmp_path, horizon=2, n_windows=1)
    truth = actuals(y_df, cutoffs=[y_df["ds"].max() - pd.Timedelta(days=2)], horizon=2)
    fake_jobs = {"m1": lambda: truth[["unique_id", "ds", "cutoff"]].assign(M1=1.0)}
    monkeypatch.setattr(backtest, "benchmark_jobs", lambda settings, data, cutoffs: fake_jobs)

    merged, timings = backtest.run_benchmark(settings, data=data)

    assert merged["M1"].eq(1.0).all()
    assert list(timings.index) == ["m1"]
    folder = backtest.cache_dir(settings, cutoffs=sorted(truth["cutoff"].unique()))
    assert (folder / "fc_m1.parquet").exists()


def test_run_benchmark_only_and_prefix_select_jobs_and_cache_files(
    tmp_path: Path, make_long: Callable[..., pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    y_df = make_long({"a": [1.0] * 10})
    data = M5Data(y_df=y_df, x_df=pd.DataFrame(), s_df=pd.DataFrame())
    cutoffs = [pd.Timestamp("2020-01-03")]
    keys = actuals(y_df, cutoffs=cutoffs, horizon=2)[["unique_id", "ds", "cutoff"]]
    fake_jobs = {"m1": lambda: keys.assign(M1=1.0), "m2": lambda: keys.assign(M2=2.0)}
    monkeypatch.setattr(backtest, "benchmark_jobs", lambda settings, data, cutoffs: fake_jobs)

    settings = Settings(data_dir=tmp_path, horizon=2)

    merged, _ = backtest.run_benchmark(
        settings,
        data=data,
        cutoffs=cutoffs,
        only=["m2"],
        cache_prefix="h_",
    )

    assert "M1" not in merged.columns
    assert merged["cutoff"].unique().tolist() == cutoffs
    assert (backtest.cache_dir(settings, cutoffs=cutoffs) / "h_fc_m2.parquet").exists()


def test_run_holiday_benchmark_uses_holiday_origins_jobs_and_prefix(
    tmp_path: Path, make_long: Callable[..., pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {}
    monkeypatch.setattr(backtest, "run_benchmark", lambda settings, **kwargs: calls.update(kwargs))
    settings = Settings(data_dir=tmp_path, holiday_cutoffs=["2020-01-03"], holiday_jobs=["m2"])

    backtest.run_holiday_benchmark(settings, data=None)

    assert calls["cutoffs"] == [pd.Timestamp("2020-01-03")]
    assert calls["only"] == ("m2",)
    assert calls["cache_prefix"] == backtest.HOLIDAY_PREFIX


def test_benchmark_jobs_lists_every_model_family(make_long: Callable[..., pd.DataFrame]) -> None:
    data = M5Data(y_df=make_long({"a": [1.0] * 10}), x_df=pd.DataFrame(), s_df=pd.DataFrame())

    jobs = backtest.benchmark_jobs(Settings(), data=data, cutoffs=[pd.Timestamp("2020-01-05")])

    assert list(jobs) == [
        "baselines",
        "arima",
        "ets",
        "LightGBM",
        "LightGBM-norm",
        "LightGBM-norm-direct",
        "LightGBM-norm-matched",
        "LightGBM-norm-matched+cat",
        "Timestral-28M",
        "Timestral-28M+cov",
        "Timestral-28M+cov+past",
        "Timestral-28M-mv",
        "Timestral-28M+cov+cat",
    ]


def test_timestral_variants_extensions_reuse_covariate_safeguards() -> None:
    variants = {
        v.alias: v for v in timestral_variants(Settings(timestral_mask_constant_price=False))
    }

    past, mv = variants["Timestral-28M+cov+past"], variants["Timestral-28M-mv"]

    assert past.past_covariates and not past.mask_constant_price
    assert mv.group_size == Settings().timestral_group_size
    assert "sell_price" not in mv.known_covariates


def test_timestral_variants_without_covariate_models_has_no_extensions() -> None:
    variants = timestral_variants(Settings(timestral_covariate_models=()))

    assert [v.alias for v in variants] == ["Timestral-28M"]


@pytest.mark.parametrize(
    ("alias", "settings_kwargs", "expected"),
    [
        ("Timestral-28M", {}, (True, False)),
        ("Timestral-28M+cov", {}, (True, True)),
        ("Timestral-28M+cov", {"timestral_mask_constant_price": False}, (True, False)),
        (
            "Timestral-28M+cov",
            {"timestral_symmetric_averaging_with_covariates": False},
            (False, True),
        ),
    ],
    ids=["history-only", "cov-defaults", "cov-no-mask", "cov-no-symmetric"],
)
def test_timestral_jobs_pass_covariate_safeguards(
    make_long: Callable[..., pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    settings_kwargs: dict[str, bool],
    expected: tuple[bool, bool],
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_backtest(
        *args: object, variant: TimestralVariant, **kwargs: object
    ) -> tuple[pd.DataFrame, float]:
        calls.append((variant.symmetric_averaging, variant.mask_constant_price))
        return pd.DataFrame(), 0.0

    monkeypatch.setattr(backtest, "load_timestral", lambda *args, **kwargs: object())
    monkeypatch.setattr(backtest, "backtest_timestral", fake_backtest)
    data = M5Data(
        y_df=make_long({"a": [1.0] * 10}),
        x_df=pd.DataFrame(),
        s_df=pd.DataFrame({"unique_id": ["a"], "store_id": ["S1"]}),
    )
    settings = Settings(timestral_models=("mistralai/Timestral-28M",), **settings_kwargs)

    backtest.benchmark_jobs(settings, data=data, cutoffs=[pd.Timestamp("2020-01-05")])[alias]()

    assert calls == [expected]


def test_timestral_variants_categorical_events_replace_the_event_flag() -> None:
    variants = {v.alias: v for v in timestral_variants(Settings())}

    cat, cov = variants["Timestral-28M+cov+cat"], variants["Timestral-28M+cov"]

    assert cat.categorical_covariates == ("event_name", "event_type")
    assert cat.known_covariates == ("sell_price", "snap")
    assert cat.mask_constant_price == cov.mask_constant_price
    assert cat.symmetric_averaging == cov.symmetric_averaging


def test_cache_dir_depends_on_subset_horizon_and_origins(tmp_path: Path) -> None:
    base = Settings(data_dir=tmp_path)
    cutoffs = [pd.Timestamp("2016-04-24")]

    folder = backtest.cache_dir(base, cutoffs=cutoffs)

    assert folder.parent == base.processed_dir
    assert folder.name.startswith("FOODS_3_CA_1-TX_1-WI_1_")
    assert folder == backtest.cache_dir(Settings(data_dir=tmp_path), cutoffs=cutoffs)
    assert folder != backtest.cache_dir(base, cutoffs=[pd.Timestamp("2016-03-27")])
    assert folder != backtest.cache_dir(Settings(data_dir=tmp_path, horizon=7), cutoffs=cutoffs)
    assert folder != backtest.cache_dir(
        Settings(data_dir=tmp_path, store_ids=("CA_1",)), cutoffs=cutoffs
    )
