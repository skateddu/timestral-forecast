"""Run every model family on the same origins and merge forecasts with actuals."""

import hashlib
import logging
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path

import pandas as pd

from timestral_forecast.config import Settings
from timestral_forecast.data import M5Data
from timestral_forecast.models.baselines import (
    arima_models,
    backtest_statsforecast,
    baseline_models,
    exponential_smoothing_models,
)
from timestral_forecast.models.lgbm import LGBM_VARIANTS, LGBMVariant, backtest_lightgbm
from timestral_forecast.models.timestral_model import (
    TimestralVariant,
    backtest_timestral,
    load_timestral,
)
from timestral_forecast.preprocessing import (
    EVENT_CATEGORIES,
    KNOWN_COVARIATES,
    backtest_cutoffs,
    split_at,
    store_of,
    store_totals,
)

logger = logging.getLogger(__name__)

KEYS = ["unique_id", "ds", "cutoff"]
STORE_COVARIATES = ("snap", "is_event")
HOLIDAY_PREFIX = "holidays_"

Job = Callable[[], pd.DataFrame]


def timestral_alias(model_id: str, with_covariates: bool) -> str:
    """Return the short model name, e.g. `Timestral-28M+cov`.

    Parameters
    ----------
    model_id : str
        Hugging Face checkpoint id.
    with_covariates : bool
        Whether the run uses known covariates.

    Returns
    -------
    str
        Checkpoint name, with a `+cov` suffix for covariate runs.
    """
    return model_id.split("/")[-1] + ("+cov" if with_covariates else "")


def timestral_variants(settings: Settings) -> list[TimestralVariant]:
    """List the Timestral configurations of the benchmark.

    History-only for every checkpoint in `timestral_models`; known covariates for those
    in `timestral_covariate_models`; and, for the first covariate checkpoint, three
    extensions: an extra past-only covariate (store total), a multivariate run (affine
    items of the same store forecast jointly) and events as categorical covariates
    (event name and type in place of the 0/1 event row).

    Parameters
    ----------
    settings : Settings
        Experiment configuration.

    Returns
    -------
    list[TimestralVariant]
        One entry per Timestral job.
    """
    variants = [
        TimestralVariant(alias=timestral_alias(m, with_covariates=False), model_id=m)
        for m in settings.timestral_models
    ]
    covariate_variants = [
        TimestralVariant(
            alias=timestral_alias(model_id, with_covariates=True),
            model_id=model_id,
            known_covariates=KNOWN_COVARIATES,
            symmetric_averaging=settings.timestral_symmetric_averaging_with_covariates,
            mask_constant_price=settings.timestral_mask_constant_price,
        )
        for model_id in settings.timestral_covariate_models
    ]
    variants += covariate_variants
    if covariate_variants:
        cov = covariate_variants[0]
        variants.append(replace(cov, alias=f"{cov.alias}+past", past_covariates=True))
        variants.append(
            TimestralVariant(
                alias=f"{timestral_alias(cov.model_id, with_covariates=False)}-mv",
                model_id=cov.model_id,
                known_covariates=STORE_COVARIATES,
                past_covariates=True,
                group_size=settings.timestral_group_size,
            )
        )
        variants.append(
            replace(
                cov,
                alias=f"{cov.alias}+cat",
                known_covariates=tuple(c for c in cov.known_covariates if c != "is_event"),
                categorical_covariates=EVENT_CATEGORIES,
            )
        )
    return variants


def benchmark_jobs(
    settings: Settings, data: M5Data, cutoffs: Sequence[pd.Timestamp]
) -> dict[str, Job]:
    """Map each cache name to the function that produces its forecasts on `cutoffs`.

    Parameters
    ----------
    settings : Settings
        Experiment configuration.
    data : M5Data
        Loaded M5 subset.
    cutoffs : Sequence[pd.Timestamp]
        Backtest origins shared by every model.

    Returns
    -------
    dict[str, Job]
        Lazy jobs, so that cached results are never recomputed.
    """
    h, n_jobs = settings.horizon, settings.n_jobs
    statsforecast_families = {
        "baselines": (
            baseline_models(settings.season_length, window=settings.moving_average_window),
            None,
        ),
        "arima": (arima_models(settings.season_length), settings.arima_train_days),
        "ets": (exponential_smoothing_models(settings.season_length), settings.arima_train_days),
    }
    jobs: dict[str, Job] = {
        name: _statsforecast_job(
            data, cutoffs=cutoffs, horizon=h, n_jobs=n_jobs, models=models, train_days=days
        )
        for name, (models, days) in statsforecast_families.items()
    }
    for variant in LGBM_VARIANTS:
        jobs[variant.alias] = _lgbm_job(settings, data, cutoffs=cutoffs, variant=variant)
    # one store-total frame (~4.7M rows) shared by every variant that needs it
    past_df = cache(lambda: store_totals(data.y_df, s_df=data.s_df))
    for variant in timestral_variants(settings):
        jobs[variant.alias] = _timestral_job(
            settings, data, cutoffs=cutoffs, variant=variant, past_df=past_df
        )
    return jobs


def _statsforecast_job(
    data: M5Data,
    cutoffs: Sequence[pd.Timestamp],
    horizon: int,
    n_jobs: int,
    models: list[object],
    train_days: int | None,
) -> Job:
    """Bind the arguments of one statsforecast backtest."""
    return lambda: backtest_statsforecast(
        data.y_df,
        cutoffs=cutoffs,
        horizon=horizon,
        n_jobs=n_jobs,
        models=models,
        train_days=train_days,
    )


def _lgbm_job(
    settings: Settings, data: M5Data, cutoffs: Sequence[pd.Timestamp], variant: LGBMVariant
) -> Job:
    """Bind the arguments of one LightGBM backtest."""
    return lambda: backtest_lightgbm(
        data.y_df,
        x_df=data.x_df,
        s_df=data.s_df,
        cutoffs=cutoffs,
        horizon=settings.horizon,
        tweedie_power=settings.lgbm_tweedie_power,
        n_jobs=settings.n_jobs,
        variant=variant,
    )


def _timestral_job(
    settings: Settings,
    data: M5Data,
    cutoffs: Sequence[pd.Timestamp],
    variant: TimestralVariant,
    past_df: Job,
) -> Job:
    """Bind the arguments of one Timestral backtest; the checkpoint is loaded only when run."""

    def run() -> pd.DataFrame:
        forecaster = load_timestral(
            variant.model_id, device=settings.device, max_context=settings.timestral_max_context
        )
        fc, seconds = backtest_timestral(
            forecaster,
            variant=variant,
            y_df=data.y_df,
            cutoffs=cutoffs,
            horizon=settings.horizon,
            x_df=data.x_df,
            past_df=past_df() if variant.past_covariates else None,
            store_of=store_of(data.s_df),
            batch_size=settings.timestral_batch_size,
            context_window=settings.timestral_max_context,
        )
        logger.info(
            "Timestral inference time",
            extra={"alias": variant.alias, "seconds": round(seconds, 1)},
        )
        return fc

    return run


def cache_dir(settings: Settings, cutoffs: Sequence[pd.Timestamp]) -> Path:
    """Cache folder of one data subset, horizon and set of origins, so runs never mix.

    Parameters
    ----------
    settings : Settings
        Experiment configuration (department, stores, horizon).
    cutoffs : Sequence[pd.Timestamp]
        Backtest origins.

    Returns
    -------
    Path
        `processed/<dept>_<stores>_<digest>`, the digest of horizon and origins.
    """
    origins = ",".join(str(pd.Timestamp(c).date()) for c in cutoffs)
    digest = hashlib.sha1(f"h={settings.horizon}|{origins}".encode()).hexdigest()[:8]
    return settings.processed_dir / f"{settings.dept_id}_{'-'.join(settings.store_ids)}_{digest}"


def run_benchmark(
    settings: Settings,
    data: M5Data,
    cutoffs: Sequence[pd.Timestamp] | None = None,
    only: Collection[str] | None = None,
    cache_prefix: str = "",
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run (or read from cache) the benchmark jobs and merge their forecasts with the actuals.

    Parameters
    ----------
    settings : Settings
        Experiment configuration.
    data : M5Data
        Loaded M5 subset.
    cutoffs : Sequence[pd.Timestamp] | None
        Backtest origins; None uses the last `settings.n_windows` windows.
    only : Collection[str] | None
        Cache names of the jobs to run; None runs them all.
    cache_prefix : str
        Prefix of the cache file names; the folder (see `cache_dir`) already depends on
        subset, horizon and origins.
    refresh : bool
        Recompute every selected job even if cached.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Merged evaluation frame and compute seconds of the selected jobs (NaN when a
        cached result has no recorded time).
    """
    if cutoffs is None:
        cutoffs = backtest_cutoffs(
            data.y_df["ds"].max(), horizon=settings.horizon, n_windows=settings.n_windows
        )
    folder = cache_dir(settings, cutoffs=cutoffs)
    timings_path = folder / f"{cache_prefix}timings.json"
    timings = (
        pd.read_json(timings_path, typ="series")
        if timings_path.exists()
        else pd.Series(dtype=float)
    )
    jobs = {
        name: job
        for name, job in benchmark_jobs(settings, data, cutoffs=cutoffs).items()
        if only is None or name in only
    }
    frames = []
    for name, job in jobs.items():
        frame, seconds = cached(folder / f"{cache_prefix}fc_{name}.parquet", job, refresh=refresh)
        if seconds is not None:
            timings[name] = seconds
            timings.to_json(timings_path)
        frames.append(frame)
    truth = actuals(data.y_df, cutoffs=cutoffs, horizon=settings.horizon)
    return merge_forecasts(truth, forecasts=frames), timings.reindex(list(jobs))


def run_holiday_benchmark(
    settings: Settings,
    data: M5Data,
    only: Collection[str] | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the benchmark on the holiday origins, with their own job list and cache files.

    Parameters
    ----------
    settings : Settings
        Experiment configuration; uses `holiday_cutoffs` and `holiday_jobs`.
    data : M5Data
        Loaded M5 subset.
    only : Collection[str] | None
        Cache names of the jobs to run; None runs `settings.holiday_jobs`.
    refresh : bool
        Recompute every selected job even if cached.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Merged evaluation frame and compute seconds per job.
    """
    return run_benchmark(
        settings,
        data=data,
        cutoffs=[pd.Timestamp(c) for c in settings.holiday_cutoffs],
        only=only or settings.holiday_jobs,
        cache_prefix=HOLIDAY_PREFIX,
        refresh=refresh,
    )


def cached(path: Path, compute: Job, refresh: bool = False) -> tuple[pd.DataFrame, float | None]:
    """Load `path` if present, otherwise run `compute`, save the result and time it.

    Parameters
    ----------
    path : Path
        Parquet cache file.
    compute : Job
        Function producing the frame.
    refresh : bool
        Recompute even if the cache exists.

    Returns
    -------
    tuple[pd.DataFrame, float | None]
        The frame and the compute seconds (None when read from cache).
    """
    if path.exists() and not refresh:
        logger.info("Reading cached forecasts", extra={"path": str(path)})
        return pd.read_parquet(path), None
    start = time.perf_counter()
    frame = compute()
    seconds = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, seconds


def actuals(y_df: pd.DataFrame, cutoffs: Sequence[pd.Timestamp], horizon: int) -> pd.DataFrame:
    """Collect the evaluation windows of all series for every origin.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    cutoffs : Sequence[pd.Timestamp]
        Backtest origins.
    horizon : int
        Days in each evaluation window.

    Returns
    -------
    pd.DataFrame
        `unique_id, ds, cutoff, y`.
    """
    parts = [split_at(y_df, cutoff=c, horizon=horizon)[1].assign(cutoff=c) for c in cutoffs]
    return pd.concat(parts, ignore_index=True)[[*KEYS, "y"]]


def merge_forecasts(truth: pd.DataFrame, forecasts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Left-join every forecast frame onto the actuals.

    Series a model could not forecast (no sale before the origin) get 0, as they would
    in production; the rule is the same for every model so the comparison stays fair.
    Demand cannot be negative, so every forecast and quantile is clipped at 0 here,
    before metrics, plots or ensembles read it (AutoARIMA, for one, goes below zero).

    Parameters
    ----------
    truth : pd.DataFrame
        Output of `actuals`.
    forecasts : Sequence[pd.DataFrame]
        Frames keyed by `unique_id, ds, cutoff`, one or more model columns each.

    Returns
    -------
    pd.DataFrame
        Actuals plus every model column, without missing or negative forecasts.
    """
    merged = truth
    for fc in forecasts:
        merged = merged.merge(fc, on=KEYS, how="left")
    model_cols = merged.columns.difference([*KEYS, "y"])
    merged[model_cols] = merged[model_cols].fillna(0).clip(lower=0)
    return merged
