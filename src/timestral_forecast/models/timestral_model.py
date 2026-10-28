"""Timestral (Mistral AI) zero-shot forecaster wrapped for batched backtests."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from timestral_forecast.multivariate import group_by_affinity
from timestral_forecast.preprocessing import (
    TimestralInputs,
    build_timestral_arrays,
    mask_constant_covariate,
)

logger = logging.getLogger(__name__)


class QuantileForecaster(Protocol):
    """Structural type of `timestral.TimestralForecaster`, so tests can pass a fake."""

    def forecast(self, context: np.ndarray, prediction_length: int, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TimestralVariant:
    """One Timestral configuration of the benchmark.

    Attributes
    ----------
    alias : str
        Prefix of the output columns.
    model_id : str
        Hugging Face checkpoint, e.g. "mistralai/Timestral-28M".
    known_covariates : tuple[str, ...]
        Columns of `x_df` passed as known-future covariates (calendar encodings are
        added automatically); empty for the history-only variant.
    past_covariates : bool
        Add the store's daily department total as a past-only covariate.
    group_size : int
        Variates per batch element: 1 is univariate, more forecasts affine items of
        the same store jointly (only store-level covariates are allowed then).
    symmetric_averaging : bool
        Average with the mirrored forecast of `-context` (Timestral's default); the mirror
        negates the covariates as well.
    mask_constant_price : bool
        Blank the price row of series whose price never moved in the context (see
        `mask_constant_covariate`).
    """

    alias: str
    model_id: str
    known_covariates: tuple[str, ...] = ()
    past_covariates: bool = False
    group_size: int = 1
    symmetric_averaging: bool = True
    mask_constant_price: bool = False


def resolve_device(device: str) -> str:
    """Map "auto" to "cuda" when a GPU is visible, else "cpu".

    Parameters
    ----------
    device : str
        "auto", "cuda" or "cpu".

    Returns
    -------
    str
        A concrete torch device name.
    """
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_timestral(
    model_id: str, device: str = "cpu", max_context: int | None = None
) -> QuantileForecaster:
    """Download (first time) and load a Timestral checkpoint from Hugging Face.

    Parameters
    ----------
    model_id : str
        Hub repo id, e.g. "mistralai/Timestral-28M".
    device : str
        "auto", "cuda" or "cpu".
    max_context : int | None
        Most recent steps kept as context; None uses the trained maximum (32,768).

    Returns
    -------
    QuantileForecaster
        A `timestral.TimestralForecaster`.
    """
    from timestral import TimestralForecaster  # heavy import (torch), loaded only when needed

    return TimestralForecaster.from_pretrained(
        model_id, device=resolve_device(device), max_context=max_context
    )


def quantile_column(alias: str, level: float) -> str:
    """Return the column name of one quantile, e.g. `Timestral-28M_q10`.

    Parameters
    ----------
    alias : str
        Model column prefix.
    level : float
        Quantile level in (0, 1).

    Returns
    -------
    str
        `<alias>_qXX`, with the level in percent and zero-padded.
    """
    return f"{alias}_q{round(level * 100):02d}"


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor (or array-like) to a NumPy array on the CPU."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def predict_quantiles(
    forecaster: QuantileForecaster,
    inputs: TimestralInputs,
    horizon: int,
    batch_size: int,
    symmetric_averaging: bool = True,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Run the forecaster in chunks of `batch_size` batch elements.

    Parameters
    ----------
    forecaster : QuantileForecaster
        Loaded Timestral model.
    inputs : TimestralInputs
        Context `(B, V, T)` and optional known and past covariates.
    horizon : int
        Days to forecast.
    batch_size : int
        Batch elements per forward pass.
    symmetric_averaging : bool
        Average with the mirrored forecast of `-context` (Timestral's default); the mirror
        negates the covariates as well.

    Returns
    -------
    tuple[np.ndarray, tuple[float, ...]]
        `(B * V, H, Q)` quantiles, aligned with `inputs.unique_ids`, and the Q levels.
    """
    chunks, levels = [], ()
    for start in range(0, len(inputs.context), batch_size):
        part = slice(start, start + batch_size)
        kwargs: dict[str, Any] = {"symmetric_averaging": symmetric_averaging}
        if inputs.known_covariates is not None:
            kwargs["known_covariates"] = inputs.known_covariates[part]
        if inputs.past_covariates is not None:
            kwargs["past_covariates"] = inputs.past_covariates[part]
        fc = forecaster.forecast(inputs.context[part], prediction_length=horizon, **kwargs)
        quantiles = _to_numpy(fc.quantiles)
        chunks.append(quantiles.reshape(-1, *quantiles.shape[2:]))
        levels = tuple(fc.quantile_levels)
    return np.concatenate(chunks, axis=0), levels


def quantiles_to_frame(
    quantiles: np.ndarray, levels: Sequence[float], inputs: TimestralInputs, alias: str
) -> pd.DataFrame:
    """Build a long frame with the median, the quantile average and all quantiles.

    The average of 21 evenly spread quantiles approximates the predictive mean; on
    intermittent series the median sits well below it, which shows up as negative bias.

    Parameters
    ----------
    quantiles : np.ndarray
        `(N, H, Q)` quantiles, aligned with `inputs.unique_ids`.
    levels : Sequence[float]
        The Q quantile levels.
    inputs : TimestralInputs
        Inputs that produced the forecast (series ids and forecast dates).
    alias : str
        Column prefix.

    Returns
    -------
    pd.DataFrame
        `unique_id, ds, <alias>_qXX..., <alias>` (median) and `<alias>-mean`.
    """
    n, h, q = quantiles.shape
    flat = quantiles.reshape(n * h, q)
    frame = pd.DataFrame(flat, columns=[quantile_column(alias, lvl) for lvl in levels])
    frame.insert(0, "unique_id", np.repeat(inputs.unique_ids, h))
    frame.insert(1, "ds", np.tile(inputs.future_ds.to_numpy(), n))
    frame[alias] = flat[:, list(levels).index(0.5)]
    frame[f"{alias}-mean"] = flat.mean(axis=1)
    return frame


def prepare_inputs(
    variant: TimestralVariant,
    y_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    x_df: pd.DataFrame | None = None,
    past_df: pd.DataFrame | None = None,
    store_of: pd.Series | None = None,
    context_window: int | None = None,
) -> list[TimestralInputs]:
    """Build the model inputs of one origin according to `variant`.

    Parameters
    ----------
    variant : TimestralVariant
        Covariates, grouping and safeguards to apply.
    y_df : pd.DataFrame
        Long target.
    cutoff : pd.Timestamp
        Last context date.
    horizon : int
        Days to forecast.
    x_df : pd.DataFrame | None
        Known covariates; required when `variant.known_covariates` is not empty.
    past_df : pd.DataFrame | None
        Past-only covariates; required when `variant.past_covariates` is set.
    store_of : pd.Series | None
        Store id by `unique_id`; required when `variant.group_size > 1`.
    context_window : int | None
        Context steps the model reads, used by the constant-price check.

    Returns
    -------
    list[TimestralInputs]
        One element for univariate variants, one per group size for multivariate ones.
    """
    inputs = build_timestral_arrays(
        y_df,
        cutoff=cutoff,
        horizon=horizon,
        x_df=x_df if variant.known_covariates else None,
        covariates=variant.known_covariates,
        past_df=past_df if variant.past_covariates else None,
    )
    if variant.mask_constant_price:
        window = context_window or inputs.context.shape[-1]
        inputs, n_masked = mask_constant_covariate(inputs, name="sell_price", window=window)
        logger.info("Constant price masked", extra={"alias": variant.alias, "n_series": n_masked})
    if variant.group_size > 1:
        return group_by_affinity(inputs, store_of=store_of, size=variant.group_size)
    return [inputs]


def backtest_timestral(
    forecaster: QuantileForecaster,
    variant: TimestralVariant,
    y_df: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    horizon: int,
    x_df: pd.DataFrame | None = None,
    past_df: pd.DataFrame | None = None,
    store_of: pd.Series | None = None,
    batch_size: int = 64,
    context_window: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Zero-shot forecast from every origin with the configuration of `variant`.

    Parameters
    ----------
    forecaster : QuantileForecaster
        Loaded Timestral model.
    variant : TimestralVariant
        Covariates, grouping and safeguards to apply.
    y_df : pd.DataFrame
        Long target.
    cutoffs : Sequence[pd.Timestamp]
        Backtest origins.
    horizon : int
        Days to forecast.
    x_df, past_df, store_of
        See `prepare_inputs`.
    batch_size : int
        Batch elements per forward pass.
    context_window : int | None
        Context steps the model reads (its `max_context`).

    Returns
    -------
    tuple[pd.DataFrame, float]
        Forecast frame (`unique_id, ds, cutoff, <alias>, <alias>-mean, <alias>_qXX`) and
        total inference seconds (data preparation excluded).
    """
    out, seconds = [], 0.0
    for cutoff in cutoffs:
        parts = prepare_inputs(
            variant,
            y_df=y_df,
            cutoff=cutoff,
            horizon=horizon,
            x_df=x_df,
            past_df=past_df,
            store_of=store_of,
            context_window=context_window,
        )
        for inputs in parts:
            start = time.perf_counter()
            quantiles, levels = predict_quantiles(
                forecaster,
                inputs=inputs,
                horizon=horizon,
                batch_size=batch_size,
                symmetric_averaging=variant.symmetric_averaging,
            )
            seconds += time.perf_counter() - start
            frame = quantiles_to_frame(
                quantiles, levels=levels, inputs=inputs, alias=variant.alias
            )
            out.append(frame.assign(cutoff=cutoff))
        logger.info(
            "Timestral fold done", extra={"alias": variant.alias, "cutoff": str(cutoff.date())}
        )
    return pd.concat(out, ignore_index=True), seconds
