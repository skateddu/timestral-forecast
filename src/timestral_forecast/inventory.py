"""Order-up-to levels from forecasts and the service they achieve (one newsvendor per period).

For each series, origin and replenishment period (e.g. a week) the stock is raised to a
level S; the period is in stock when its demand does not exceed S. S comes either from the
forecast quantiles (Timestral) or from the classic planner rule, point forecast plus a
normal safety stock. Timestral gives quantiles per day, not for their sum, so the quantile
of a period is built under one of two bounding assumptions:

* "comonotonic": days perfectly correlated, the period quantile is the sum of the daily
  quantiles at the same level (widest spread, most stock);
* "independent": days independent, the period distribution is the convolution of the daily
  ones, estimated by sampling (narrowest spread, least stock);
* "correlated": in between, days linked by a Gaussian copula with one common factor, so that
  every pair of days has correlation `correlation` (0 is "independent", 1 "comonotonic").
  The correlation can be fitted on the backtest, origin by origin (`cross_validate_correlation`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from timestral_forecast.exceptions import ShapeMismatchError
from timestral_forecast.metrics import aggregate_periods

SERVICE_LEVELS = (0.80, 0.90, 0.95, 0.98)
HISTORY_DAYS = 56
KEYS = ["unique_id", "cutoff", "period"]
# denser near zero, where the backtest puts the correlation between days
CORRELATIONS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
METHODS = {"comonotonic", "independent", "correlated"}
# bounds the memory of the sampled array (periods x days x samples)
_MAX_SAMPLED_VALUES = 5_000_000


def quantile_levels(df: pd.DataFrame, alias: str) -> tuple[float, ...]:
    """Levels of the `<alias>_qXX` quantile columns of `df`, sorted.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation frame with quantile columns.
    alias : str
        Model column prefix.

    Returns
    -------
    tuple[float, ...]
        Quantile levels in (0, 1).
    """
    prefix = f"{alias}_q"
    suffixes = [c[len(prefix) :] for c in df.columns if c.startswith(prefix)]
    return tuple(sorted(int(s) / 100 for s in suffixes if s.isdigit()))


def _daily_blocks(
    df: pd.DataFrame, columns: Sequence[str], days: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Keys, `(n, days, len(columns))` daily values and period demand of each block."""
    period = (df["ds"] - df["cutoff"]).dt.days.sub(1) // days
    ordered = df.assign(period=period).sort_values([*KEYS, "ds"])
    sizes = ordered.groupby(KEYS, observed=True).size()
    if (sizes != days).any():
        raise ShapeMismatchError("period blocks", (len(sizes), days), expected=f"{days} days each")
    n = len(sizes)
    values = ordered[list(columns)].to_numpy(dtype=float).reshape(n, days, len(columns))
    demand = ordered["y"].to_numpy(dtype=float).reshape(n, days).sum(axis=1)
    return sizes.index.to_frame(index=False), values, demand


def _interpolate(quantiles: np.ndarray, levels: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linear interpolation of `(..., Q)` quantiles at `targets` levels, `(..., len(targets))`."""
    if targets.min() < levels[0] or targets.max() > levels[-1]:
        raise ValueError(f"service levels must lie within [{levels[0]}, {levels[-1]}]")
    idx = np.clip(np.searchsorted(levels, targets) - 1, 0, len(levels) - 2)
    w = (targets - levels[idx]) / (levels[idx + 1] - levels[idx])
    return quantiles[..., idx] * (1 - w) + quantiles[..., idx + 1] * w


def _copula_levels(
    rng: np.random.Generator, shape: tuple[int, int, int], correlation: float
) -> np.ndarray:
    """Uniform draws `(n, days, samples)` whose days share a common normal factor."""
    n, days, n_samples = shape
    common = rng.standard_normal(size=(n, 1, n_samples))
    own = rng.standard_normal(size=shape)
    return norm.cdf(np.sqrt(correlation) * common + np.sqrt(1 - correlation) * own)


def _sampled_period_quantiles(
    daily: np.ndarray,
    levels: np.ndarray,
    targets: np.ndarray,
    n_samples: int,
    seed: int,
    correlation: float = 0.0,
) -> np.ndarray:
    """Quantiles of the sum of days drawn from their quantile functions through a copula.

    Draws beyond the outermost levels (q01-q99) take the outermost quantile, so the
    tails are truncated. The same seed gives the same normal draws for every
    correlation, which keeps a sweep over correlations smooth.
    """
    rng = np.random.default_rng(seed)
    n, days, q = daily.shape
    chunk = max(1, _MAX_SAMPLED_VALUES // (days * n_samples))
    out = []
    for start in range(0, n, chunk):
        part = daily[start : start + chunk]
        u = _copula_levels(rng, shape=(len(part), days, n_samples), correlation=correlation)
        u = np.clip(u, levels[0], levels[-1])
        idx = np.clip(np.searchsorted(levels, u, side="right") - 1, 0, q - 2)
        w = (u - levels[idx]) / (levels[idx + 1] - levels[idx])
        lo = np.take_along_axis(part, idx, axis=-1)
        hi = np.take_along_axis(part, idx + 1, axis=-1)
        totals = (lo * (1 - w) + hi * w).sum(axis=1)
        out.append(np.quantile(totals, targets, axis=1).T)
    return np.concatenate(out, axis=0)


def _long_levels(
    keys: pd.DataFrame,
    demand: np.ndarray,
    order_up_to: np.ndarray,
    service_levels: Sequence[float],
    policy: str,
    integer: bool,
) -> pd.DataFrame:
    """Stack `(n, len(service_levels))` order-up-to levels into a long frame."""
    levels = np.clip(order_up_to, 0, None)
    if integer:
        # whole units; the tolerance keeps exact integers from rounding up
        levels = np.ceil(levels - 1e-9)
    frames = [
        keys.assign(y=demand, service_level=alpha, order_up_to=levels[:, k], policy=policy)
        for k, alpha in enumerate(service_levels)
    ]
    return pd.concat(frames, ignore_index=True)


def quantile_order_up_to(
    df: pd.DataFrame,
    alias: str,
    days: int,
    service_levels: Sequence[float] = SERVICE_LEVELS,
    method: str = "comonotonic",
    n_samples: int = 1000,
    seed: int = 0,
    integer: bool = True,
    correlation: float = 0.0,
) -> pd.DataFrame:
    """Order-up-to level of each period: the forecast quantile of its demand at each service level.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation frame `unique_id, ds, cutoff, y, <alias>_qXX...`, complete periods.
    alias : str
        Model whose quantile columns are used.
    days : int
        Period length; must divide the horizon.
    service_levels : Sequence[float]
        Target cycle service levels, within the outermost quantile levels.
    method : str
        "comonotonic" (sum of daily quantiles), "independent" (sampled convolution) or
        "correlated" (sampled through a one-factor Gaussian copula).
    n_samples : int
        Draws per period for the sampled methods.
    seed : int
        Random seed for the sampled methods.
    integer : bool
        Round the levels up to whole units.
    correlation : float
        Correlation between any two days for "correlated", in [0, 1].

    Returns
    -------
    pd.DataFrame
        `unique_id, cutoff, period, y, service_level, order_up_to, policy`, where `y` is the
        period demand and `policy` reads `<alias> (<method>)`.

    Raises
    ------
    ValueError
        If `method` is unknown, `correlation` is outside [0, 1] or a service level is
        outside the quantile grid.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}: use one of {sorted(METHODS)}")
    if not 0 <= correlation <= 1:
        raise ValueError(f"correlation must lie within [0, 1], got {correlation}")
    levels = quantile_levels(df, alias=alias)
    columns = [f"{alias}_q{round(lvl * 100):02d}" for lvl in levels]
    keys, daily, demand = _daily_blocks(df, columns=columns, days=days)
    # quantile crossing would break the interpolation
    daily = np.sort(daily, axis=-1)
    grid, targets = np.asarray(levels), np.asarray(service_levels, dtype=float)
    if method == "comonotonic":
        order_up_to = _interpolate(daily.sum(axis=1), levels=grid, targets=targets)
    else:
        order_up_to = _sampled_period_quantiles(
            daily,
            levels=grid,
            targets=targets,
            n_samples=n_samples,
            seed=seed,
            correlation=correlation if method == "correlated" else 0.0,
        )
    return _long_levels(
        keys,
        demand=demand,
        order_up_to=order_up_to,
        service_levels=service_levels,
        policy=f"{alias} ({method})",
        integer=integer,
    )


def demand_std(
    y_df: pd.DataFrame, cutoffs: Sequence[pd.Timestamp], history_days: int = HISTORY_DAYS
) -> pd.Series:
    """Standard deviation of daily demand over the last `history_days` before each origin.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    cutoffs : Sequence[pd.Timestamp]
        Origins.
    history_days : int
        Days of history used.

    Returns
    -------
    pd.Series
        Standard deviation indexed by `unique_id, cutoff` (0 for a single observation).
    """
    parts = []
    for cutoff in cutoffs:
        recent = y_df[y_df["ds"].between(cutoff - pd.Timedelta(days=history_days - 1), cutoff)]
        std = recent.groupby("unique_id", observed=True)["y"].std().fillna(0)
        parts.append(std.to_frame(name="sigma").assign(cutoff=cutoff))
    return pd.concat(parts).set_index("cutoff", append=True)["sigma"]


def normal_order_up_to(
    df: pd.DataFrame,
    model: str,
    y_df: pd.DataFrame,
    days: int,
    service_levels: Sequence[float] = SERVICE_LEVELS,
    history_days: int = HISTORY_DAYS,
    integer: bool = True,
) -> pd.DataFrame:
    """Classic planner rule: period point forecast plus z * sigma * sqrt(days).

    Sigma is the standard deviation of daily demand over the `history_days` before the
    origin, so the rule uses no information from the evaluation window.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation frame `unique_id, ds, cutoff, y, <model>`.
    model : str
        Point forecast column.
    y_df : pd.DataFrame
        Long target, for the demand variability before each origin.
    days : int
        Period length.
    service_levels : Sequence[float]
        Target cycle service levels.
    history_days : int
        Days of history for sigma.
    integer : bool
        Round the levels up to whole units.

    Returns
    -------
    pd.DataFrame
        Same layout as `quantile_order_up_to`, `policy` reading `<model> + normal SS`.
    """
    periods = aggregate_periods(df, models=[model], days=days)
    sigma = demand_std(y_df, cutoffs=sorted(df["cutoff"].unique()), history_days=history_days)
    index = pd.MultiIndex.from_frame(periods[["unique_id", "cutoff"]])
    sigma = sigma.reindex(index).fillna(0).to_numpy()
    z = norm.ppf(np.asarray(service_levels, dtype=float))
    order_up_to = periods[model].to_numpy()[:, None] + sigma[:, None] * np.sqrt(days) * z
    return _long_levels(
        periods[KEYS],
        demand=periods["y"].to_numpy(),
        order_up_to=order_up_to,
        service_levels=service_levels,
        policy=f"{model} + normal SS",
        integer=integer,
    )


def service_metrics(levels_df: pd.DataFrame) -> pd.DataFrame:
    """Service achieved by each policy and target level, and the stock it takes.

    Parameters
    ----------
    levels_df : pd.DataFrame
        Output of `quantile_order_up_to` / `normal_order_up_to`, possibly concatenated.

    Returns
    -------
    pd.DataFrame
        Indexed by `policy, service_level`: `cycle service` (share of periods with demand
        within S), `fill rate` (share of units served) and `leftover per unit sold`
        (stock left at the end of the period over total demand).
    """
    s, y = levels_df["order_up_to"], levels_df["y"]
    frame = levels_df.assign(
        in_stock=(y <= s).astype(float), served=np.minimum(y, s), leftover=(s - y).clip(lower=0)
    )
    totals = frame.groupby(["policy", "service_level"], observed=True)[
        ["in_stock", "served", "leftover", "y"]
    ].agg({"in_stock": "mean", "served": "sum", "leftover": "sum", "y": "sum"})
    return pd.DataFrame(
        {
            "cycle service": totals["in_stock"],
            "fill rate": totals["served"] / totals["y"],
            "leftover per unit sold": totals["leftover"] / totals["y"],
        }
    )


def correlation_sweep(
    df: pd.DataFrame,
    alias: str,
    days: int,
    correlations: Sequence[float] = CORRELATIONS,
    service_levels: Sequence[float] = SERVICE_LEVELS,
    n_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Order-up-to levels of the "correlated" method for every candidate correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation frame with the `<alias>_qXX` quantile columns.
    alias : str
        Model whose quantile columns are used.
    days : int
        Period length.
    correlations : Sequence[float]
        Candidate correlations between days, in [0, 1].
    service_levels : Sequence[float]
        Target cycle service levels.
    n_samples : int
        Draws per period.
    seed : int
        Random seed, shared by every correlation.

    Returns
    -------
    pd.DataFrame
        Layout of `quantile_order_up_to` plus a `correlation` column.
    """
    frames = [
        quantile_order_up_to(
            df,
            alias=alias,
            days=days,
            service_levels=service_levels,
            method="correlated",
            n_samples=n_samples,
            seed=seed,
            correlation=rho,
        ).assign(correlation=rho)
        for rho in correlations
    ]
    return pd.concat(frames, ignore_index=True)


def calibration_error(levels_df: pd.DataFrame, by: Sequence[str]) -> pd.Series:
    """Mean absolute gap between achieved and target cycle service, over the target levels.

    Parameters
    ----------
    levels_df : pd.DataFrame
        Order-up-to levels with `y`, `order_up_to`, `service_level` and the `by` columns.
    by : Sequence[str]
        Columns that define the groups (e.g. `correlation`, `cutoff`).

    Returns
    -------
    pd.Series
        Calibration error indexed by `by`, in service points (0.05 = five points off).
    """
    in_stock = (levels_df["y"] <= levels_df["order_up_to"]).astype(float)
    keys = [levels_df[c] for c in [*by, "service_level"]]
    service = in_stock.groupby(keys, observed=True).mean()
    gap = (service - service.index.get_level_values("service_level")).abs()
    return gap.groupby(level=list(by), observed=True).mean()


def cross_validate_correlation(sweep_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Leave-one-origin-out choice of the correlation between days.

    Each origin gets the correlation with the smallest calibration error on the other
    origins pooled, so no origin is scored with a correlation fitted on itself.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of `correlation_sweep`, with at least two origins.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Order-up-to levels with `policy` reading `<alias> (fitted correlation)`, and the
        correlation chosen for each origin.

    Raises
    ------
    ValueError
        If `sweep_df` has fewer than two origins.
    """
    cutoffs = sorted(sweep_df["cutoff"].unique())
    if len(cutoffs) < 2:
        raise ValueError(f"cross-validation needs at least two origins, got {len(cutoffs)}")
    chosen, parts = {}, []
    for cutoff in cutoffs:
        held_out = sweep_df["cutoff"] == cutoff
        rho = calibration_error(sweep_df[~held_out], by=["correlation"]).idxmin()
        chosen[cutoff] = rho
        parts.append(sweep_df[held_out & (sweep_df["correlation"] == rho)])
    levels_df = pd.concat(parts, ignore_index=True).drop(columns="correlation")
    policy = levels_df["policy"].str.replace("(correlated)", "(fitted correlation)", regex=False)
    chosen_s = pd.Series(chosen, name="correlation").rename_axis("cutoff")
    return levels_df.assign(policy=policy), chosen_s
