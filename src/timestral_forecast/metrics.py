"""Forecast accuracy metrics: WMAPE, bias, score, quantile loss and comparison tables."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def wmape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Weighted MAPE: total absolute error over total actual volume (equal to MAE%).

    Parameters
    ----------
    y : np.ndarray
        Actuals.
    y_hat : np.ndarray
        Forecasts, same shape as `y`.

    Returns
    -------
    float
        `sum |y_hat - y| / sum y`.
    """
    return float(np.abs(y_hat - y).sum() / y.sum())


def bias(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Bias%: total error over total volume; positive means over-forecasting.

    Parameters
    ----------
    y : np.ndarray
        Actuals.
    y_hat : np.ndarray
        Forecasts, same shape as `y`.

    Returns
    -------
    float
        `sum (y_hat - y) / sum y`.
    """
    return float((y_hat - y).sum() / y.sum())


def score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """VN1 score (Vandeput) = WMAPE + |Bias|; penalises models that win WMAPE by under-forecasting.

    Parameters
    ----------
    y : np.ndarray
        Actuals.
    y_hat : np.ndarray
        Forecasts, same shape as `y`.

    Returns
    -------
    float
        `wmape + |bias|`.
    """
    return wmape(y, y_hat) + abs(bias(y, y_hat))


def weighted_quantile_loss(y: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]) -> float:
    """Mean scaled pinball loss over quantile levels (WQL), as in GIFT-Eval.

    Parameters
    ----------
    y : np.ndarray
        `(n,)` actuals.
    quantiles : np.ndarray
        `(n, Q)` forecast quantiles.
    levels : Sequence[float]
        The Q probability levels.

    Returns
    -------
    float
        `2 * mean_q sum_i pinball_q / sum_i |y_i|`.
    """
    q = np.asarray(levels)[None, :]
    err = y[:, None] - quantiles
    pinball = np.maximum(q * err, (q - 1) * err).sum(axis=0)
    return float(2 * pinball.mean() / np.abs(y).sum())


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals falling inside `[lower, upper]`.

    Parameters
    ----------
    y : np.ndarray
        Actuals.
    lower, upper : np.ndarray
        Interval bounds, same shape as `y`.

    Returns
    -------
    float
        Empirical coverage in [0, 1].
    """
    return float(((y >= lower) & (y <= upper)).mean())


def evaluate(
    df: pd.DataFrame, models: Sequence[str], by: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute WMAPE, bias and score of each model column, optionally per group.

    Negative forecasts are clipped to zero, since demand cannot be negative.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with actuals in `y` and one column per model.
    models : Sequence[str]
        Model columns to evaluate.
    by : str | Sequence[str] | None
        Grouping columns (e.g. "cutoff", "segment").

    Returns
    -------
    pd.DataFrame
        One row per (group, model) with `wmape`, `bias`, `score`.
    """
    by_cols = [by] if isinstance(by, str) else list(by or [])
    groups = df.groupby(by_cols, observed=True) if by_cols else [((), df)]
    rows = []
    for key, part in groups:
        y = part["y"].to_numpy(dtype=float)
        keys = dict(zip(by_cols, key, strict=True))
        for model in models:
            y_hat = part[model].clip(lower=0).to_numpy(dtype=float)
            rows.append(
                {
                    **keys,
                    "model": model,
                    "wmape": wmape(y, y_hat),
                    "bias": bias(y, y_hat),
                    "score": score(y, y_hat),
                }
            )
    return pd.DataFrame(rows)


def fold_table(df: pd.DataFrame, models: Sequence[str], metric: str = "score") -> pd.DataFrame:
    """Pivot one metric to models × backtest origins, with the mean across origins.

    Parameters
    ----------
    df : pd.DataFrame
        Merged evaluation frame with a `cutoff` column.
    models : Sequence[str]
        Model columns to evaluate.
    metric : str
        "wmape", "bias" or "score".

    Returns
    -------
    pd.DataFrame
        One row per model, one column per origin plus `mean`, sorted by `mean`.
    """
    per_fold = evaluate(df, models=models, by="cutoff")
    table = per_fold.pivot(index="model", columns="cutoff", values=metric)
    table.columns = [c.date() for c in table.columns]
    table["mean"] = table.mean(axis=1)
    return table.sort_values("mean")


def forecast_value_added(table: pd.DataFrame, reference: str, column: str = "mean") -> pd.Series:
    """Forecast Value Added: relative error reduction of each model versus `reference`.

    Parameters
    ----------
    table : pd.DataFrame
        Error table indexed by model (e.g. the output of `fold_table`).
    reference : str
        Model used as the yardstick, usually a naive method.
    column : str
        Error column to compare.

    Returns
    -------
    pd.Series
        `1 - error / error(reference)`; positive means better than the reference.
    """
    return (1 - table[column] / table.loc[reference, column]).rename(f"FVA vs {reference}")


def aggregate_periods(df: pd.DataFrame, models: Sequence[str], days: int) -> pd.DataFrame:
    """Sum actuals and forecasts of each series over consecutive blocks of `days` per origin.

    Replenishment decisions are rarely daily: an order covers a week or a lead time.
    Errors on single days partly cancel within such a block, so evaluating at the
    decision level shows the accuracy the planner actually experiences.

    Parameters
    ----------
    df : pd.DataFrame
        Merged evaluation frame `unique_id, ds, cutoff, y, <models>`.
    models : Sequence[str]
        Model columns to aggregate.
    days : int
        Block length: 7 for weekly buckets, the horizon for the whole lead time.

    Returns
    -------
    pd.DataFrame
        `unique_id, cutoff, period, y, <models>`, one row per series, origin and block.
    """
    period = (df["ds"] - df["cutoff"]).dt.days.sub(1) // days
    cols = ["y", *models]
    values = df[cols].astype(float)
    keys = [df["unique_id"], df["cutoff"], period.rename("period")]
    return values.groupby(keys, observed=True).sum().reset_index()


def bootstrap_scores(
    df: pd.DataFrame, models: Sequence[str], n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Mean score per origin of each model on series-level bootstrap resamples.

    Each resample draws the series with replacement and keeps all their days and origins,
    so the uncertainty reflects which items happen to be in the portfolio. Paired
    differences between models (see `bootstrap_difference`) are much tighter than the
    spread of each score, because every model is scored on the same resample.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation frame `unique_id, cutoff, y, <models>` (daily or from `aggregate_periods`).
    models : Sequence[str]
        Model columns to score.
    n_boot : int
        Number of resamples.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        `(n_boot, len(models))` mean score per origin, one column per model.
    """
    models = list(models)
    values = df[["y", *models]].astype(float).clip(lower=0)
    abs_err = values[models].sub(values["y"], axis=0).abs().add_suffix("|abs")
    err = values[models].sub(values["y"], axis=0)
    parts = pd.concat([values[["y"]], err, abs_err], axis=1)
    sums = parts.groupby([df["unique_id"], df["cutoff"]], observed=True).sum()
    cube = sums.unstack("cutoff", fill_value=0.0)  # series x (column, cutoff)
    rng = np.random.default_rng(seed)
    n_series = len(cube)
    weights = rng.multinomial(n_series, np.full(n_series, 1 / n_series), size=n_boot)
    totals = weights @ cube.to_numpy()  # (n_boot, columns x cutoffs)
    totals = pd.DataFrame(totals, columns=cube.columns)
    volume = totals["y"].to_numpy()
    out = {}
    for model in models:
        wmape_b = totals[f"{model}|abs"].to_numpy() / volume
        bias_b = totals[model].to_numpy() / volume
        out[model] = (wmape_b + np.abs(bias_b)).mean(axis=1)
    return pd.DataFrame(out)


def bootstrap_difference(boot: pd.DataFrame, reference: str, level: float = 0.95) -> pd.DataFrame:
    """Paired score difference of each model versus `reference`, with a percentile interval.

    Parameters
    ----------
    boot : pd.DataFrame
        Output of `bootstrap_scores`.
    reference : str
        Model every other column is compared with.
    level : float
        Confidence level of the interval.

    Returns
    -------
    pd.DataFrame
        One row per model: median difference, interval bounds and the share of resamples
        in which the model scores better (lower) than `reference`.
    """
    diff = boot.sub(boot[reference], axis=0).drop(columns=reference)
    tail = (1 - level) / 2
    return pd.DataFrame(
        {
            "difference": diff.median(),
            "lower": diff.quantile(tail),
            "upper": diff.quantile(1 - tail),
            "share better": (diff < 0).mean(),
        }
    ).sort_values("difference")
