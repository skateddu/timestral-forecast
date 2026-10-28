"""Charts for the benchmark: data overview, Timestral inputs, forecasts and error tables."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from timestral_forecast.preprocessing import TimestralInputs, store_of

FAMILY_COLORS = {
    "baseline": "#9aa0a6",
    "statistical": "#5f6b7a",
    "lightgbm": "#2a9d8f",
    "timestral": "#e76f51",
    "ensemble": "#6a4c93",
}
STATISTICAL_MODELS = {"AutoARIMA", "AutoETS", "AutoCES"}
FIG_WIDTH = 12
HISTORY_COLOR = "#264653"


def model_family(model: str) -> str:
    """Map a model column to its family, used to colour charts consistently.

    Parameters
    ----------
    model : str
        Model column name.

    Returns
    -------
    str
        One of the keys of `FAMILY_COLORS`.
    """
    if model.startswith("Timestral"):
        return "timestral"
    if model.startswith("LightGBM"):
        return "lightgbm"
    if model.startswith("Ens"):
        return "ensemble"
    if model in STATISTICAL_MODELS:
        return "statistical"
    return "baseline"


def plot_sales_overview(
    y_df: pd.DataFrame, s_df: pd.DataFrame, windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]]
) -> Figure:
    """Plot weekly department sales per store, shading the backtest windows.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    s_df : pd.DataFrame
        Static attributes with `unique_id, store_id`.
    windows : Sequence[tuple[pd.Timestamp, pd.Timestamp]]
        Start and end of each evaluation window.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    store = y_df["unique_id"].map(store_of(s_df))
    weekly = (
        y_df.groupby([store.rename("store"), pd.Grouper(key="ds", freq="W")])["y"].sum().unstack(0)
    )
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))
    weekly.plot(ax=ax, lw=1)
    for start, end in windows:
        ax.axvspan(start, end, color="#e76f51", alpha=0.2, lw=0)
    ax.set_title("Weekly FOODS_3 sales per store (shaded: backtest windows)")
    ax.set_xlabel("")
    ax.set_ylabel("units / week")
    return fig


def plot_segment_examples(y_df: pd.DataFrame, examples: pd.Series, n_days: int = 182) -> Figure:
    """Plot the last `n_days` of one example series per demand segment.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    examples : pd.Series
        Series id indexed by segment name.
    n_days : int
        Days shown.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(
        len(examples), 1, figsize=(FIG_WIDTH, 1.7 * len(examples)), sharex=True
    )
    for ax, (segment, uid) in zip(np.atleast_1d(axes), examples.items(), strict=True):
        series = y_df[y_df["unique_id"] == uid].tail(n_days)
        ax.bar(series["ds"], series["y"], width=1, color=HISTORY_COLOR)
        ax.set_title(f"{segment}: {uid}", loc="left", fontsize=9)
    fig.suptitle(f"One series per demand segment, last {n_days} days")
    return fig


def plot_timestral_inputs(inputs: TimestralInputs, unique_id: str, n_days: int = 120) -> Figure:
    """Plot what Timestral receives for one series: target context and every covariate row.

    Parameters
    ----------
    inputs : TimestralInputs
        Univariate arrays of one origin.
    unique_id : str
        Series to show.
    n_days : int
        Context days shown before the origin.

    Returns
    -------
    Figure
        One panel for the target and one per covariate row.
    """
    i = inputs.index_of(unique_id)
    t, h = inputs.context.shape[-1], len(inputs.future_ds)
    dates = inputs.context_ds.append(inputs.future_ds)
    names = inputs.covariate_names
    fig, axes = plt.subplots(
        1 + len(names), 1, figsize=(FIG_WIDTH, 1.1 * (1 + len(names)) + 0.6), sharex=True
    )
    axes[0].plot(dates[t - n_days : t], inputs.context[i, 0, -n_days:], color=HISTORY_COLOR, lw=1)
    axes[0].set_ylabel("target", rotation=0, ha="right", fontsize=8)
    for ax, row, name in zip(axes[1:], inputs.known_covariates[i], names, strict=True):
        ax.plot(dates[t - n_days :], row[t - n_days :], color="#2a9d8f", lw=1)
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)
    for ax in axes:
        ax.axvspan(inputs.future_ds[0], dates[t + h - 1], color="#e76f51", alpha=0.15, lw=0)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Timestral input for {unique_id}: context (B, 1, T) and known covariates (B, C, T + H)"
    )
    return fig


def plot_series_forecast(
    history: pd.DataFrame, window: pd.DataFrame, alias: str, unique_id: str, n_history: int = 112
) -> Axes:
    """Plot recent history, actuals and a Timestral forecast with q10-q90 and q05-q95 bands.

    Parameters
    ----------
    history : pd.DataFrame
        Long target `unique_id, ds, y` before the origin.
    window : pd.DataFrame
        Merged evaluation frame for one origin (actuals and quantile columns).
    alias : str
        Timestral column prefix.
    unique_id : str
        Series to plot.
    n_history : int
        Days of history shown.

    Returns
    -------
    Axes
        The matplotlib axes.
    """
    hist = history[history["unique_id"] == unique_id].tail(n_history)
    fc = window[window["unique_id"] == unique_id].sort_values("ds")
    _, ax = plt.subplots(figsize=(FIG_WIDTH, 4))
    ax.plot(hist["ds"], hist["y"], color="#333333", lw=1, label="history")
    ax.plot(fc["ds"], fc["y"], color="#333333", lw=1, ls=":", label="actual")
    ax.fill_between(
        fc["ds"],
        fc[f"{alias}_q05"],
        fc[f"{alias}_q95"],
        color="#e76f51",
        alpha=0.12,
        label="q05-q95",
    )
    ax.fill_between(
        fc["ds"],
        fc[f"{alias}_q10"],
        fc[f"{alias}_q90"],
        color="#e76f51",
        alpha=0.25,
        label="q10-q90",
    )
    ax.plot(fc["ds"], fc[alias], color="#e76f51", lw=1.5, label="median")
    ax.plot(fc["ds"], fc[f"{alias}-mean"], color="#e76f51", lw=1.5, ls="--", label="mean")
    ax.set_title(f"{unique_id} - {alias}")
    ax.legend(loc="upper left", fontsize=8, ncol=6)
    return ax


def plot_wmape_bias(summary: pd.DataFrame) -> Axes:
    """Scatter WMAPE against bias per model, coloured by family.

    Parameters
    ----------
    summary : pd.DataFrame
        Indexed by model, with columns `wmape` and `bias`.

    Returns
    -------
    Axes
        The matplotlib axes.
    """
    _, ax = plt.subplots(figsize=(FIG_WIDTH, 6))
    for model, row in summary.iterrows():
        ax.scatter(
            row["bias"] * 100, row["wmape"] * 100, s=60, color=FAMILY_COLORS[model_family(model)]
        )
        ax.annotate(
            model,
            (row["bias"] * 100, row["wmape"] * 100),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Bias %  (under-forecast <- | -> over-forecast)")
    ax.set_ylabel("WMAPE %")
    ax.set_title("WMAPE / bias trade-off (all origins)")
    return ax


def plot_score_bars(summary: pd.DataFrame, title: str, metric: str = "score") -> Axes:
    """Horizontal bars of one metric per model, sorted, coloured by family.

    Parameters
    ----------
    summary : pd.DataFrame
        Indexed by model.
    title : str
        Chart title.
    metric : str
        Column to plot.

    Returns
    -------
    Axes
        The matplotlib axes.
    """
    data = summary.sort_values(metric, ascending=False)
    colors = [FAMILY_COLORS[model_family(m)] for m in data.index]
    _, ax = plt.subplots(figsize=(FIG_WIDTH, 0.4 * len(data) + 1))
    bars = ax.barh(data.index, data[metric], color=colors)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in data[metric]], padding=3, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_ylabel("")
    ax.set_title(title)
    return ax


def plot_heatmap(table: pd.DataFrame, xlabel: str, title: str) -> Axes:
    """Heatmap of a model × group error table (lower is greener).

    Parameters
    ----------
    table : pd.DataFrame
        Models as rows, groups (segments, origins, levels) as columns.
    xlabel : str
        Label of the column axis.
    title : str
        Chart title.

    Returns
    -------
    Axes
        The matplotlib axes.
    """
    _, ax = plt.subplots(figsize=(FIG_WIDTH, 0.4 * len(table) + 1.5))
    sns.heatmap(table, annot=True, fmt=".0%", cmap="RdYlGn_r", cbar=False, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)
    return ax
