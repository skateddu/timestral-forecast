"""Backtest splits and conversion of long data into Timestral input arrays."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from timestral_forecast.exceptions import ShapeMismatchError

KNOWN_COVARIATES = ("sell_price", "snap", "is_event")
CALENDAR_COVARIATES = ("dow_sin", "dow_cos", "doy_sin", "doy_cos")


@dataclass(frozen=True)
class TimestralInputs:
    """Arrays for `TimestralForecaster.forecast`, plus the labels needed to read the output.

    Attributes
    ----------
    unique_ids : np.ndarray
        `(B * V,)` series ids in row-major order of the `(B, V)` context axes; with one
        variate per batch element (V = 1) they are simply aligned with B.
    context : np.ndarray
        `(B, V, T)` target history; NaN before launch or when missing.
    known_covariates : np.ndarray | None
        `(B, C, T + H)` covariates known over context and horizon, or None.
    covariate_names : tuple[str, ...]
        Names of the C known covariate rows.
    future_ds : pd.DatetimeIndex
        The H forecast dates.
    past_covariates : np.ndarray | None
        `(B, P, T)` covariates observed over the context only, or None.
    """

    unique_ids: np.ndarray
    context: np.ndarray
    known_covariates: np.ndarray | None
    covariate_names: tuple[str, ...]
    future_ds: pd.DatetimeIndex
    past_covariates: np.ndarray | None = None

    @property
    def context_ds(self) -> pd.DatetimeIndex:
        """The T context dates, ending the day before the first forecast date."""
        end = self.future_ds[0] - pd.Timedelta(days=1)
        return pd.date_range(end=end, periods=self.context.shape[-1], freq="D")

    def index_of(self, unique_id: str) -> int:
        """Position of `unique_id` in `unique_ids`.

        Parameters
        ----------
        unique_id : str
            Series id.

        Returns
        -------
        int
            Index along the flattened `(B * V)` axis.

        Raises
        ------
        KeyError
            If the series is not in the inputs.
        """
        hits = np.flatnonzero(self.unique_ids == unique_id)
        if not len(hits):
            raise KeyError(unique_id)
        return int(hits[0])


def trim_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows of each series before its first non-zero sale.

    Series that never sold are removed entirely: before launch the product did not exist,
    so those zeros are not demand.

    Parameters
    ----------
    df : pd.DataFrame
        Long frame with `unique_id, ds, y`.

    Returns
    -------
    pd.DataFrame
        Trimmed copy, same columns.
    """
    started = (df["y"] > 0).groupby(df["unique_id"], observed=True).cummax()
    return df[started].reset_index(drop=True)


def backtest_cutoffs(last_ds: pd.Timestamp, horizon: int, n_windows: int) -> list[pd.Timestamp]:
    """Return non-overlapping rolling origins, oldest first, ending `horizon` days before the end.

    Parameters
    ----------
    last_ds : pd.Timestamp
        Last observed date.
    horizon : int
        Days forecast from each origin.
    n_windows : int
        Number of origins.

    Returns
    -------
    list[pd.Timestamp]
        Cutoff dates: training data is `ds <= cutoff`.
    """
    return [last_ds - pd.Timedelta(days=horizon * k) for k in range(n_windows, 0, -1)]


def split_at(
    df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long frame into training (`ds <= cutoff`) and the next `horizon` days.

    Parameters
    ----------
    df : pd.DataFrame
        Long frame with a `ds` column.
    cutoff : pd.Timestamp
        Last training date.
    horizon : int
        Days in the evaluation window.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and evaluation frames.
    """
    end = cutoff + pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff]
    test = df[(df["ds"] > cutoff) & (df["ds"] <= end)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def store_of(s_df: pd.DataFrame) -> pd.Series:
    """Store id of each series.

    Parameters
    ----------
    s_df : pd.DataFrame
        Static attributes with `unique_id, store_id`.

    Returns
    -------
    pd.Series
        `store_id` indexed by `unique_id`.
    """
    return s_df.set_index("unique_id")["store_id"]


def store_totals(y_df: pd.DataFrame, s_df: pd.DataFrame) -> pd.DataFrame:
    """Daily department sales of each series' store, as a past-only covariate frame.

    The total is observed only up to the forecast origin, so Timestral receives it as a
    past covariate: it carries store traffic and closures without leaking the future.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    s_df : pd.DataFrame
        Static attributes with `unique_id, store_id`.

    Returns
    -------
    pd.DataFrame
        `unique_id, ds, store_total`, aligned with `y_df`.
    """
    store = y_df["unique_id"].map(store_of(s_df))
    total = y_df.groupby([store, y_df["ds"]], observed=True)["y"].transform("sum")
    return pd.DataFrame({"unique_id": y_df["unique_id"], "ds": y_df["ds"], "store_total": total})


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Encode day of week and day of year as sine/cosine pairs.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Daily dates over context and horizon.

    Returns
    -------
    np.ndarray
        `(4, len(dates))` float32 rows `dow_sin, dow_cos, doy_sin, doy_cos`.
    """
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    return np.stack(
        [
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]
    ).astype(np.float32)


def _context_matrix(y_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Wide `(series, dates)` history up to `cutoff`, NaN before each series' first sale."""
    wide = y_df[y_df["ds"] <= cutoff].pivot(index="unique_id", columns="ds", values="y")
    full_dates = pd.date_range(wide.columns.min(), cutoff, freq="D")
    wide = wide.reindex(columns=full_dates)
    launched = (wide.fillna(0) > 0).cummax(axis=1)
    return wide.where(launched).dropna(how="all")


def _covariate_tensor(
    x_df: pd.DataFrame, unique_ids: pd.Index, dates: pd.DatetimeIndex, columns: tuple[str, ...]
) -> np.ndarray:
    """Stack per-series covariates `(B, len(columns), len(dates))`; NaN where unknown."""
    window = x_df[x_df["ds"].between(dates[0], dates[-1])]
    rows = [
        window.pivot(index="unique_id", columns="ds", values=col).reindex(
            index=unique_ids, columns=dates
        )
        for col in columns
    ]
    return np.stack([r.to_numpy(dtype=np.float32) for r in rows], axis=1)


def build_timestral_arrays(
    y_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    x_df: pd.DataFrame | None = None,
    covariates: tuple[str, ...] = KNOWN_COVARIATES,
    past_df: pd.DataFrame | None = None,
) -> TimestralInputs:
    """Convert long data into Timestral arrays for one backtest origin.

    Univariate setup: each series is one batch element with a single target variate.
    Series with no sale up to `cutoff` are left out (nothing to condition on).

    Parameters
    ----------
    y_df : pd.DataFrame
        Long target `unique_id, ds, y`.
    cutoff : pd.Timestamp
        Last context date.
    horizon : int
        Forecast horizon H.
    x_df : pd.DataFrame | None
        Long covariates; when given, `covariates` plus calendar encodings become
        known-future covariates over `T + H`.
    covariates : tuple[str, ...]
        Columns of `x_df` to use.
    past_df : pd.DataFrame | None
        Long frame `unique_id, ds, <columns>` whose columns become past-only covariates
        over the T context days (e.g. the output of `store_totals`).

    Returns
    -------
    TimestralInputs
        Context `(B, 1, T)`, optional known covariates `(B, C, T + H)` and optional
        past covariates `(B, P, T)`.

    Raises
    ------
    ShapeMismatchError
        If the known covariates do not span exactly context plus horizon.
    """
    wide = _context_matrix(y_df, cutoff)
    ctx_dates = pd.DatetimeIndex(wide.columns)
    future_ds = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    context = wide.to_numpy(dtype=np.float32)[:, None, :]
    known, names = None, ()
    if x_df is not None:
        all_dates = ctx_dates.append(future_ds)
        per_series = _covariate_tensor(
            x_df, unique_ids=wide.index, dates=all_dates, columns=covariates
        )
        cal = np.broadcast_to(
            calendar_features(all_dates), (len(wide), len(CALENDAR_COVARIATES), len(all_dates))
        )
        known = np.concatenate([per_series, cal], axis=1)
        names = (*covariates, *CALENDAR_COVARIATES)
        if known.shape[-1] != context.shape[-1] + horizon:
            raise ShapeMismatchError("known_covariates", known.shape, expected="(B, C, T + H)")
    past = None
    if past_df is not None:
        past_names = tuple(c for c in past_df.columns if c not in ("unique_id", "ds"))
        past = _covariate_tensor(
            past_df, unique_ids=wide.index, dates=ctx_dates, columns=past_names
        )
    return TimestralInputs(
        unique_ids=wide.index.to_numpy(),
        context=context,
        known_covariates=known,
        covariate_names=names,
        future_ds=future_ds,
        past_covariates=past,
    )


def mask_constant_covariate(
    inputs: TimestralInputs, name: str, window: int
) -> tuple[TimestralInputs, int]:
    """Blank a covariate row (NaN = missing) where it never moved in the last `window` steps.

    Timestral normalises each covariate row by its own variability over the context. A price
    that is flat for years and then changes in the horizon (a first discount) becomes an
    enormous normalised jump, and the upper quantiles explode (mistralai/timestral#7).
    Marking the row as missing for those series lets the model ignore it.

    Parameters
    ----------
    inputs : TimestralInputs
        Arrays with known covariates.
    name : str
        Covariate to check, e.g. "sell_price".
    window : int
        Context steps the model actually reads (its `max_context`).

    Returns
    -------
    tuple[TimestralInputs, int]
        Updated inputs and the number of series whose row was blanked.
    """
    if inputs.known_covariates is None or name not in inputs.covariate_names:
        return inputs, 0
    row = inputs.covariate_names.index(name)
    t = inputs.context.shape[-1]
    known = inputs.known_covariates.copy()
    ctx = known[:, row, max(0, t - window) : t]
    observed = ~np.isnan(ctx)
    lo = np.where(observed, ctx, np.inf).min(axis=1)
    hi = np.where(observed, ctx, -np.inf).max(axis=1)
    flat = ~(hi > lo)  # also True when the row is never observed in the window
    known[flat, row, :] = np.nan
    return replace(inputs, known_covariates=known), int(flat.sum())
