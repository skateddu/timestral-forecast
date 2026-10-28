"""Simple-average ensembles of point forecasts from different model families."""

from collections.abc import Mapping, Sequence

import pandas as pd

from timestral_forecast.exceptions import ModelNotFoundError

ENSEMBLES: dict[str, tuple[str, ...]] = {
    "Ens(TS+cov, LGBM-norm, ARIMA)": ("Timestral-28M+cov-mean", "LightGBM-norm", "AutoARIMA"),
    "Ens(TS+cov, LGBM-norm, ETS)": ("Timestral-28M+cov-mean", "LightGBM-norm", "AutoETS"),
    "Ens(TS+cov, LGBM-norm)": ("Timestral-28M+cov-mean", "LightGBM-norm"),
}


def add_ensembles(
    df: pd.DataFrame, ensembles: Mapping[str, Sequence[str]] = ENSEMBLES
) -> pd.DataFrame:
    """Add one column per ensemble, the equal-weight mean of its member forecasts.

    Equal weights need no fitting, so they cannot overfit the backtest; in forecasting
    competitions (M4, M5, VN1) simple combinations of diverse models are hard to beat.

    Parameters
    ----------
    df : pd.DataFrame
        Merged evaluation frame with the member columns.
    ensembles : Mapping[str, Sequence[str]]
        Ensemble name to member columns.

    Returns
    -------
    pd.DataFrame
        `df` with one extra column per ensemble (a new frame).

    Raises
    ------
    ModelNotFoundError
        If a member column is missing from `df`.
    """
    columns = {}
    for name, members in ensembles.items():
        missing = [m for m in members if m not in df.columns]
        if missing:
            raise ModelNotFoundError(ensemble=name, missing=missing)
        columns[name] = df[list(members)].mean(axis=1)
    return df.assign(**columns)
