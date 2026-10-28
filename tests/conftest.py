from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_long() -> Callable[..., pd.DataFrame]:
    """Factory of long frames `unique_id, ds, y` from a dict of value lists, daily."""

    def _make(series: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
        parts = [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range(start, periods=len(values), freq="D"),
                    "y": np.asarray(values, dtype="float32"),
                }
            )
            for uid, values in series.items()
        ]
        return pd.concat(parts, ignore_index=True)

    return _make


@pytest.fixture
def make_covariates() -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Factory of covariate frames matching a long target frame, extended `extra_days` ahead."""

    def _make(y_df: pd.DataFrame, extra_days: int = 0) -> pd.DataFrame:
        last = y_df["ds"].max() + pd.Timedelta(days=extra_days)
        rows = []
        for uid in y_df["unique_id"].unique():
            dates = pd.date_range(y_df["ds"].min(), last, freq="D")
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": dates,
                        "sell_price": 2.0,
                        "snap": (dates.day <= 10).astype("float32"),
                        "is_event": (dates.day == 3).astype("float32"),
                        "event_name": pd.Categorical(
                            np.where(dates.day == 3, "Christmas", "none")
                        ),
                        "event_type": pd.Categorical(np.where(dates.day == 3, "National", "none")),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    return _make
