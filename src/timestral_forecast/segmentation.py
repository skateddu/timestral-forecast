"""Syntetos-Boylan demand classification (ADI / CV²)."""

import numpy as np
import pandas as pd

from timestral_forecast.preprocessing import trim_leading_zeros

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def adi_cv2_class(y_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each series as smooth, erratic, intermittent or lumpy.

    Computed from the first sale onwards, so leading zeros do not inflate ADI.

    Parameters
    ----------
    y_df : pd.DataFrame
        Long history `unique_id, ds, y` (use only data available at the origin).

    Returns
    -------
    pd.DataFrame
        Indexed by `unique_id` with columns `adi`, `cv2`, `segment`, `volume`.
    """
    trimmed = trim_leading_zeros(y_df)
    nonzero = trimmed[trimmed["y"] > 0]
    sizes = nonzero.groupby("unique_id", observed=True)["y"].agg(["count", "mean", "std"])
    periods = trimmed.groupby("unique_id", observed=True).size()
    out = pd.DataFrame(index=sizes.index)
    out["adi"] = periods.reindex(out.index) / sizes["count"]
    # population std, consistent with the usual definition of CV²
    std = sizes["std"].fillna(0) * np.sqrt((sizes["count"] - 1) / sizes["count"])
    out["cv2"] = (std / sizes["mean"]) ** 2
    out["segment"] = np.select(
        condlist=[
            (out["adi"] < ADI_THRESHOLD) & (out["cv2"] < CV2_THRESHOLD),
            out["adi"] < ADI_THRESHOLD,
            out["cv2"] < CV2_THRESHOLD,
        ],
        choicelist=["smooth", "erratic", "intermittent"],
        default="lumpy",
    )
    out["volume"] = trimmed.groupby("unique_id", observed=True)["y"].sum()
    return out
