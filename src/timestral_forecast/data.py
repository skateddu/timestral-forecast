"""Download and subset the M5 dataset into Nixtla long format."""

import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from timestral_forecast.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass(frozen=True)
class M5Data:
    """M5 subset in long format.

    Attributes
    ----------
    y_df : pd.DataFrame
        Target with columns `unique_id, ds, y`.
    x_df : pd.DataFrame
        Covariates known in advance: `unique_id, ds, sell_price, snap, is_event, event_type`.
    s_df : pd.DataFrame
        Static attributes, one row per `unique_id`.
    """

    y_df: pd.DataFrame
    x_df: pd.DataFrame
    s_df: pd.DataFrame


def download_m5(url: str, raw_dir: Path) -> Path:
    """Download the M5 zip archive unless it is already on disk.

    Parameters
    ----------
    url : str
        Source URL of `m5.zip`.
    raw_dir : Path
        Destination folder.

    Returns
    -------
    Path
        Path of the local archive.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "m5.zip"
    if not dest.exists():
        logger.info("Downloading M5", extra={"url": url, "dest": str(dest)})
        urllib.request.urlretrieve(url, dest)
    return dest


def _read_csv(archive: zipfile.ZipFile, name: str, **kwargs: object) -> pd.DataFrame:
    """Read one CSV member of the zip archive."""
    with archive.open(name) as fh:
        return pd.read_csv(fh, **kwargs)


def _sales_long(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide sales table and attach calendar dates and week ids."""
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=d_cols,
        var_name="d",
        value_name="y",
    )
    long = long.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")
    long["unique_id"] = long["item_id"] + "_" + long["store_id"]
    long["ds"] = pd.to_datetime(long["date"])
    long["y"] = long["y"].astype("float32")
    return long.drop(columns=["d", "date"])


def _covariates(long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Build the known-in-advance covariates: price, SNAP day, event flag and type."""
    cal = calendar.assign(ds=pd.to_datetime(calendar["date"]))
    x = long[["unique_id", "ds", "item_id", "store_id", "state_id", "wm_yr_wk"]].merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    x = x.merge(cal[["ds", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]], on="ds", how="left")
    snap = np.select(
        condlist=[x["state_id"] == s for s in ("CA", "TX", "WI")],
        choicelist=[x["snap_CA"], x["snap_TX"], x["snap_WI"]],
        default=0,
    )
    return pd.DataFrame(
        {
            "unique_id": x["unique_id"],
            "ds": x["ds"],
            "sell_price": x["sell_price"].astype("float32"),
            "snap": snap.astype("float32"),
            "is_event": x["event_type_1"].notna().astype("float32"),
            "event_type": x["event_type_1"].fillna("none").astype("category"),
        }
    )


def load_m5_subset(zip_path: Path, dept_id: str, store_ids: tuple[str, ...]) -> M5Data:
    """Read the M5 archive and keep one department in a few stores.

    Filtering happens on the wide table, before melting, so memory stays low.

    Parameters
    ----------
    zip_path : Path
        Local `m5.zip`.
    dept_id : str
        Department to keep, e.g. "FOODS_3".
    store_ids : tuple[str, ...]
        Stores to keep, e.g. ("CA_1", "TX_1").

    Returns
    -------
    M5Data
        Target, covariates and static attributes.

    Raises
    ------
    DataNotFoundError
        If the archive is missing or the filter selects no series.
    """
    if not zip_path.exists():
        raise DataNotFoundError("M5 archive not found", location=str(zip_path))
    with zipfile.ZipFile(zip_path) as archive:
        sales = _read_csv(archive, "sales_train_evaluation.csv")
        sales = sales[(sales["dept_id"] == dept_id) & sales["store_id"].isin(store_ids)]
        if sales.empty:
            raise DataNotFoundError("No series for filter", location=f"{dept_id} in {store_ids}")
        calendar = _read_csv(archive, "calendar.csv")
        # the Nixtla mirror drops the `d` key: rows are in day order, d_1 = first row
        calendar["d"] = [f"d_{i}" for i in range(1, len(calendar) + 1)]
        prices = _read_csv(archive, "sell_prices.csv")
    prices = prices[prices["store_id"].isin(store_ids) & prices["item_id"].isin(sales["item_id"])]
    long = _sales_long(sales, calendar)
    s_df = sales[ID_COLS].assign(unique_id=sales["item_id"] + "_" + sales["store_id"])
    s_df = s_df[["unique_id", *ID_COLS]].reset_index(drop=True)
    logger.info("Loaded M5 subset", extra={"n_series": len(s_df), "n_rows": len(long)})
    return M5Data(
        y_df=long[["unique_id", "ds", "y"]].reset_index(drop=True),
        x_df=_covariates(long, calendar, prices).reset_index(drop=True),
        s_df=s_df,
    )


def load_or_build(
    zip_path: Path, cache_dir: Path, dept_id: str, store_ids: tuple[str, ...]
) -> M5Data:
    """Return the subset from the parquet cache, building it on first call.

    Parameters
    ----------
    zip_path : Path
        Local `m5.zip`.
    cache_dir : Path
        Folder for the parquet files.
    dept_id : str
        Department to keep.
    store_ids : tuple[str, ...]
        Stores to keep.

    Returns
    -------
    M5Data
        Target, covariates and static attributes.
    """
    tag = f"{dept_id}_{'-'.join(store_ids)}"
    paths = {name: cache_dir / f"m5_{tag}_{name}.parquet" for name in ("y", "x", "s")}
    if all(p.exists() for p in paths.values()):
        return M5Data(*(pd.read_parquet(paths[name]) for name in ("y", "x", "s")))
    data = load_m5_subset(zip_path, dept_id=dept_id, store_ids=store_ids)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("y", "x", "s"), (data.y_df, data.x_df, data.s_df), strict=True):
        frame.to_parquet(paths[name], index=False)
    return data
