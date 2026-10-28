import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timestral_forecast.data import download_m5, load_m5_subset, load_or_build
from timestral_forecast.exceptions import DataNotFoundError

N_DAYS = 14


def _write_fake_m5(path: Path) -> Path:
    """Tiny archive with the schema of the Nixtla M5 mirror (no `d` column in calendar)."""
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + np.arange(N_DAYS) // 7,
            "event_name_1": [None] * (N_DAYS - 1) + ["Christmas"],
            "event_type_1": [None] * (N_DAYS - 1) + ["National"],
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    sales = pd.DataFrame(
        {
            "item_id": ["FOODS_3_001", "FOODS_3_002", "HOBBIES_1_001"],
            "dept_id": ["FOODS_3", "FOODS_3", "HOBBIES_1"],
            "cat_id": ["FOODS", "FOODS", "HOBBIES"],
            "store_id": "CA_1",
            "state_id": "CA",
            **{f"d_{i}": [1, 0, 2] for i in range(1, N_DAYS + 1)},
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": "CA_1",
            "item_id": ["FOODS_3_001", "FOODS_3_001"],
            "wm_yr_wk": [11101, 11102],
            "sell_price": 2.5,
        }
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in {
            "calendar.csv": calendar,
            "sales_train_evaluation.csv": sales,
            "sell_prices.csv": prices,
        }.items():
            archive.writestr(name, frame.to_csv(index=False))
    return path


def test_load_m5_subset_filters_and_builds_long_frames(tmp_path: Path) -> None:
    data = load_m5_subset(
        _write_fake_m5(tmp_path / "m5.zip"), dept_id="FOODS_3", store_ids=("CA_1",)
    )

    assert set(data.s_df["unique_id"]) == {"FOODS_3_001_CA_1", "FOODS_3_002_CA_1"}
    assert len(data.y_df) == 2 * N_DAYS
    assert data.x_df["is_event"].sum() == 2  # last day, both series
    assert data.x_df["event_name"].value_counts().to_dict() == {
        "none": 2 * (N_DAYS - 1),
        "Christmas": 2,
    }
    assert data.x_df.loc[data.x_df["unique_id"] == "FOODS_3_002_CA_1", "sell_price"].isna().all()
    assert data.x_df.loc[data.x_df["ds"] == "2011-01-29", "snap"].eq(0).all()  # day 29 > 10


def test_load_m5_subset_unknown_store_raises(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError):
        load_m5_subset(_write_fake_m5(tmp_path / "m5.zip"), dept_id="FOODS_3", store_ids=("XX_9",))


def test_load_m5_subset_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError, match="not found"):
        load_m5_subset(tmp_path / "missing.zip", dept_id="FOODS_3", store_ids=("CA_1",))


def test_load_or_build_second_call_reads_cache(tmp_path: Path) -> None:
    zip_path = _write_fake_m5(tmp_path / "m5.zip")
    first = load_or_build(
        zip_path, cache_dir=tmp_path / "cache", dept_id="FOODS_3", store_ids=("CA_1",)
    )
    zip_path.unlink()

    second = load_or_build(
        zip_path, cache_dir=tmp_path / "cache", dept_id="FOODS_3", store_ids=("CA_1",)
    )

    pd.testing.assert_frame_equal(first.y_df, second.y_df)


def test_download_m5_existing_file_is_not_downloaded(tmp_path: Path) -> None:
    (tmp_path / "m5.zip").write_bytes(b"x")

    assert download_m5("https://invalid.example/m5.zip", raw_dir=tmp_path) == tmp_path / "m5.zip"


def test_load_or_build_rebuilds_cache_without_event_name(tmp_path: Path) -> None:
    zip_path = _write_fake_m5(tmp_path / "m5.zip")
    cache_dir = tmp_path / "cache"
    first = load_or_build(zip_path, cache_dir=cache_dir, dept_id="FOODS_3", store_ids=("CA_1",))
    x_path = next(cache_dir.glob("*_x.parquet"))
    first.x_df.drop(columns="event_name").to_parquet(x_path, index=False)

    second = load_or_build(zip_path, cache_dir=cache_dir, dept_id="FOODS_3", store_ids=("CA_1",))

    assert "event_name" in second.x_df.columns
    assert "event_name" in pd.read_parquet(x_path).columns
