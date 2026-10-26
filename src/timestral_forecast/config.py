"""Experiment settings, overridable through `TF_*` environment variables or a `.env` file."""

from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration of the Timestral vs benchmarks experiment.

    Attributes
    ----------
    data_dir : Path
        Root data folder; raw downloads go in `raw/`, cached results in `processed/`.
    m5_url : str
        Public mirror of the M5 competition files (the one used by `datasetsforecast`).
    dept_id : str
        M5 department kept in the subset.
    store_ids : tuple[str, ...]
        M5 stores kept in the subset.
    horizon : int
        Forecast horizon in days (28, as in the M5 competition).
    n_windows : int
        Number of rolling backtest origins, each `horizon` days apart, ending at the last
        observed date.
    holiday_cutoffs : tuple[date, ...]
        Extra origins whose 28-day windows contain US holidays (Independence Day,
        Thanksgiving, Christmas and New Year), used to stress-test the best models.
    holiday_jobs : tuple[str, ...]
        Benchmark jobs (cache names) also run on `holiday_cutoffs`.
    season_length : int
        Weekly seasonality of daily retail data.
    moving_average_window : int
        Window of the moving-average baseline.
    arima_train_days : int
        AutoARIMA, AutoETS and AutoCES are fit on the most recent days only, to keep
        runtime reasonable.
    lgbm_tweedie_power : float
        Tweedie variance power of LightGBM (values close to 1 worked best in M5).
    timestral_models : tuple[str, ...]
        Hugging Face ids of the Timestral checkpoints to benchmark (history only). The
        209M is left out: with covariates its upper quantiles blew up even with the
        constant-price safeguard (mistralai/timestral#7).
    timestral_covariate_models : tuple[str, ...]
        Checkpoints also run with known covariates.
    timestral_batch_size : int
        Batch elements per forward pass; lower it if memory runs out.
    timestral_max_context : int
        Most recent days fed to Timestral (covariate cost grows with context length).
    timestral_symmetric_averaging_with_covariates : bool
        Timestral's mirrored pass (on by default) also negates the covariates. Turning it
        off reduced the upper-quantile blow-ups with a price covariate, but masking
        constant prices fixes the cause, so the default stays on.
    timestral_mask_constant_price : bool
        Blank the price covariate of series whose price never moved in the context:
        a first price change in the horizon otherwise blows up the upper quantiles
        (mistralai/timestral#7).
    timestral_group_size : int
        Items of the same store forecast jointly as variates in the multivariate run.
    device : str
        "cpu", "cuda", or "auto" to pick CUDA when available. All reported results were
        produced on CPU.
    n_jobs : int
        Parallel workers for statsforecast and LightGBM. Requires OPENBLAS_NUM_THREADS=1,
        otherwise every worker spawns its own BLAS thread pool and memory runs out.
    """

    model_config = SettingsConfigDict(env_prefix="TF_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    m5_url: str = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
    dept_id: str = "FOODS_3"
    store_ids: tuple[str, ...] = ("CA_1", "TX_1", "WI_1")
    horizon: int = 28
    n_windows: int = 3
    holiday_cutoffs: tuple[date, ...] = (date(2015, 6, 21), date(2015, 11, 22), date(2015, 12, 20))
    holiday_jobs: tuple[str, ...] = (
        "baselines",
        "arima",
        "ets",
        "LightGBM-norm",
        "Timestral-28M",
        "Timestral-28M+cov",
        "Timestral-28M+cov+past",
        "Timestral-28M-mv",
    )
    season_length: int = 7
    moving_average_window: int = 28
    arima_train_days: int = 730
    lgbm_tweedie_power: float = 1.1
    timestral_models: tuple[str, ...] = ("mistralai/Timestral-28M",)
    timestral_covariate_models: tuple[str, ...] = ("mistralai/Timestral-28M",)
    timestral_batch_size: int = 64
    timestral_max_context: int = 1024
    timestral_symmetric_averaging_with_covariates: bool = True
    timestral_mask_constant_price: bool = True
    timestral_group_size: int = 4
    device: str = "cpu"
    n_jobs: int = 16

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
