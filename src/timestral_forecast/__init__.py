"""Benchmark of the Timestral foundation model against classical demand forecasting methods."""

from importlib.metadata import version

from timestral_forecast.config import Settings

__version__ = version("timestral-forecast")

__all__ = ["Settings", "__version__"]
