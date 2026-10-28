"""Custom exception hierarchy of the package."""


class AppError(Exception):
    """Base exception for the package."""


class DataNotFoundError(AppError):
    """Raised when an expected data file or subset is missing or empty.

    Attributes
    ----------
    location : str
        File path or filter description that produced no data.
    """

    def __init__(self, message: str, location: str) -> None:
        super().__init__(f"{message}: {location}")
        self.location = location


class ShapeMismatchError(AppError):
    """Raised when model inputs do not have the shapes the model expects.

    Attributes
    ----------
    name : str
        Name of the offending array.
    shape : tuple[int, ...]
        Actual shape of the array.
    """

    def __init__(self, name: str, shape: tuple[int, ...], expected: str) -> None:
        super().__init__(f"{name} has shape {shape}, expected {expected}")
        self.name = name
        self.shape = shape


class ModelNotFoundError(AppError):
    """Raised when an ensemble refers to model columns that were not computed.

    Attributes
    ----------
    ensemble : str
        Name of the ensemble.
    missing : list[str]
        Member columns absent from the evaluation frame.
    """

    def __init__(self, ensemble: str, missing: list[str]) -> None:
        super().__init__(f"Ensemble {ensemble!r} needs missing model columns: {missing}")
        self.ensemble = ensemble
        self.missing = missing
