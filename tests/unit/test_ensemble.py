import pandas as pd
import pytest

from timestral_forecast.ensemble import add_ensembles
from timestral_forecast.exceptions import ModelNotFoundError


def test_add_ensembles_averages_members_with_equal_weights() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0], "A": [1.0, 4.0], "B": [3.0, 0.0]})

    out = add_ensembles(df, ensembles={"E": ("A", "B")})

    assert out["E"].tolist() == [2.0, 2.0]
    assert "E" not in df.columns


def test_add_ensembles_missing_member_raises() -> None:
    with pytest.raises(ModelNotFoundError, match="C"):
        add_ensembles(pd.DataFrame({"A": [1.0]}), ensembles={"E": ("A", "C")})
