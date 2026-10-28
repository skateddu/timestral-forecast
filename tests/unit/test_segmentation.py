from collections.abc import Callable

import pandas as pd

from timestral_forecast.segmentation import adi_cv2_class


def test_adi_cv2_class_assigns_expected_segments(make_long: Callable[..., pd.DataFrame]) -> None:
    df = make_long(
        {
            "smooth": [10, 11, 9, 10, 10, 11, 9, 10],
            "erratic": [1, 30, 2, 25, 1, 40, 3, 20],
            "intermittent": [5, 0, 0, 5, 0, 0, 5, 0],
            "lumpy": [1, 0, 0, 40, 0, 0, 2, 0],
        }
    )

    out = adi_cv2_class(df)

    assert out["segment"].to_dict() == {
        "erratic": "erratic",
        "intermittent": "intermittent",
        "lumpy": "lumpy",
        "smooth": "smooth",
    }


def test_adi_cv2_class_ignores_leading_zeros(make_long: Callable[..., pd.DataFrame]) -> None:
    df = make_long({"late": [0, 0, 0, 0, 0, 0, 5, 5, 5, 5]})

    out = adi_cv2_class(df)

    assert out.loc["late", "adi"] == 1.0
    assert out.loc["late", "volume"] == 20
