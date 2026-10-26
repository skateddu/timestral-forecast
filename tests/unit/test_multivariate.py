import numpy as np
import pandas as pd
import pytest

from timestral_forecast.exceptions import ShapeMismatchError
from timestral_forecast.multivariate import (
    affinity_groups,
    affinity_order,
    group_by_affinity,
    stack_variates,
    weekly_profiles,
)
from timestral_forecast.preprocessing import TimestralInputs


def make_inputs(context: np.ndarray, ids: list[str]) -> TimestralInputs:
    """Univariate inputs with one store-level known covariate row equal to the batch index."""
    b, _, t = context.shape
    return TimestralInputs(
        unique_ids=np.array(ids),
        context=context,
        known_covariates=np.arange(b, dtype=np.float32)[:, None, None] * np.ones((1, 1, t + 2)),
        covariate_names=("snap",),
        future_ds=pd.date_range("2021-01-01", periods=2),
    )


def test_weekly_profiles_sums_full_weeks_and_ignores_nan() -> None:
    context = np.ones((1, 1, 16), dtype=np.float32)
    context[0, 0, -1] = np.nan

    profiles = weekly_profiles(context, weeks=52)

    assert profiles.tolist() == [[7.0, 6.0]]


def test_affinity_order_places_correlated_series_together() -> None:
    up, down = np.arange(10.0), np.arange(10.0)[::-1]
    profiles = np.stack([up, down, up * 2 + 1, down + 3])

    order = affinity_order(profiles).tolist()

    pairs = {frozenset(order[:2]), frozenset(order[2:])}
    assert pairs == {frozenset({0, 2}), frozenset({1, 3})}


def test_affinity_order_handles_constant_profiles() -> None:
    profiles = np.stack([np.zeros(5), np.ones(5), np.arange(5.0)])

    assert sorted(affinity_order(profiles).tolist()) == [0, 1, 2]


def test_affinity_groups_never_mix_stores() -> None:
    inputs = make_inputs(np.random.default_rng(0).random((5, 1, 21)), ids=list("abcde"))
    store_of = pd.Series(["A", "A", "A", "B", "B"], index=list("abcde"))

    groups = affinity_groups(inputs, store_of=store_of, size=2)

    assert all(store_of.iloc[g].nunique() == 1 for g in groups)
    assert sorted(np.concatenate(groups).tolist()) == [0, 1, 2, 3, 4]


def test_stack_variates_builds_group_axis_and_shares_lead_covariates() -> None:
    inputs = make_inputs(np.arange(4 * 3, dtype=np.float32).reshape(4, 1, 3), ids=list("abcd"))

    out = stack_variates(inputs, [np.array([2, 0]), np.array([1, 3])])

    assert out.context.shape == (2, 2, 3)
    assert out.unique_ids.tolist() == ["c", "a", "b", "d"]
    assert out.known_covariates[:, 0, 0].tolist() == [2.0, 1.0]


def test_stack_variates_unequal_groups_raise() -> None:
    inputs = make_inputs(np.zeros((3, 1, 3), dtype=np.float32), ids=list("abc"))

    with pytest.raises(ShapeMismatchError):
        stack_variates(inputs, [np.array([0, 1]), np.array([2])])


def test_group_by_affinity_returns_one_input_per_group_size() -> None:
    inputs = make_inputs(np.random.default_rng(1).random((5, 1, 14)), ids=list("abcde"))
    store_of = pd.Series("A", index=list("abcde"))

    parts = group_by_affinity(inputs, store_of=store_of, size=2)

    assert [p.context.shape[1] for p in parts] == [2, 1]
    assert sorted(np.concatenate([p.unique_ids for p in parts]).tolist()) == list("abcde")
