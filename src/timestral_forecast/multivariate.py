"""Group related series of the same store into joint variates for multivariate Timestral."""

import warnings
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from timestral_forecast.exceptions import ShapeMismatchError
from timestral_forecast.preprocessing import TimestralInputs

AFFINITY_WEEKS = 52


def weekly_profiles(context: np.ndarray, weeks: int = AFFINITY_WEEKS) -> np.ndarray:
    """Sum the last `weeks` full weeks of each context row (NaN counts as no sale).

    Parameters
    ----------
    context : np.ndarray
        `(B, 1, T)` univariate context.
    weeks : int
        Weeks kept; fewer if the context is shorter.

    Returns
    -------
    np.ndarray
        `(B, weeks)` weekly sales, oldest week first.
    """
    days = min(weeks, context.shape[-1] // 7) * 7
    recent = np.nan_to_num(context[:, 0, context.shape[-1] - days :])
    return recent.reshape(len(recent), -1, 7).sum(axis=-1)


def affinity_order(profiles: np.ndarray) -> np.ndarray:
    """Order series so that those with correlated weekly sales sit next to each other.

    Average-linkage clustering on `1 - correlation`; the dendrogram leaf order places
    the most similar series side by side, so consecutive chunks form affine groups.

    Parameters
    ----------
    profiles : np.ndarray
        `(n, weeks)` weekly sales of the series to order.

    Returns
    -------
    np.ndarray
        Permutation of `range(n)`.
    """
    if len(profiles) < 3:
        return np.arange(len(profiles))
    # constant profiles have undefined correlation: treat them as uncorrelated
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dist = np.nan_to_num(pdist(profiles, metric="correlation"), nan=1.0)
    return leaves_list(linkage(dist, method="average"))


def affinity_groups(inputs: TimestralInputs, store_of: pd.Series, size: int) -> list[np.ndarray]:
    """Partition the batch into groups of up to `size` affine series of the same store.

    Parameters
    ----------
    inputs : TimestralInputs
        Univariate inputs `(B, 1, T)`.
    store_of : pd.Series
        Store id indexed by `unique_id`.
    size : int
        Variates per group; the last group of each store may be smaller.

    Returns
    -------
    list[np.ndarray]
        Batch indices of each group.
    """
    stores = store_of.reindex(inputs.unique_ids).to_numpy()
    profiles = weekly_profiles(inputs.context)
    groups = []
    for store in pd.unique(stores):
        members = np.flatnonzero(stores == store)
        ordered = members[affinity_order(profiles[members])]
        groups += [ordered[i : i + size] for i in range(0, len(ordered), size)]
    return groups


def stack_variates(inputs: TimestralInputs, groups: Sequence[np.ndarray]) -> TimestralInputs:
    """Turn equally sized groups of univariate series into multivariate batch elements.

    Covariates are taken from the first member of each group, so they must be shared
    by the whole group (store-level rows such as SNAP days, events, calendar, store
    totals); item-level rows such as price do not fit a multivariate element.

    Parameters
    ----------
    inputs : TimestralInputs
        Univariate inputs `(B, 1, T)`.
    groups : Sequence[np.ndarray]
        Batch indices of each group, all of the same length V.

    Returns
    -------
    TimestralInputs
        Context `(G, V, T)` with `unique_ids` flattened in `(G, V)` order.

    Raises
    ------
    ShapeMismatchError
        If the groups do not all have the same size.
    """
    sizes = tuple(len(g) for g in groups)
    if len(set(sizes)) != 1:
        raise ShapeMismatchError("groups", sizes, expected="(G, V) equally sized groups")
    index = np.stack(groups)
    shared = {
        name: None if (arr := getattr(inputs, name)) is None else arr[index[:, 0]]
        for name in ("known_covariates", "past_covariates")
    }
    return replace(
        inputs,
        unique_ids=inputs.unique_ids[index.ravel()],
        context=inputs.context[index, 0, :],
        **shared,
    )


def group_by_affinity(
    inputs: TimestralInputs, store_of: pd.Series, size: int
) -> list[TimestralInputs]:
    """Split univariate inputs into multivariate inputs, one per distinct group size.

    Parameters
    ----------
    inputs : TimestralInputs
        Univariate inputs `(B, 1, T)` with store-level covariates only.
    store_of : pd.Series
        Store id indexed by `unique_id`.
    size : int
        Target number of variates per group.

    Returns
    -------
    list[TimestralInputs]
        Multivariate inputs; together they cover every series exactly once.
    """
    groups = affinity_groups(inputs, store_of=store_of, size=size)
    by_size: dict[int, list[np.ndarray]] = {}
    for group in groups:
        by_size.setdefault(len(group), []).append(group)
    return [stack_variates(inputs, by_size[v]) for v in sorted(by_size, reverse=True)]
