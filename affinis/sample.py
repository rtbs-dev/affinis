import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence, Iterator
from itertools import count, cycle, islice, takewhile
from scipy.spatial.distance import squareform
from .utils import (
    _std_incidence_vec,
    _e_to_ij,
    _ij_to_e,
    _rank1_downdate,
    _rank1_update,
)
from .proximity import forest


def _random_jump(M_cdf: ArrayLike, start: int, rng=np.random.default_rng()) -> int:
    thres = rng.random()
    return np.searchsorted(M_cdf[start], thres)


def _random_walk(M_cdf: ArrayLike, start: int, rng=np.random.default_rng()):
    # M_cdf = np.cumsum(M, axis=1)

    while True:
        start = _random_jump(M_cdf, start, rng=rng)
        yield start


def _loop_erasure(walk: Sequence[int]):
    def _get_loops(jumps: Sequence[tuple[int]]):
        # yield from
        ...

    ...


def _lerw(
    walk_generator: Iterator[int], history: Sequence, start: int
) -> Iterator[int]:
    return takewhile(lambda x: x not in history, walk_generator)


def _random_cut(cutter_gen, B, Q):
    """modifies Q in-place!"""
    # edge_to_cut = rng.choice(B.shape[0], p=cut_pmf)
    edge_to_cut = next(cutter_gen)
    # e_incidence = _std_incidence_vec(n, _e_to_ij(n, edge_to_cut))
    e_incidence = B[edge_to_cut]
    _rank1_downdate(Q, e_incidence)
    Q = Q * (~np.isclose(Q, 0))
    return edge_to_cut


def _random_bridge(E_pmf, Q, rng=np.random.default_rng()):
    """modifies Q in-place!"""
    n = Q.shape[0]
    bridge_p = E_pmf * squareform(np.isclose(Q, 0), checks=False)
    new_edge = rng.choice(bridge_p.shape[0], p=bridge_p / bridge_p.sum())
    e_incidence = _std_incidence_vec(n, _e_to_ij(n, new_edge))
    _rank1_update(Q, e_incidence)
    return new_edge


def _random_tree_jump(cutter_gen, add_pmf, B_tree, Q_tree, rng=np.random.default_rng()):
    """Modifies B,Q in-place!"""
    n = B_tree.shape[1]
    e_cut = _random_cut(cutter_gen, B_tree, Q_tree)
    e_add = _random_bridge(add_pmf, Q_tree, rng=rng)
    B_tree[e_cut] = _std_incidence_vec(n, _e_to_ij(n, e_add))

    return B_tree


def _random_tree_gibbs(add_pmf, B_tree, Q_tree=None, rng=np.random.default_rng()):
    B = np.copy(B_tree)
    if Q_tree is None:
        Q_tree = forest(B_tree.T @ B_tree)
    Q = np.copy(Q_tree)
    cutter = cycle(islice(count(), B.shape[0]))
    while True:
        B = _random_tree_jump(cutter, add_pmf, B, Q_tree=Q, rng=rng)
        yield B
