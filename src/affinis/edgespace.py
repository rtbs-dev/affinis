from typing import TypeAlias

import numpy as np
from jaxtyping import Int, Shaped
from plum import dispatch
from sparse import COO, einsum, triu
from scipy.spatial.distance import squareform

Idx: TypeAlias = Int[np.ndarray, "*elems"]


def sq_e_ij(n: int, e: Idx) -> tuple[Idx, Idx]:
    """Closed-form expression to map edge-to-node-pair indices.

    Get a row-column location in the upper-triangle of a symmetric
    matrix, given the linear index of it's unrolled vector representation.

    Parameters:
        n: dimension of square/symmetric matrix
        e: index of edge(s) from vector-representation

    Returns:
        pair of row(s),column(s) indices in corresponding matrix.
    """
    i = n - 2 - np.floor(np.sqrt(-8 * e + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    j = e + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
    return i.astype(int), j.astype(int)
    # return _map_edge_to_nodes(n)[e]


def sq_ij_e(n: int, ij: tuple[Idx, Idx]) -> Idx:
    """Closed-form expression to map edge-to-node-pair indices.

    Get an edge index from a row-column index pai  upper-triangle of a
    symmetric matrix.

    Parameters:
        n: dimension of square/symmetric matrix
        ij: pair of row(s),column(s) indices in corresponding matrix.

    Returns:
        index of edge(s) from vector-representation
    """
    i, j = ij
    e = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
    return e.astype(int)
    # return _map_edge_to_nodes(n).inverse[ij]


@dispatch(precedence=1)
def sq2flat(A: Shaped[np.ndarray, "n n"]) -> Shaped[np.ndarray, "e"]:
    """could just use `squareform`..."""
    n = min(A.shape[-1], A.shape[-2])
    return A[np.triu_indices(n, k=1)]


@dispatch
def sq2flat(A: Shaped[COO, "*batch n n"]) -> Shaped[COO, "*batch e"]:
    """New sparse+batched implementation"""
    n = min(A.shape[-1], A.shape[-2])
    a = triu(A, k=1)  # wow this works with ndim>2 as well!
    coords = sq_ij_e(n, a.coords[-2:, :])  # which means I can too!
    shape = int(n * n / 2 - n / 2)

    if a.ndim > 2:  # maybe there's a slicing/indexing way to make implicit
        coords = np.vstack([a.coords[0], coords])
        shape = (a.shape[0], shape)
    return COO(shape=shape, coords=coords, data=a.data)


@dispatch(precedence=1)
def flat2sq(e: Shaped[np.ndarray, "e"]) -> Shaped[np.ndarray, "n n"]:
    return squareform(e)


@dispatch
def flat2sq(e: Shaped[COO, "e"]) -> Shaped[COO, "n n"]:
    ## NOT CURRENTLY BATCH-DIM COMPATIBLE
    n = int(np.ceil(np.sqrt(e.shape[0] * 2)))
    # Check that e is of valid dimensions.
    if n * (n - 1) != e.shape[0] * 2:  # identical check from scipy
        raise ValueError("Incompatible vector size. It must be a triangular number.")
    tri_coords = sq_e_ij(n, e.coords[0])
    tri = COO(coords=tri_coords, shape=(n, n), data=e.data)
    return tri + tri.T


def csr_rows_idx(matrix):
    """Return column indices for data in matrix, per row (empty array if none)"""
    rows = matrix.shape[0]
    for index in range(rows):
        indptr_start = matrix.indptr[index]
        indptr_end = matrix.indptr[index + 1]
        # values = matrix.data[indptr_start:indptr_end]
        indices = matrix.indices[indptr_start:indptr_end]
        # func(indices, values)
        yield indices


def binary_feature_edge_cliques(X):
    X_edgespace = sq2flat(einsum("bi,bo->bio", X, X)).tocsr()
    for row in csr_rows_idx(X_edgespace):
        yield row
