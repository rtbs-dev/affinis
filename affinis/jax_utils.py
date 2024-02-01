import jax.numpy as np
import numpy as onp
import networkx as nx


def squareform(edgelist):
    """edgelist to adj. matrix"""
    e = edgelist.shape[0]
    n = ((1 + np.sqrt(1 + 8*e))/2.).astype('int32')
    empty = np.zeros((n,n))
#     half = index_add(empty, index[np.triu_indices(n,1)], edgelist)
    half = empty.at[np.triu_indices(n,1)].add(edgelist)
    full = half+half.T
    return full


def edgelist(squareform):
    n = squareform.shape[0]
    idx = index[np.triu_indices(n,1)]
    return squareform[idx]


def cosine_sim(X):
    norm = X/np.sqrt((X**2).sum(axis=0))
#     cos = X.T.dot(X)/(onp.sqrt((X**2).sum()))
    cos = norm.T.dot(norm)
    return cos - np.diag(onp.diag(cos))  # valid adjacency
