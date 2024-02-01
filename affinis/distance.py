import numpy as np
from .proximity import forest


def bilinear_dists(K):
    """symmetric bilinear form associated with kernel K

    If K provides a quadratic form as q(x)=x'Kx, then the associated 
    bilinear form b_q(x_i,x_j)=(q(x_i+x_j)-q(x_i)-q(x_j))/2. 

    If K is a proximity, then 1-b_q(x_i,x_j) = (K_ii+K_jj)/2 - K_ij defines a 
    distance metric D_ij
    """
    k_ii = np.diag(K)
    return np.add.outer(k_ii, k_ii) / 2.0 - K


def adjusted_forest_dists(L, beta=1.0):
    """due to Chebotarev and Avrachenkov"""
    return beta*bilinear_dists(forest(L, beta=beta))


def generalized_graph_dists(L, beta=1.0):
    """due to Chebotarev and Avrachenkov"""

    Q = forest(L, beta=beta)
    norm = np.sqrt(np.multiply.outer((q_ii := np.diag(Q)), q_ii))
    return -np.log(Q / norm)
