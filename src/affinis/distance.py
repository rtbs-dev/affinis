import numpy as np
from .proximity import forest
from .utils import _norm_diag

def bilinear_dists(K):
    r"""symmetric bilinear form associated with kernel K
    
    If (square symmetric kernel)  provides a quadratic form as

    $$
    q(x)=x'Kx
    $$
    
    then the associated bilinear form is

    $$
    b_q(x_i,x_j)=\frac{1}{2}(q(x_i+x_j)-q(x_i)-q(x_j))
    $$
        
    If K is a proximity, then
    
    $$
    D_{ij} = 1-b_q(x_i,x_j) = \frac{1}{2}(K_{ii}+K_{jj}) - K_{ij}
    $$
      
    defines a distance metric

    Args:
      K: kernel of similarities 

    Returns: bilinear distances induced by K

    """
    k_ii = np.diag(K)
    return np.add.outer(k_ii, k_ii) / 2.0 - K


def adjusted_forest_dists(L, beta=1.0):
    """due to Chebotarev, Shamis, & Avrachenkov

    See [Avrachenkov et al (2017)](https://doi.org/10.1080/10556788.2016.1193176)
    """
    return beta * bilinear_dists(forest(L, beta=beta))


def generalized_graph_dists(L, beta=1.0):
    """due to Chebotarev, Shamis, & Avrachenkov
    
    See [Avrachenkov et al (2017)](https://doi.org/10.1080/10556788.2016.1193176)
    """

    Q = forest(L, beta=beta)
    return -np.log(_norm_diag(Q))
