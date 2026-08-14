import numpy as np
from .utils import _fast_PD_inverse, _norm_diag
from .types import SimsMat 

def forest(L:SimsMat, beta:float=1):
    r"""relative forest accessibilities
    
    $$
    Q = (I+ \beta L)^{-1}
    $$
    
    Gauranteed to be PD, stochastic, and strictly positive for connected graphs.
    For disconnected graphs, a q_ij=0 means no paths exist between nodes i,j.

    For each entry of Q, 
    
    $$
    q_{ij}=\frac{\varepsilon(\mathcal{F}^{ij})}{\varepsilon(\mathcal{F})}
    $$ 

    which is the fraction of spanning forests on the graph of L where i,j are 
    in the same spanning tree, rooted at i. 

    Also called the Regularized Inverse Laplacian"""

    # return np.linalg.inv(np.eye(L))

    return _fast_PD_inverse(np.eye(L.shape[0]) + beta * L)

def forest_correlation(L:SimsMat, beta=1.):
    """ Re-scaling of forest matrix to have unit diagonal entries.
    
    If Q=q_ij gives the fraction of spanning forests where i,j share a 
    spanning tree (rooted at i), then normalizing by the relative forest 
    accessibilities q_ii is analagous to 
    - re-scaling a covariance matrix into correlations, or 
    - rescaling a grammian into a cosine similarity. 
    """
    return _norm_diag(forest(L,beta=beta))

def sinkhorn(A:SimsMat, i:int=0, err:float=1e-6, it_max:int=1000):
    """ Make matrix A doubly-stochastic (rows/cols sum to 1), if possible. 
    
    Uses sinkhorn-knop (iterated proportional fitting) to project an n x n matrix A 
    onto the closest point in the Birkhoff polytope.  
    
    """
    marg_sum = A.sum(0)
    if np.std(marg_sum) < err:
        return np.maximum(A, A.T)
    elif i > it_max:
        import warnings

        warnings.warn("sinkhorn iterations did not converge...", RuntimeWarning)
        return A
    else:
        return sinkhorn((A / marg_sum).T, i=i + 1)

def bilinear_kern(D:SimsMat)->SimsMat:
    """
    Inverse of the bilinear distance operation to recover a kernel. 

    Args:
      D: matrix of distances on nodes
    """
    J = np.ones_like(D)/D.shape[0]
    H = np.eye(D.shape[0]) - J
    
    return -H@(D@H) + J
    
    
    
