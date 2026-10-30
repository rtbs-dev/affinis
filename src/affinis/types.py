from jaxtyping import Bool, Num
from typing import TypeAlias, Literal
import numpy as np
from scipy.sparse import sparray
import sparse
from plum import conversion_method, add_conversion_method
from networkx import Graph, adjacency_matrix

Arr: TypeAlias = sparse.SparseArray | sparray | np.ndarray
"""Possible acceptable array backends"""

FeatMat: TypeAlias = Bool[sparse.COO, "obs feat"]
"""Boolean feature matrix (rows=observations, cols=features)"""

SimsMat: TypeAlias = Num[Arr, "feat feat"]
"""Feature similarity matrix (rows,cols=features)"""

FPopts: TypeAlias = Literal["edge-prob", "interaction", "forest-max", "counts"]
"""Variations on what to return from running Forest Pursuit.

See [documentation][affinis.associations.forest_pursuit] for use-cases.
"""


Number: TypeAlias = float | int
"""An individual number, for type-checked parameters to functions"""

PsdCtOpts: TypeAlias = Literal["min-connect", "zero-sum"]
"""derived/empirical values for the pseudocount prior"""

PsdCts: TypeAlias = (
    tuple[Number, Number] | tuple[PsdCtOpts, Number] | PsdCtOpts | Number
)
"""Allowed priors for beta-binonial estimates

See [documentation][affinis.priors] for more on how these options are
handled. 
"""

add_conversion_method(sparse.SparseArray, sparse.COO, sparse.COO)
add_conversion_method(np.ndarray, sparse.COO, sparse.COO.from_numpy)
add_conversion_method(sparray, sparse.COO, sparse.COO.from_scipy_sparse)

@conversion_method(type_from=Graph, type_to=sparse.COO)
def nx_to_coo(G):
    return sparse.COO.from_scipy_sparse(adjacency_matrix(G))


