from jaxtyping import Bool, Num
from typing import TypeAlias, Literal
import numpy as np
from scipy.sparse import sparray
import sparse


Arr: TypeAlias = sparse.SparseArray | sparray | np.ndarray
"""Possible acceptable array backends"""

FeatMat: TypeAlias = Bool[Arr, "obs feat"]
"""Boolean feature matrix (rows=observations, cols=features)"""

SimsMat: TypeAlias = Num[Arr, "feat feat"]
"""Feature similarity matrix (rows,cols=features)"""

FPopts: TypeAlias = Literal['edge-prob', 'interaction', 'forest-max', 'counts']
"""Variations on what to return from running Forest Pursuit.

See [documentation][affinis.associations.forest_pursuit] for use-cases.
"""


Number: TypeAlias = float | int
"""An individual number, for type-checked parameters to functions"""

PsdCtOpts:TypeAlias = Literal["min-connect","zero-sum"]
"""derived/empirical values for the pseudocount prior"""

PsdCts: TypeAlias = tuple[Number, Number] | tuple[PsdCtOpts, Number] | PsdCtOpts | Number
"""Allowed priors for beta-binonial estimates

See [documentation][affinis.priors] for more on how these options are
handled. 
"""
