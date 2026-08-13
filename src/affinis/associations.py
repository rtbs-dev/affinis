
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path, reconstruct_path
from scipy.sparse import coo_array, issparse
import sparse
from .utils import (
    _sq,
    _outer,
    _sparse_directed_to_symmetric,
    # groupby_col0,
    csr_rows_idx,
    complete_edgelist_on_nodes,
    sq_ij_e,
    edge_weights_to_laplacian,
    _norm_diag,
)
from .types import FeatMat, SimsMat, FPopts
from .priors import pseudocount, PsdCts
from .distance import adjusted_forest_dists
from .proximity import sinkhorn
from .filter import min_connected_filter

__all__ = [
    "coocur_prob",
    "odds_ratio",
    "mutual_information",
    "chow_liu",
    "yule_y",
    "yule_q",
    "ochiai",
    "binary_cosine_similarity",
    "hyperbolic_project",
    "resource_project",
    "high_salience_skeleton",
    "doubly_stochastic_filter",
    "forest_pursuit",       
]

# TODO: define type signatures by array shapes
# TODO: dispatch on dataframes, static-frames, etc.

def _safe_div(num, den):
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=den != 0,
    )

def _gram(X1, X2):
    grammian = X1.T @ X2
    return grammian.toarray() if issparse(grammian) else grammian


def coocur_prob(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """probability of a co-ocurrence per-observation

    Args:
      X: feature matrix     
      pseudocts: additive smoothing parameter

    """

    cts = _gram(X, X)
    tot = X.shape[0]
    return pseudocount(pseudocts)(cts, tot)


# def cond_prob(X:FeatMat, pseudocts:PsdCts=0.):
#     cts = X.T@X + pseudocts
#     tot = X.sum(axis=0)+2
#     return cts/tot


def _not(X):
    if issparse(X):
        nX = X < coo_array(np.ones(X.shape))
    else:
        nX = 1 - X
    return nX


def _marginal_cts(X):
    return X.sum(axis=0), _not(X).sum(axis=0)


def _marginal_prob(X: FeatMat, pseudocts: PsdCts = 0.5):
    # psdct = pseudocount(pseudocts)  #TODO cannot use _sq(num) when num is 1D!!
    # return [psdct(ct, X.shape[0]) for ct in _marginal_cts(X)]
    return [(ct + pseudocts) / (X.shape[0] + 2 * pseudocts) for ct in _marginal_cts(X)]


def _contingency_cts(X):
    nX = _not(X)
    both = _sq(_gram(X, X))
    one = _sq(_gram(X, nX))
    other = _sq(_gram(nX, X))
    neither = _sq(_gram(nX, nX))
    return neither, one, other, both


def _contingency_prob(X: FeatMat, pseudocts: PsdCts = 0.5):
    psdct = pseudocount(pseudocts)
    cts = np.vstack(_contingency_cts(X))
    return [psdct(ct, X.shape[0]) for ct in cts]


def _binary_contingency(X):
    n = X.shape[0]
    both = coocur_prob(X)
    neither = coocur_prob(1 - X)
    one = (X.T @ (1 - X) + 1) / (
        n + 2
    )  # np.logical_and.outer(X, ~X).mean(axis=2).mean(axis=0)
    other = one.T
    return neither, one, other, both


def odds_ratio(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    r"""Ratio of the odds of a true pos/neg to false pos/neg
    
    For associations, we replace pos/neg and true/false with
    a=1/0 b=1/0, which gives odds ratio as:
    
    $$
    \text{OR}=\frac{p_{11}p_{00}}{p_{01}p_{10}}
    $$

    Additive smoothing is applied via the initial calculation of
    conditional probabilities.
    
    Args:
      X: feature matrix
      pseudocts: additive smoothing parameter
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(a * d / (b * c)) + np.eye(X.shape[1])


def mutual_information(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    r"""Mutual Information over binary random variables

    For use in e.g. Chow-Liu Trees


    An estimate for the mutual information (i.e., between the sample
    distributions) can be derived from the marginals, as with
    OR/Yule's Q/Y/etc., though it is more compactly represented as a
    pairwise sum over the domains of each distribution being compared:     

    $$
    \text{MI}(x_i,x_j)\approx \sum_{i,j\in[0,1]} p_{ij} \log \left( \frac{p_{ij}}{p_{i\bullet}p_{\bullet j}} \right) 
    $$

    Additive smoothing is applied via the initial calculation of
    conditional probabilities.
    
    Args:
        X: feature matrix
        pseudocts:  additive smoothing parameter

    Returns:

    """
    Pxy = np.array(_contingency_prob(X, pseudocts=pseudocts))

    # Pxy = np.vstack([sq(i) for i in cond_table])
    yes, no = _marginal_prob(X)
    # TODO had to hard-code 0.5 for now :(, pseudocts=pseudocts)

    # print(Pxy.sum(axis=1))
    PxPy = np.vstack(
        [
            _sq(np.multiply.outer(a, b))
            for a, b in [(no, no), (yes, no), (no, yes), (yes, yes)]
        ]
    )
    entropy = -np.vstack([x * np.log(x) for x in (yes, no)]).sum(axis=0)

    MI_pairs = (Pxy * np.log(Pxy) - Pxy * np.log(PxPy)).sum(axis=0)
    return _sq(MI_pairs) + np.diag(entropy)
    # return sq(entropy(Pxy, PxPy))


def chow_liu(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """Chow-Liu Tree on the features of a (binary) design matrix

    computes mutual information over all pairs of features, and returns
    the maximum spanning tree on them. Assumes a symmetric adjacency is wanted.

    Args:
      X: feature matrix
      pseudocts: additive smoothing parameter

    Returns: Adjacency matrix of the Chow Liu MST

    """
    # return sq(sq(minimum_spanning_tree(-MI_binary(X)).todense()))
    return _sq(
        _sq(
            minimum_spanning_tree(
                np.exp(-(mutual_information(X, pseudocts=pseudocts)))
            ).todense()
        )
    )


def yule_y(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    r"""a.k.a. Coefficient of Colligation.

    Mobius transform of the Odds Ratio to the range [-1,1], scaled so that
    transformed contingency table of each feature pair has unitary off-
    diagonals and diagonal (associations) $sqrt{OR}$.

    $$
    Y=\frac{\sqrt{\text{OR}}-1}{\sqrt{\text{OR}}+1}
    $$

    Args:
      X: feature matrix
      pseudocts: additive smoothing parameter

    Returns: square matrix containing Yule's Y

    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := np.sqrt(a * d)) - (bc := np.sqrt(b * c))) / (ad + bc))
    # return ((sor:=np.sqrt(OR))-1)/(sor+1)
    # return ((ad:=np.sqrt(both*neither))-(bc:=np.sqrt(one*other)))/(ad+bc)


def yule_q(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    r"""a.k.a. Goodman & Kruskal's gamma for 2x2.

    mobius transform of the Odds Ratio to the range [-1,1]:
    
    $$
    Q = \frac{\text{OR}-1}{\text{OR}+1}
    $$

    Args:
      X: feature matrix
      pseudocts:  additive smoothing parameter 

    Returns: square matrix containing Yule's Q

    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := a * d) - (bc := b * c)) / (ad + bc))


def ochiai(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    r"""a.k.a cosine similarity on binary sets

    Effectively an uncentered correlation, but for binary observations the "cosine similarity" is also called the _Ochiai Coefficient_ between two sets $A,B$, where binary "1" stands for an element belonging to the set.
    See [Janson and Vegelius (1981)](https://doi.org/10.1007/bf00347601).

    In our use case, we define measured as 

    $$
    \frac{|A \cap B |}{\sqrt{|A||B|}}=\sqrt{p_{1\bullet}p_{\bullet 1}} \rightarrow  \frac{X^TX}{\sqrt{\mathbf{s}_i\mathbf{s}_i^T}}\quad \mathbf{s}_i = \sum_i \mathbf{x}_i
    $$  

    This interpretation of cosine similarity as the geometric mean of
    conditional probabilities is particularly useful when trying to
    approximate interaction rates, and especially to apply additive
    smoothing. 

    Args:
      X: feature matrix
      pseudocts:  additive smoothing parameter 

    Returns: square cosine similarity matrix (incl. ones in the diagonal)

    """
    # I = np.eye(X.shape[-1])
    co_occurs = _sq(_gram(X, X))  # + pseudocts
    exposures = X.sum(axis=0)
    pseudo_exposure = _sq(np.sqrt(_outer(np.multiply, exposures)))  # + 2 * pseudocts
    # return _sq(co_occurs) / pseudo_exposure + np.eye(X.shape[1])
    return _sq(pseudocount(pseudocts)(co_occurs, pseudo_exposure)) + np.eye(X.shape[1])


def binary_cosine_similarity(X: FeatMat, pseudocts: PsdCts = 0.5) -> SimsMat:
    """alias of ochiai(X), provided for user convenience

    Args:
      X: feature matrix
      pseudocts:  additive smoothing parameter

    Returns: cosine similarity on binary feature vectors

    """
    return ochiai(X, pseudocts=pseudocts)



def hyperbolic_project(
    X:FeatMat, pseudocts:PsdCts=0.5
)->SimsMat:
    """
    [Newman (2001)](https://doi.org/10.1103/physreve.64.016132)
    TODO: add to tests and document

    Note: passing a pseudocount currently does not have an effect,
    as there is not a trivial way to interpret this bipartite projection
    as a probability.

    Args:
      X: feature matrix

    """
    
    reweight = _safe_div(X.T, (X.sum(axis=1)-1)).T  # right-stochastic bipartite
    return _gram(reweight,X) 


def resource_project(
    X: FeatMat, pseudocts: PsdCts = 0.5, sym_func=np.maximum
) -> SimsMat:
    """Bipartite projection due to [Zhou et al (2007)](https://doi.org/10.1103/PhysRevE.76.046115).

    Goes one step further than hyperbolic projection, by viewing each
    node as having some “amount” of a resource to spend, which gets
    re-allocated by observational unit.
    Can be interpreted as one step of iterative proportional fitting on the
    bipartite adjacency matrix. 

    NOTE: For additive smoothing to work, we assume no smoothing is needed
    for the "forward" projection (features->observations), since we assume
    all observations have some feature participation,
    while some features may have no (observed) observations.

    By default, we symmetrize with "maximum", meaning that association is
    considered as the strongest of the directions it could take. This
    can be overridden with any function of two same-shaped arrays.

    Args:
      X: feature matrix
      pseudocts: additive smoothing parameter
      sym_func:  function to symmetrize the resulting association measure

    Returns: symmetrized "resource projection" similarities

    """
    psdct_func = pseudocount(pseudocts)

    fwd = _safe_div(X.T, X.sum(axis=1)).T  # right-stochastic bipartite
    # bwd = psdct_func(X, X.sum(axis=0))

    # P = _gram(((X.T) / (X.sum(axis=1))).T, ((X) / (X.sum(axis=0))))
    num = _gram(X, fwd)  # project back
    # den = np.multiply.outer(X.sum(axis=0), np.ones(X.shape[1]))
    den = X.sum(axis=0)  # normalized by node occurrences (bipartite degree)
    P = psdct_func(num, den)
    return sym_func(P, P.T)


def high_salience_skeleton(X: FeatMat, prior_dists:SimsMat|None=None, pseudocts: PsdCts = "min-connect"):
    """Backboning technique from [Grady et al. (2012)](https://doi.org/10.1038/ncomms1847).
    Calculates shortest paths from every node, and counts the
    number of trees each edge ended up being used in.

    Args:
      X: feature matrix
      prior_dists:  (Default = `-log(ochiai)`) prior distances for shortest paths
      pseudocts: additive smoothing parameter 

    """
    if prior_dists is None:
        prior_dists = np.abs(-np.log(ochiai(X, pseudocts=pseudocts)))
    d, pred = shortest_path(prior_dists, return_predecessors=True)
    E_obs = np.array(
        [
            _sq(
                _sparse_directed_to_symmetric(
                    reconstruct_path(d, p, directed=False).astype(bool)
                ).toarray()
            )
            for p in pred
        ]
    )
    hss = pseudocount(pseudocts)(E_obs.sum(axis=0), E_obs.shape[0])
    # hss = (E_obs.sum(axis=0) + pseudocts) / (E_obs.shape[0] + 2 * pseudocts)
    return _sq(hss)


def doubly_stochastic_filter(
    X:FeatMat,
    reg:float=0.1,
    pseudocts:PsdCts=0.5,
    prior_dists:SimsMat|None=None,
    **sink_kws
)->SimsMat:
    """From [P.B. Slater(2009)](https://doi.org/10.1073/pnas.0904725106),
    
    _“A two-stage algorithm for extracting the multiscale backbone of complex weighted networks”_

    This implementation is effectively a wrappper around
    [sinkhorn][affinis.proximity.sinkhorn] and
    [min_connected_filter][affinis.filter.min_connected_filter] to
    accomplish the "two stage" process used.

    Args:
      X:feature matrix
      pseudocts:  additive smoothing parameter
      prior_sims:  (Default will calculate cosine similarity)
      sink_kws: kwargs to pass to `sinkhorn()`
    """

    if prior_dists is None:
        prior_dists= -np.log(ochiai(X, pseudocts=pseudocts))

    kern = _sq(np.exp(-_sq(prior_dists)/reg))
    ds = sinkhorn(kern, **sink_kws)

    return _sq(min_connected_filter(_sq(ds)))


def _pursue_tree_basis(dists, nodes, edge_priors=False, beta=0.001):
    N = dists.shape[0]
    all_E = complete_edgelist_on_nodes(N, nodes)
    subset_dists = dists[nodes].T[nodes].T
    if edge_priors:
        # if we are recieving an estimate for the Graph Laplacian
        # ...approximate steiner with local shortest path dists. 
        subset_dists = adjusted_forest_dists(subset_dists, beta=beta)
    tree = sparse.COO.from_scipy_sparse(minimum_spanning_tree(subset_dists))
    tree_E = sq_ij_e(tree.shape[0], tree.coords)
    return all_E[tree_E]


def _spanning_forests_obs_bootstrap(X, prior_dists=None, edge_priors=False, beta=0.001):
    """resample with a kernel bootstrap on MSTs in a manifold"""
    if (prior_dists is None) and (not edge_priors):
        prior_dists = -np.log(ochiai(X, pseudocts=0.5))

    elif edge_priors:
        # TODO: enforce laplacian!
        # assert np.allclose(prior_dists.sum(axis=0),0)
        pass
        
    # N_obs = X.toarray() if issparse(X) else X
    N_obs = sparse.COO.from_scipy_sparse(X) if issparse(X) else sparse.COO(X)
    N_activations = csr_rows_idx(N_obs.tocsr())
    E_activations = [
        _pursue_tree_basis(prior_dists, nodes, edge_priors=edge_priors, beta=beta)
        if nodes.size>0 else nodes for nodes in N_activations
    ]

    E_coords = np.vstack((
        np.repeat(np.arange(N_obs.shape[0]), np.array([len(e) for e in E_activations])),
        np.concatenate(E_activations),
    ))
    n = X.shape[1]
    m = n * (n - 1) // 2
    return coo_array(
        sparse.COO(E_coords, data=1, shape=(X.shape[0], m)).to_scipy_sparse()
    )

    # E_obs = coo_array(
    #     [
    #         _sq(
    #             minimum_spanning_tree(_get_masked_sq(_sq(prior_dists), i)).todense() > 0
    #         )
    #         for i in N_obs
    #     ]
    # )
    # return E_obs



def forest_pursuit_cts(X: FeatMat, prior_dists:SimsMat|None=None) -> SimsMat:
    """Point estimate for number of actual edge activations, rather than
    node-node co-occurrences.
    Uses the Empirical Bayes estimate of the Spanning Forest Density

    Args:
      X: feature matrix
      prior_dists:  (Default value = None)

    Returns: counts for approximate steiner tree occurrences.

    """

    # est_dists = -np.log(prior(X, pseudocts=pseudocts))
    e_obs = _spanning_forests_obs_bootstrap(X, prior_dists=prior_dists)
    return _sq(e_obs.sum(axis=0))

def expected_forest_maximization(
    X: FeatMat,
    prior_struct:SimsMat=None,
    beta:float=0.001,
    eps:float=1e-5,
    max_iter:int=100,
    verbose:bool=False,
) -> SimsMat:
    """Expectation Maximization Scheme to recover structure.

    Shown to have minor accuracy improvement over vanilla Forest Pursuit,
    at the cost of significantly decreased scalability, as it uses an
    alternating-minimization scheme to jointly estimate edge activation
    probabilities, _and_ network structure.

    For more detail, see [Expected Forest Maximization](https://dissertation.rtbs.dev/content/part2/2-06-latent-forest-alloc.html#expected-forest-maximization). 
    
    Args:
      X: FeatMat: 
      prior_struct:
      beta:  
      eps:  
      max_iter: 
      verbose:  

    Returns:

    """
    if prior_struct is None:
        e_prob = _sq(forest_pursuit_edge(X))
        prior_struct = _norm_diag(edge_weights_to_laplacian(e_prob))

    uv_cts = _sq(_gram(X,X))    
    
    diff, it = 1.,0

    while (diff>eps) and (it<max_iter):
        it+=1
        e_cts = _spanning_forests_obs_bootstrap(X, prior_dists=prior_struct, edge_priors=True, beta=beta).sum(axis=0)
        e_prob_new = e_prob + (e_cts - e_prob*uv_cts)/(uv_cts+1)  # posterior for Beta(a, 1-a)
        diff = np.max(np.abs(e_prob_new - e_prob))
        e_prob = e_prob_new
        prior_struct = _norm_diag(edge_weights_to_laplacian(e_prob))

    if verbose:
        return _sq(e_prob), it
    else:
        return _sq(e_prob)
        

def forest_pursuit_edge(
    X: FeatMat, prior_dists:SimsMat|None=None, pseudocts: PsdCts = "min-connect"
) -> SimsMat:
    """point estimate for edge-activation probability, conditional on
    both nodes being a priori activated.
    Uses the Spanning Forest Density non-parametric estimator

    Args:
      X: feature matrix
      prior_dists:  default uses `-log(cos)`
      pseudocts:   additive smoothing parameter

    Returns: probability of edge traversal given a co-occurrence. 
    """
    e_cts = _sq(forest_pursuit_cts(X, prior_dists=prior_dists))
    uv_cts = _sq(_gram(X, X))
    # e_prob = (e_cts + pseudocts) / (uv_cts + 2 * pseudocts)
    e_prob = pseudocount(pseudocts)(e_cts, uv_cts)
    return _sq(e_prob)


def forest_pursuit_interaction(
    X: FeatMat,
    prior_dists:SimsMat|None=None,
    precalc_prob: SimsMat | None = None,
    pseudocts: PsdCts = "min-connect",
) -> SimsMat:
    """point estimate for probability of observing an edge traversal,
    using the Spanning Forest Density non-parametric estimator. Weights conditional
    edge traversal probability by the base co-occurrence probability. 

    If you have already calculated forest_pursuit_edge, you can pass it as precalc_prob. 

    Args:
      X: feature matrix
      prior_dists: default uses `-log(cos)`
      precalc_prob: to avoid re-computing edge prob if you already have
      pseudocts:  additive smoothing parameter

    Returns: probability of observing an edge traversal

    """
    # TODO: check X and precalc_cts have correct dim
    if not precalc_prob:
        e_prob = _sq(
            forest_pursuit_edge(X, prior_dists=prior_dists, pseudocts=pseudocts)
        )
    else:
        e_prob = _sq(precalc_prob)
    uv_prob = _sq(coocur_prob(X, pseudocts=pseudocts))
    e_margP = e_prob * uv_prob
    return _sq(e_margP)


def forest_pursuit(
    X: FeatMat,
    mode:FPopts="edge-prob",
    pseudocts:PsdCts = "min-connect",
    prior_dists:SimsMat|None=None,
    **efm_kws    
)->SimsMat:
    """Structure estimation under spreading-process assumption [(Sexton, 2025)](https://doi.org/10.13016/o252-lfmi)

    This is the original implementation, as demonstrated in
    [Rachael Sextons's dissertation (2025)](https://dissertation.rtbs.dev/).
    For more details on how it works, see [Approximate Recovery in Near-linear Time by Forest Pursuit](https://dissertation.rtbs.dev/content/part2/2-05-forest-pursuit.html), and subsequent chapters for modifications and
    descriptions of the "modes" presented here. 

    Which mode you select determines how you will interpret the resulting
    association scores.
     
    - If "edge-prob", FP will estimate the probability of an edge
      being activated, _given_ you know the two nodes it connects
      are known to be active. This is a formalization of what it means
      for an edge to "exist" in the Desire-Path Density framing.
      See [forest_pursuit_edge][affinis.associations.forest_pursuit_edge].
    - If "counts", FP simply reports the estimated number of edge
      activations found from spanning trees in rows of X.
      See [forest_pursuit_cts][affinis.associations.forest_pursuit_cts].
    - If "forest-max", use Expected Forest Maximization, an E-M scheme
      to estimate the posterior probability of both the edge
      activations _and_ the underlying graph structure, via
      alternating minimization.
      See [expected_forest_maximization][affinis.associations.expected_forest_maximization].
    - If "interaction", re-weight the "edge-prob" values to be the
      likelihood of observing an edge activation, _given_ the
      observed data (useful e.g. for bayesian inference). 
      See [forest_pursuit_interaction][affinis.associations.forest_pursuit_interaction].

    Args:
      X: feature matrix
      mode: inference mode
      pseudocts: additive smoothing parameter
      prior_dists: previously calculated inter-node distances (default:`-log(cos)`)
      **efm_kws: kwargs to pass to
          [expected_forest_maximization][affinis.associations.expected_forest_maximization]
          if that mode is selected. 

    """
    match mode:
        case 'edge-prob':
            return forest_pursuit_edge(X, prior_dists=prior_dists, pseudocts=pseudocts)
        case 'counts':
            return forest_pursuit_cts(X, prior_dists=prior_dists)
        case 'forest-max':
            return expected_forest_maximization(X, **efm_kws)
        case 'interaction':
            return forest_pursuit_interaction(X, prior_dists, pseudocts=pseudocts)


def _forest_pursuit_normdegs(X, prior_dists=None, pseudocts="min-connect"):
    """DEPRECATED re-scaling of SFD Interraction probability that tends to
    move the highest-probability edges for each node toward 1.0.

    If a graph generating random walks has an underlying "simple, connected"
    structure (unweighted), then edges should tend toward "existing" if they
    are the "best possible" edge available to a given node. This function re-weights
    edges away from "interaction" probabilities by the geometric mean of the strongest
    interraction probabilities for each node, individually.

    I.e. even if an edge is somewhat rare because the node connected to it has many other
    edges (high-degree), our belief in its existence is weighted by its comparative strength of other interactions its two nodes experience.
    """
    e_margP = _sq(
        forest_pursuit_interaction(X, prior_dists=prior_dists, pseudocts=pseudocts)
    )
    gmean_max = _sq(_outer(np.multiply, np.sqrt(_sq(e_margP).max(axis=0))))
    return _sq(e_margP / gmean_max)
