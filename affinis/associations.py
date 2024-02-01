import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path, reconstruct_path
from scipy.sparse import coo_array, issparse
from .utils import _sq, _outer, _sparse_directed_to_symmetric, _get_masked_sq
from .priors import pseudocount

"""
All functions in this module take in data as _design matrices_ 
(i.e. observations x features), and return a feature association measure 
(i.e. features x features). 

Note that some of these functions return valid adjacency matrices (e.g. a feature 
is not associated to itself), while others return covariance or correlations (features 
are partially or fully correlated to themselves). 

Where appropriate, the methods here allow for additive/laplace smoothing, even in cases 
where this is not traditionally done (like cosine similarity). We give interpretations
of meaning that allow for this, where we can.

TODO: define type signatures by array shapes
TODO: dispatch on dataframes, static-frames, etc.
"""


def _gram(X1, X2):
    grammian = X1.T @ X2
    return grammian.toarray() if issparse(grammian) else grammian


def coocur_prob(X, pseudocts=0.5):
    """probability of a co-ocurrence per-observation"""

    cts = _gram(X, X)
    tot = X.shape[0]
    return pseudocount(pseudocts)(cts, tot)


# def cond_prob(X, pseudocts=0.):
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


def _marginal_prob(X):
    return [ct / X.shape[0] for ct in _marginal_cts(X)]


def _contingency_cts(X):
    nX = _not(X)
    both = _sq(_gram(X, X))
    one = _sq(_gram(X, nX))
    other = _sq(_gram(nX, X))
    neither = _sq(_gram(nX, nX))
    return neither, one, other, both


def _contingency_prob(X, pseudocts=0.5):
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


def odds_ratio(X, pseudocts=0.5):
    """Ratio of the odds of a true pos/neg to false pos/neg

    For associations, we replace pos/neg and true/false with
    a=yes/no and b=yes/no.
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(a * d / (b * c)) + np.eye(X.shape[1])


def mutual_information(X, pseudocts=0.5):
    """Mutual Information over binary random variables

    For use in e.g. Chow-Liu Trees
    """
    Pxy = np.array(_contingency_prob(X, pseudocts=pseudocts))

    # Pxy = np.vstack([sq(i) for i in cond_table])
    yes, no = _marginal_prob(X)

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


def chow_liu(X, pseudocts=0.5):
    """Chow-Liu Tree on the features of a (binary) design matrix

    computes mutual information over all pairs of features, and returns
    the maximum spanning tree on them. Assumes a symmetric adjacency is wanted.
    """
    # return sq(sq(minimum_spanning_tree(-MI_binary(X)).todense()))
    return _sq(
        _sq(
            minimum_spanning_tree(
                np.exp(-(mutual_information(X, pseudocts=pseudocts)))
            ).todense()
        )
    )


def yule_y(X, pseudocts=0.5):
    """a.k.a. Coefficient of Colligation.

    mobius transform of the Odds Ratio to the range [-1,1]
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := np.sqrt(a * d)) - (bc := np.sqrt(b * c))) / (ad + bc))
    # return ((sor:=np.sqrt(OR))-1)/(sor+1)
    # return ((ad:=np.sqrt(both*neither))-(bc:=np.sqrt(one*other)))/(ad+bc)


def yule_q(X, pseudocts=0.5):
    """a.k.a. Goodman & Kruskal's gamma for 2x2.

    mobius transform of the Odds Ratio to the range [-1,1]
    """
    a, b, c, d = _contingency_prob(X, pseudocts=pseudocts)
    return _sq(((ad := a * d) - (bc := b * c)) / (ad + bc))


def ochiai(X, pseudocts=0.5):
    """AKA cosine similarity on binary sets

    This code illustrates the idea that we can interpret it as conditional probability:
    The "exposure" of pairwise co-occurrences can't be larger than the sample size,
    so instead we approximate it as a psuedo-variable  having the geometric average of
    two original exposure rates $\sqrt{X_1X_2}$

    This interpretation has a nice side-effect of letting us "smooth" the measure with
    laplace/additive pseudocounts on each bernoulli(-ish) "co-occurrence variable".
    """
    # I = np.eye(X.shape[-1])
    co_occurs = _sq(_gram(X, X))  # + pseudocts
    exposures = X.sum(axis=0)
    pseudo_exposure = _sq(np.sqrt(_outer(np.multiply, exposures)))  # + 2 * pseudocts
    # return _sq(co_occurs) / pseudo_exposure + np.eye(X.shape[1])
    return _sq(pseudocount(pseudocts)(co_occurs, pseudo_exposure)) + np.eye(X.shape[1])


def binary_cosine_similarity(X, pseudocts=0.5):
    """alias of `ochiai(X), provided for user convenience"""
    return ochiai(X, pseudocts=pseudocts)


def resource_project(X):
    """bipartite project due to Zhao et al.

    Really just a step of sinkhorn-knopp on the bipartite adjacency
    https://doc.rero.ch/record/8503/files/zhang_bnp.pdf
    """
    P = _gram(((X.T) / (X.sum(axis=1))).T, ((X) / (X.sum(axis=0))))
    return np.maximum(P, P.T)


def high_salience_skeleton(X, prior=ochiai, pseudocts="min-connect"):
    """Grady et al. (2012)
    Calculates shortest paths from every node, and counts the
    number of trees each edge ended up being used in.
    """
    est_dists = np.abs(-np.log(prior(X, pseudocts=pseudocts)))
    d, pred = shortest_path(est_dists, return_predecessors=True)
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


def _spanning_forests_obs_bootstrap(X, prior_dists=None):
    """resample with a kernel bootstrap on MSTs in a manifold"""
    if not prior_dists:
        prior_dists = -np.log(ochiai(X, pseudocts=0.5))

    N_obs = X.toarray() if issparse(X) else X
    E_obs = coo_array(
        [
            _sq(
                minimum_spanning_tree(_get_masked_sq(_sq(prior_dists), i)).todense() > 0
            )
            for i in N_obs
        ]
    )
    return E_obs


def SFD_interaction_cts(X, prior_dists=None):
    """Point estimate for number of actual edge activations, rather than
    node-node co-occurrences.
    Uses the Spanning Forest Density (Forest for the Trees)
    """

    # est_dists = -np.log(prior(X, pseudocts=pseudocts))
    e_obs = _spanning_forests_obs_bootstrap(X, prior_dists=prior_dists)
    return _sq(e_obs.sum(axis=0))

    # def unroll_node_obs(X):
    #     trirow, tricol = np.triu_indices(n=X.shape[1],k=1)
    #     return np.einsum('ij,ik->ijk', X, X)[:,trirow, tricol]

    # project into edge-space, only retaining MST activations
    # co_prob = coocur_prob(X,pseudocts=pseudocts)
    # e_prob = (E_obs.sum(axis=0)+pseudocts)*_sq(co_prob)/_sq(_gram(X,X)+1)

    # (E_obs.sum(axis=0)+0.5)*_sq(coocur_prob(X, pseudocts=0.5))/_sq(X.T@X+1)


def SFD_edge_cond_prob(X, prior_dists=None, pseudocts="min-connect"):
    """point estimate for edge-activation probability, conditional on
    both nodes being a priori activated.
    Uses the Spanning Forest Density non-parametric estimator
    """
    e_cts = _sq(SFD_interaction_cts(X, prior_dists=prior_dists))
    uv_cts = _sq(_gram(X, X))
    # e_prob = (e_cts + pseudocts) / (uv_cts + 2 * pseudocts)
    e_prob = pseudocount(pseudocts)(e_cts, uv_cts)
    return _sq(e_prob)


def SFD_interaction_prob(X, prior_dists=None, pseudocts=0.5):
    """point estimate for edge-activation probability using the
    Spanning Forest Density non-parametric estimator
    """
    e_prob = _sq(SFD_edge_cond_prob(X, prior_dists=prior_dists, pseudocts=pseudocts))
    uv_prob = _sq(coocur_prob(X, pseudocts=pseudocts))
    e_margP = e_prob * uv_prob
    return _sq(e_margP)


def SFD_edge_prob(X, prior_dists=None, pseudocts=0.5):
    """re-scaling of SFD Interraction probability that tends to
    move the highest-probability edges for each node toward 1.0.

    If a graph generating random walks has an underlying "simple, connected"
    structure (unweighted), then edges should tend toward "existing" if they
    are the "best possible" edge available to a given node. This function re-weights
    edges away from "interaction" probabilities by the geometric mean of the strongest
    interraction probabilities for each node, individually.

    I.e. even if an edge is somewhat rare because the node connected to it has many other
    edges (high-degree), our belief in its existence is weighted by its comparative strength
    of other interactions its two nodes experience.
    """
    e_margP = _sq(SFD_interaction_prob(X, prior_dists=prior_dists, pseudocts=pseudocts))
    gmean_max = _sq(_outer(np.multiply, np.sqrt(_sq(e_margP).max(axis=0))))
    return _sq(e_margP / gmean_max)
