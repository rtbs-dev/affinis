from __future__ import annotations
from plum import dispatch

# from numbers import Number
from typing import Callable, Literal, TypeAlias
import numpy as np
from jaxtyping import Num, Shaped
from sparse import COO
from affinis.types import Number, PsdCts, Arr

ElemWise: TypeAlias = Shaped[np.ndarray, "*elems"]
"""generic batched typed array"""

ElemReduceFunc: TypeAlias = Callable[[ElemWise, ElemWise], ElemWise]
"""merges two arrays into one e.g. a numerator and denominator"""


def _safe_div(num: ElemWise, den: ElemWise) -> ElemWise:
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=den != 0,
    )


@dispatch(precedence=1)
def beta_binom_post(succs: ElemWise, trial: ElemWise|Number, a: float, b: float) -> ElemWise:
    return np.divide(
        succs+a,
        trial+a+b,
        out=np.zeros_like(succs, dtype=float),
        where=(trial!=0),
    )

@dispatch
def beta_binom_post(  #type:ignore
    succs: Shaped[COO, "*elems"],
    trial: Shaped[COO, "*elems"],
    a:float, b:float
)-> Shaped[COO, "*elems"]:
    s_flat = np.ravel_multi_index(succs.coords, trial.shape)
    t_flat = np.ravel_multi_index(trial.coords, trial.shape)
    t_idx = np.searchsorted(t_flat, s_flat)
    assert np.all(
        (t_idx < len(t_flat)) &
        (t_flat[t_idx] == s_flat)
    ), "Binomial assumption violated: S ⊄ T"
    # align_succs = succs[tuple(trial.coords)]
    align_succs = np.zeros_like(trial.data)
    align_succs[t_idx] = succs.data
    sparse_post = beta_binom_post(align_succs, trial.data, a, b)
    return COO(
        coords=trial.coords,
        data=sparse_post,
        shape=trial.shape,
        fill_value = a/(a+b)
    )

@dispatch
def beta_binom_post(  #type:ignore
    succs: Shaped[COO, "*elems"],
    trial: Number,
    a: float, b:float,
)-> Shaped[COO, "*elems"]:
    sparse_post = beta_binom_post(succs.data, trial, a, b)
    return COO(
        coords=succs.coords,
        data=sparse_post,
        shape=succs.shape,
        fill_value=a/(trial+a+b)
    )

    # lower = den.data + (a+b)
    # fill_prob = a/(a+b)
    # post_data = num.data+
    
    


@dispatch
def pseudocount(prior: Number) -> ElemReduceFunc:
    """Additive smoothing binomial with symmetric beta prior (a = b)"""

    def _beta_binom_post(num: ElemWise, den: ElemWise) -> ElemWise:
        return beta_binom_post(num,den,prior,2*prior)
    return _beta_binom_post


@dispatch
def pseudocount(prior: tuple[Number, Number]) -> ElemReduceFunc: # type:ignore
    """additive smoothing binomial with (possibly) asymmetric prior"""
    a, b = prior

    def _beta_binom_post(num: ElemWise, den: ElemWise) -> ElemWise:
        return beta_binom_post(num,den,a,b)

    return _beta_binom_post


@dispatch(precedence=1)
def pseudocount(prior: Literal["min-connect"]) -> ElemReduceFunc:  # type:ignore
    r"""additive smoothing binomial with asymmetric prior biasing sparsity

    NOTE: Requires triangular-number `n` (i.e. lower triangles of square matrix).

    If observations are trials over an array of graph edges, then
    the number of edges that are on or off in a graph is "zero sum" (one extra "on" means one less "off).
    So, the proportion of time we will be observing an "on" edge might be thought of as a Wiener
    Process, and thus follows a (generalized) arcsine distribution.
    This means we need a "bathtub" prior `a+b=1`.
    As it happens, the expected value for this will be `a`.


    If a complete graph has `n(n-2)/2` edges, while a min. connected one has `n-1`, then
    we can bias toward sparsity such that the expected ratio of edges to possible edges
    (and therefore the expected value of our bathtub prior) should be:

    $$
    \frac{(n-1)}{\frac{n*(n-1)}{2}}
    $$

    This comes out to `a=2/n`, `b=1-2/n`, so

    $$
    P(p|a,b) = \frac{\textrm{\#successes}+2/n}{\textrm{\#trials}+1}
    $$

    Args:
      prior: "min-connect"



    """

    def _beta_binom_post(num: ElemWise, den: ElemWise) -> ElemWise:

        n_edges = num.shape[0]  # in complete graph
        n = (1 + np.sqrt(1 + 8 * n_edges)) / 2.0  # nodes in complete graph
        return beta_binom_post(num, den, 2./n, 1-2./n)

    return _beta_binom_post


@dispatch
def pseudocount(prior: tuple[Literal["zero-sum"], Number]) -> ElemReduceFunc:  # type:ignore
    """TODO derive the approx-cts for projection onto simplex

    unlike the other methods, this directly returns the `a` values for a
    Beta(a, 1-a) distribution. For use when the full
    Beta distribution is desired e.g. for active learning or uncertainty.

    (it turns out that a/(a+1-a) == a, so this is also the mean of the distribution)
    """
    a = prior[1]
    # b = 1 - a

    def _beta_binom_post(suc: ElemWise, tot: ElemWise) -> ElemWise:
        return beta_binom_post(suc, tot, a, 1-a)
        ## Clever but no longer necessary
        # c = (suc - a * tot) / (tot + 1)
        # a_n = a + c
        # return a_n

    return _beta_binom_post


@dispatch
def pseudocount(  # type:ignore
    prior: tuple[Literal["zero-sum"], Literal["min-connect"]],
) -> ElemReduceFunc:
    """A combination of a zero-sum (a, 1-a) beta prior and a=2/n for tree-like"""

    # def _beta_binom_post(suc: ElemWise, tot: ElemWise) -> ElemWise:
    #     n_edges = suc.shape[0]
    #     n = (1 + np.sqrt(1 + 8 * n_edges)) / 2.0  # nodes in complete graph
    #     return beta_binom_post(suc, tot, 2./n, 1-2/n.)
    #     # a = 2 / n
    #     # b = 1 - a
    #     c = (suc - a * tot) / (tot + 1)
    #     a_n = a + c
    #     # b_n = 1 - a_n
    #     # return _safe_div(a_n, a_n + b_n)
    #     return a_n

    # return _beta_binom_post
    return pseudocount("min-connect")  # same


@dispatch.abstract
def pseudocount(prior: PsdCts) -> ElemReduceFunc:  # type:ignore
    """Additive binomial smoothing via beta prior (beta-binomial)

    Can accept a variety of prior settings, which are described below, and
    handled via [plum dispatch](https://beartype.github.io/plum/intro.html).

    - single float (e.g. `0.5`): Symmetric beta prior (a=b). Common choices
      of symmetric beta prior include a=0.0 (Haldane), a=1.0 (Laplace), and
      a=0.5 (Jeffrey's). `affinis` tends to use the 0.5 Jeffrey's prior as
      default, unless otherwise noted (see [here](https://en.wikipedia.org/wiki/Beta_distribution#Bayesian_inference) for a deeper discussion).
    - pair of floats (e.g. `(0.1,1.5)`): asymmetric prior
      (see [Beta Distribution Shapes](https://en.wikipedia.org/wiki/Beta_distribution#Shapes)).
    - `'zero-sum'`: passed along with a float (e.g. `('zero-sum',0.2)`).
      Indicates that b should be _derived_ from the provided parameter,
      such that `a+b=1`. This is a generalized arcsine distribution,
      Beta(a, 1-a). This can be useful to model expected fractions of a whole
      (ergo, "zero-sum"); see [_What is the Arcsine Law_](https://math.osu.edu/sites/math.osu.edu/files/What_is_2018_Arcsine_Law.pdf).
    - `'min-connect'`: can be used instead of a single float. This will give
      a different `a` parameter, depending on the size of the `feat` dimension.
      Intuitively for network recovery, a maximally-sparse (connected) graph
      should be a tree, which has a number of edges linear in node-count
      (n-1) while the number of possible edges is quadratic (n choose 2).

    NOTE: Unlike the other options, 'min-connect' assumes that the passed
    arrays will have a shape that can be folded as a lower-triangle of a
    square matrix (i.e. a _triangular number_, n-choose-2)

    Args:
        prior:PsdCts: beta priors (a,b), either explicit or implictly derived.

    Returns:
        ElemReduceFunc
    """
    ...
