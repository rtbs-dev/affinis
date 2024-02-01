#!/usr/bin/env python3
from plum import dispatch
from numbers import Number
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray as AL
from affinis.utils import _sq

ElemReduceFunc = Callable[[AL, AL], AL]


def _safe_div(num: AL, den: AL) -> AL:
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=den != 0,
    )


@dispatch.abstract
def pseudocount(
    prior: tuple[Number, Number] | tuple[str, Number] | str | Number
) -> ElemReduceFunc:
    """Additive binomial smoothing via beta prior (beta-binomial)"""
    ...


@dispatch
def pseudocount(prior: Number) -> ElemReduceFunc:
    """additiv smoothing binomial with symmetric beta prior

    \\alpha == \\beta == prior

    Common cases
    ---
    |prior | \\alpha|
    | - | - |
    |Haldane | 0.|
    |Laplace | 1.|
    |Jeffreys | 0.5|
    ---
    """

    def _beta_binom_post(num: AL, den: AL) -> AL:
        return _safe_div(
            num + prior,
            den + 2 * prior,
        )

    return _beta_binom_post


@dispatch
def pseudocount(prior: tuple[Number, Number]) -> ElemReduceFunc:
    """additive smoothing binomial with (possibly) asymmetric prior"""
    a, b = prior

    def _beta_binom_post(num: AL, den: AL) -> AL:
        return _safe_div(num + a, den + a + b)

    return _beta_binom_post


@dispatch(precedence=1)
def pseudocount(prior: Literal["min-connect"]) -> ElemReduceFunc:
    """additive smoothing binomial with asymmetric prior biasing sparsity

    if observations are trials over an array of graph edges.

    The number of edges that are on or off in a graph is "zero sum"... One extra "on" means
    one less "off.
    So, the proportion of time we will be observing an "on" edge might be thought of as a Wiener
    Process, and thus follows a (generalized) arcsine distribution.
    This means we need a "bathtub" prior (a,b<1).

    For a concave beta (a,b <1), the anti-mode is the least likely spot, with the two
    (limiting) modes being at 0,1.
    If a complete graph has `n(n-2)/2` edges, while a min. connected one has `n-1`, then
    we can bias toward non-edges such that the least-likely p is the ratio `(1-(n-1)/(n*(n-1)/2)))`

    This comes out to `a=2/n, b=1-2/n`, so the P(p|a,b) = (succ+2/n)/(trials+1)
    """

    def _beta_binom_post(num: AL, den: AL) -> AL:
        n = _sq(num).shape[0]
        return _safe_div(num + 2.0 / n, den + 1.0)

    return _beta_binom_post


@dispatch
def pseudocount(prior: tuple[Literal["zero-sum"], Number]) -> ElemReduceFunc:
    """TODO derive the approx-cts for projection onto simplex"""
    a = prior[1]
    b = 1 - a

    def _beta_binom_post(suc: AL, tot: AL) -> AL:
        c = (suc - a * tot) / (tot + 1)
        a_n = a + c
        b_n = 1 - a_n
        return _safe_div(a_n, a_n + b_n)

    return _beta_binom_post


@dispatch
def pseudocount(
    prior: tuple[Literal["zero-sum"], Literal["min-connect"]]
) -> ElemReduceFunc:
    """A combination of a zero-sum (a, 1-a) beta prior and a=2/n for tree-like"""

    def _beta_binom_post(suc: AL, tot: AL) -> AL:
        n = _sq(suc).shape[0]
        a = 2 / n
        b = 1 - a
        c = (suc - a * tot) / (tot + 1)
        a_n = a + c
        b_n = 1 - a_n
        return _safe_div(a_n, a_n + b_n)

    return _beta_binom_post
