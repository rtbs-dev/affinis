---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: affinis
  language: python
  name: affinis
---

# Online Tree Metric Approximation through Hyperbolic Commute Kernels

Building on the previous (observation-MSTs). 

What if the ultimate goal of an annotator was _specifically_ to extract tree-like structure, such as taxonomy. 
See: nickel & kiela, etc. 
In this case, we need to enforce a tree-like distance metric that is used prior to downstream 
(...)

## Main idea

Rather than simply using DS or a commute-kernel, we want a graph-smoothing process that _also_ biases the kernel toward being "tree-like". This is the hyperbolicity, and we can measure the distortion of a given metric space using the gromov product and the 4-point condition. 

We show that the use of an exponential map of a graph's Laplacian can be performed by viewing it as a grammian matrix of the vertices existing on a high-dimensional "edge" manifold. 
This is the same as using the incidence matrix (i.e.. the bipartite adjacency matrix of the Caley graph) as a set of "observations", with observations $V$ and feature space of dimension $\|E\|$. 

if we say the original vertices are data points $V\in\mathbb{R}^{\|E\|}$, then we actually recover exactly the original Laplacian $L$ as the linear kernel! 

Instead, we assume $V\in\mathcal{H}^{\|E\|}$, where $\mathcal{H}^{n}$ is the n-dimensional upper-half plane (lorentzian) hyperbolic manifold embedded in $\mathbb{R}^{n+1}$ 
We can now compute a "hyperbolic laplacian", by interpreting it as the exponential map of the linear kernel into the hyperbolic plane.

```{code-cell} ipython3

```

## Euclidean Case

```{code-cell} ipython3

```
