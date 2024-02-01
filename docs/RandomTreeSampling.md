---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: affinis
  language: python
  name: affinis
---

```{code-cell} ipython3
from affinis.sample import _random_walk, _random_tree_jump
from affinis.plots import hinton
from affinis import utils
# from affinis.utils import _rank1_update, _rank1_downdate 
from affinis.proximity import forest
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from scipy.spatial.distance import squareform
from scipy.sparse import csr_array, coo_array

from itertools import pairwise, count, chain, takewhile
from toolz import groupby, countby, mapcat
```

## Cut-bridge

```{code-cell} ipython3
# def forest(L, β=1):
#     return np.linalg.inv(np.eye(L.shape[0])+β*L)

# def rank1_downdate(B,u):
#     return B + (B @ u) @ (u.T @ B) / (1 - u.T @ B @ u)


tree = nx.random_tree(20)
pos=nx.layout.kamada_kawai_layout(tree)


A=nx.to_numpy_array(tree)
Q=forest(L:=nx.laplacian_matrix(tree).toarray())
B = nx.incidence_matrix(tree, oriented=True).toarray().T
plt.subplot(121, aspect='equal')
nx.draw_networkx(tree, pos=pos)
plt.subplot(122)
hinton(A)
```

```{code-cell} ipython3
bridge_p = squareform(Q, checks=False)
# cut_p = squareform(A)
# cut_p = np.ones(9)/9
# cut_p, bridge_p
```

```{code-cell} ipython3
from affinis.sample import _random_tree_gibbs

tree_gen = _random_tree_gibbs(bridge_p, B)
```

```{code-cell} ipython3
plt.figure(figsize=(8,8))
for i in range(1,17):
    plt.subplot(4,4,i)
    B_new = next(tree_gen)
    hinton(B_new.T@B_new)
    # plt.spy(B_new.T@B_new)
```

```{code-cell} ipython3
%timeit next(tree_gen)
```

```{code-cell} ipython3
f,ax=plt.subplots(ncols=2, nrows=1)

# plt.subplot(121, aspect=1.2)
nx.draw_networkx(tree, pos=pos, ax=ax[0], edge_color='grey')
nx.draw_networkx_edges(
    Tree_new:=nx.from_numpy_array(-squareform(squareform((B_new:=next(tree_gen)).T@B_new, 
                                              checks=False)), create_using=nx.Graph), 
    pos=pos, edge_color='xkcd:sky', ax=ax[0]
)
ax[0].set_title('original',color='grey')
# plt.subplot(122, aspect=1.2)
nx.draw_networkx(
    Tree_new, pos=(new_pos:=nx.kamada_kawai_layout(Tree_new)), ax=ax[1], edge_color='grey'
)
nx.draw_networkx_edges(tree, pos=new_pos, edge_color='xkcd:rust', ax=ax[1])
ax[1].set_title('after (many) samples',color='grey',)
f.patch.set_alpha(0.)
ax[0].patch.set_alpha(0.)
ax[1].patch.set_alpha(0.)
plt.tight_layout()
```

```{code-cell} ipython3
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree
from affinis.distance import generalized_graph_dists


mst=minimum_spanning_tree(generalized_graph_dists(L, beta=1)).tocoo()

# plt.spy(coo_array((np.ones(19), (np.arange(19),mst.col))))
# hinton()
B_tree=utils.sparse_adj_to_incidence(mst).todense()
hinton(B_tree.T@B_tree)
B_tree
```

```{code-cell} ipython3
# utils._upper_triangular_to_symmetric(
sel=np.random.choice(np.arange(20), 4)
plt.spy(minimum_spanning_tree(Q[sel][:,sel]))
tst=minimum_spanning_tree(Q[sel][:,sel]).tocoo()
n=tst.col.shape[0]
ones,idx=np.ones(n), np.arange(n)
coo_array((ones, (idx,tst.row)), shape=(n,tst.shape[0]))
```

```{code-cell} ipython3
nx.draw_networkx(nx.from_numpy_matrix(np.diag(np.abs(B_tree).sum(axis=0))-B_tree.T@B_tree), pos=pos)
```

```{code-cell} ipython3
np.linalg.inv(
    (Vhalf:=np.diag(np.sqrt(np.diag(Q))))
    @(W:=L+np.eye(L.shape[0]))@Vhalf
)

Ωinv = Vhalf@W@Vhalf
-Ωinv/np.sqrt(np.multiply.outer(ωii:=np.diag(Ωinv), ωii))
# hinton(np.linalg.inv(Ωinv))
```

```{code-cell} ipython3
-W/np.sqrt(np.multiply.outer(wii:=np.diag(W), wii))
```

```{code-cell} ipython3
np.exp(-generalized_graph_dists(L))
```

```{code-cell} ipython3
from itertools import islice
trees=list(islice(tree_gen, 50))
trees[0]
```

```{code-cell} ipython3
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()

im = ax.spy(B.T@B)

def update_hinton(i):
    # im_normed = np.random.rand(6, 6)
    # ax.imshow(im_normed)
    # ax.set_axis_off()
    B = trees[i]
    im.set_data(B.T@B)
    # anim=hinton(B.T@B, ax=ax)
    return (im,)
anim = FuncAnimation(fig, update_hinton, frames=50, interval=50)
HTML(anim.to_html5_video())
# plt.show()
```

```{code-cell} ipython3
# squareform(A)*np.tril_indices(10)
nx.draw_networkx(tree, pos=pos)
which_del=np.random.choice(9)
to_del =np.array(np.where(A*np.tril(np.ones(10))))[:,which_del]
nx.draw_networkx_edges(tree, pos=pos, edgelist=[to_del.tolist()], edge_color='r', width=3)

tree.edges, to_del
```

```{code-cell} ipython3
hinton(Q)
```

```{code-cell} ipython3
np.outer(Q[4]-Q[2], Q[:,4]-Q[:,2])
```

```{code-cell} ipython3
# u=np.diff(np.eye(10)[to_del], axis=0)[:,...].T

# u=B[which_del,:,np.newaxis]
u = np.atleast_2d(B[which_del])
subtreeQ = np.copy(Q)
_rank1_downdate(subtreeQ, u)
hinton(subtreeQ)
plt.spy(np.outer(u,u), marker='.')
# hinton(forest(L-np.multiply.outer(u,u)))
# np.diff(np.eye(10)[to_del], axis=0)
u
```

```{code-cell} ipython3
%timeit _rank1_downdate(np.copy(Q), u)
```

```{code-cell} ipython3
%timeit forest(L - np.outer(u,u))
```

```{code-cell} ipython3
# tree.remove_edge(*list(tree.edges)[np.random.choice(9)])
squareform(np.isclose(subtreeQ, np.zeros(10)))
```

```{code-cell} ipython3
bridge_p = squareform(Q, checks=False)*squareform(np.isclose(subtreeQ, np.zeros(10)))
bridge_p
```

```{code-cell} ipython3
def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
hinton(squareform(bridge_p/bridge_p.sum()))
plt.spy(np.outer(u,u), marker='.')
```

```{code-cell} ipython3
new_e = np.eye(45)[np.random.choice(45,p=bridge_p/bridge_p.sum())]
new_A = squareform(
    -squareform(L-np.outer(u,u), checks=False)  # A/e
    + 
    new_e  # e'
)
to_add=np.where(squareform(new_e)!=0)[0]

# pos=nx.layout.kamada_kawai_layout(tree)

nx.draw_networkx(tree, pos=pos)
nx.draw_networkx_edges(tree, pos=pos, edgelist=[to_del.tolist()], edge_color='r', width=3)
nx.draw_networkx_edges(nx.from_numpy_array(new_A, create_using=nx.DiGraph), edgelist=[to_add.tolist()], 
                       pos=pos,connectionstyle='arc3,rad=0.2', edge_color='g', width=3)
to_del, to_add
```

```{code-cell} ipython3
hinton(_rank1_update(subtreeQ, np.diff(np.eye(10)[to_add], axis=0)))
```

```{code-cell} ipython3
np.allclose(
    _rank1_update(subtreeQ, np.diff(np.eye(10)[to_add], axis=0)), # cheap way
    forest(np.diag(new_A.sum(axis=1))-new_A)  # true way
)  # same? 
```

```{code-cell} ipython3
new_B=np.copy(B)
new_Q = forest(new_B.T@new_B)

for i in range(B.shape[0]): 
    new_Q = _rank1_downdate(new_Q, new_B[i])
    # new_Q = _rank1_downdate(forest(new_B.T@new_B), new_B[i])
    bridge_p = squareform(Q, checks=False)*squareform(np.isclose(new_Q, np.zeros(B.shape[1])))
    new_edge = np.eye(bridge_p.shape[0])[np.random.choice(bridge_p.shape[0],p=bridge_p/bridge_p.sum())]
    new_e = np.diff(np.eye(B.shape[1])[coo_array(squareform(new_edge)).row], axis=0)
    new_Q = _rank1_update(new_Q, new_e)
    new_B[i] = new_e
    new_A = squareform(
        -squareform(L-np.outer(u,u), checks=False)  # A/e
        + 
        new_e  # e'
    )
new_L = new_B.T@new_B
new_A = np.diag(np.diag(new_L)) - new_L
new_T = nx.from_numpy_array(new_A)

f,ax=plt.subplots(ncols=2)
nx.draw_networkx(tree, pos=pos, ax=ax[0])
nx.draw_networkx(new_T, pos=nx.layout.kamada_kawai_layout(new_T), edge_color='g', width=2, ax=ax[1])
```

```{code-cell} ipython3
new_B[0]
```

```{code-cell} ipython3
forest(L)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from scipy.linalg import inv as scinv
from numpy.linalg import inv as npinv
from scipy.sparse.linalg import inv as spinv
from scipy.linalg import lapack

from functools import lru_cache


@lru_cache
def inds_cache(n): 
    return np.tri(n, k=-1, dtype=bool)

def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    inds = inds_cache(n)
    ut[inds] = ut.T[inds]

def fast_positive_definite_inverse(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv
```

```{code-cell} ipython3
%timeit scinv(L+np.eye(10))
%timeit npinv(L+np.eye(10))
%timeit spinv(csr_array(L+np.eye(10)))
%timeit fast_positive_definite_inverse(L+np.eye(10))
```

```{code-cell} ipython3
%timeit np.eye(50)[1] - np.eye(50)[5]
%timeit np.diff(np.eye(50)[[1,5]], axis=0)
```

```{code-cell} ipython3
np.diff(np.eye(50)[[1,5]], axis=0)
```

```{code-cell} ipython3
np.arange(25).reshape((5,5)), squareform(np.arange(25).reshape((5,5)), checks=False), np.arange(25).reshape((5,5))[np.triu_indices(5, k=1)]
```

```{code-cell} ipython3
%timeit np.arange(25).reshape((5,5))[np.triu_indices(5, k=1)]
%timeit squareform(np.arange(25).reshape((5,5)), checks=False)
```

```{code-cell} ipython3
list(chain(*[walk[i.begin:i.end] for i in final]))
```

```{code-cell} ipython3
G = nx.complete_graph(10)
G.nodes
nx.draw_kamada_kawai(G, with_labels=True)
# nx.draw_networkx(
```

```{code-cell} ipython3
A = nx.to_numpy_array(G)
M = A/A.sum(axis=0)
M_cdf = np.cumsum(M.T, axis=1)
start=0
walk_gen = _random_walk(M_cdf, start)
# np.searchsorted(M_cdf[0], np.random.rand())
# for i in range(50): 
#     print(next(walk_gen))
# M_cdf
```

```{code-cell} ipython3
# walk = [1,1,2,3,4,2,3,4,5,5,6]
from typing import NamedTuple
from intervaltree import Interval, IntervalTree

class RandJump(NamedTuple): 
    jump_id: int
    start: int
    end: int

sink = 8
walk = list(pairwise([start]+list(takewhile(lambda x: x not in [sink], walk_gen))+[sink]))
# groupby(1, (RandJump(i,s,t) for i,(s,t) in enumerate(walk)))
cycle_start_end = mapcat(pairwise, groupby(1, (RandJump(i,s,t) for i,(s,t) in enumerate(walk))).values())

# list(cycle_start_end)
# intervals = sorted([walk[i[0].jump_id:i[1].jump_id]
#                     for i in cycle_start_end],
#                    key=lambda i: i[0][0])
# walk, intervals
intervals = sorted([Interval(from_idx:=i[0].jump_id, to_idx:=i[1].jump_id, walk[from_idx:to_idx]) 
                    for i in cycle_start_end],
                   key=lambda i: i.begin)
walk, intervals
```

```{code-cell} ipython3
tree = IntervalTree(intervals)
len(tree)
popped = []
unpopped = intervals.copy()

## NOTE sort by node visited (groupby) and then by start index
while not tree.is_empty():
    i = unpopped.pop(0)
    if i in tree and len(i.data)>1:
        tree.remove_overlap(i.begin, i.end)
        print(tree)
        popped += [i]
popped
# for i in intervals: 
#     tree.remove_overlap(intervals[0].begin, intervals[0].end)
```

```{code-cell} ipython3
deleted = IntervalTree(popped)
deleted.merge_overlaps()
deleted
```

```{code-cell} ipython3
final = IntervalTree([Interval(0,len(walk))])
for i in deleted: 
    final.chop(i.begin, i.end)
final
```

```{code-cell} ipython3

```
