---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: affinis
  language: python
  name: affinis
---

```{code-cell} ipython3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from affinis.filter import min_connected_filter
import scipy.sparse as sprs
from affinis.distance import bilinear_dists, generalized_graph_dists
from affinis.proximity import sinkhorn, forest
from affinis.utils import _norm_diag, edge_mask_to_laplacian, _outer
# import matplotlib 
# %matplotlib notebook
```

```{code-cell} ipython3
rng = np.random.default_rng(2)
```

```{code-cell} ipython3
df = pd.read_csv('../data/snafu_sample.csv', dtype={'category':'category'})
idlist=df.id.rename('idlist').str.cat(df.listnum.astype(str))
df = df.assign(
    item=df['item']
     .str.replace('aligator', 'alligator')
     .str.replace('^a+rdva+rk', 'aardvark')
     .str.replace('baboob', 'baboon')
     .str.replace('antaloupe', 'antelope'),
    idlist=idlist
)
df=df.set_index([idlist, 'item'], drop=False)
df
# df.set_index(idlist, name='idlist')
# df.query('category=="animals"')['item'].sort_values().unique().tolist()
```

```{code-cell} ipython3

# idlist=df.id.str.cat(df.listnum.astype(str))
# idlist
animals = (df
#  .assign(idlist=idlist)
#  [df.item.isin((df.item.value_counts()>20).index.tolist())]#index.tolist())]

#  [df.item.isin((df.item.value_counts().pipe(lambda s: s[s>20])).index.tolist())]
 .query('category=="animals"')#[['idlist','item']]
 .assign(animals=1.)['animals']
#  .set_index(['idlist','item'])
 .pipe(lambda df: df[~df.index.duplicated(keep='first')])
 .unstack().fillna(0.)#.drop_levels(0)
 .pipe(lambda df: df.loc[:,df.sum()>50])
)

all_X = animals.values
animals
```

```{code-cell} ipython3
sns.histplot(df.groupby(level=0).item.count())
(df.groupby(level=0).item.count()).min()
```

```{code-cell} ipython3
plt.imshow(animals.T@animals)
```

```{code-cell} ipython3
# sliding_window()
from numpy.lib.stride_tricks import as_strided, sliding_window_view
def arr_cooc(x, n=2):
    # print(x.strides)
    # return as_strided(x, shape=(x.shape[0], n), strides=x.strides*2)
    # print(x.shape)
    return sliding_window_view(x,min(n, x.shape[0]))
# as_strided(df.item.values, shape=(df.length,2), strides=
# (arr_cooc(df.item.values, 3))
# df.groupby(level=0)['item']
animal_occ = (df
#  [df.item.isin(df.item.value_counts(ascending=False).head(100).index.tolist())]
#  [df.item.isin((df.item.value_counts().pipe(lambda s: s[s>20])).index.tolist())]
 .query('category=="animals"')
 .item.astype('category')
)
dummies = np.eye(animal_occ.dtype.categories.shape[0])
roll_X=np.vstack([dummies[arr_cooc(g[1].values, n=10),:].max(axis=1) for g in animal_occ.cat.codes.groupby(level=0)])#[:,]
roll_X = roll_X[:,animal_occ[animal_occ.isin(animals.columns.tolist())].cat.codes.unique()]
# print(animal_occ.cat.codes.sort_values())
# print(animal_cooc.max(axis=0))
# animal_cooc.astype(int).sum(axis=1)
# roll_X=np.array([dummies[ix] for ix in animal_cooc.T])#.sum(axis=1)
# roll_X = roll_X.astype(bool).max(axis=0)
# animal_occ
# animal_occ.cat.codes
# np.eye(animal_occ.dtype.categories.shape[0])[animal_cooc[0]].shape
# animal_occ.cat.codes.unique()
# roll_animals=
# plt.spy(roll_X)
# dummies[arr_cooc(animal_occ.cat.codes.values, n=3)].max(axis=1)#.shape#.shape#max(axis=0)#.sum(axis=0)#.max(axis=0)
roll_X.shape
# animal_occ[animal_occ.isin(animals.columns.tolist())].cat.codes.values
```

```{code-cell} ipython3
from affinis.associations import ochiai, resource_project, chow_liu, coocur_prob
from affinis.utils import _sq

# X = all_X
X = roll_X


sns.histplot(_sq(ochiai(all_X, pseudocts=0.5)), stat='density')
sns.histplot(_sq(ochiai(roll_X, pseudocts=0.5)), stat='density')
# sns.displot(_sq(resource_project(X)))
```

```{code-cell} ipython3
plt.figure(figsize=(15,15))
G = nx.from_pandas_adjacency(pd.DataFrame(chow_liu(X, pseudocts=0.5), index=animals.columns, columns=animals.columns))
pos_tree = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos=pos_tree, node_color='w')
```

```{code-cell} ipython3



def top_tree_pct(x, mult=1):
    pct=np.percentile(_sq(x), 100-100*mult*2/x.shape[0])
    print(pct)
    return x>=pct
plt.figure(figsize=(15,15))
G = nx.from_pandas_adjacency(pd.DataFrame(_sq(~(min_connected_filter(_sq(sinkhorn(_sq(_sq(X.T@X))))).mask)), index=animals.columns, columns=animals.columns))
# pos_cos = nx.kamada_kawai_layout(G, dist = pd.DataFrame(-np.log(ochiai(X)), columns=animals.columns, index=animals.columns).to_dict())
pos_cos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos=pos_cos, node_color='w')
nx.connected.is_connected(G)
```

```{code-cell} ipython3
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
# graphical_lasso()
plt.figure(figsize=(15,15))
glasso = (lambda o: o/(1+o))(np.abs(
    _sq(
        GraphicalLassoCV()#covariance='precomputed')
        # .fit(ochiai(X))
        .fit(X)
        .get_precision()
)))
G = nx.from_pandas_adjacency(pd.DataFrame(_sq(~min_connected_filter(glasso).mask), index=animals.columns, columns=animals.columns))
# pos_cos = nx.kamada_kawai_layout(G, dist = pd.DataFrame(-np.log(ochiai(X)), columns=animals.columns, index=animals.columns).to_dict())
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos=pos, node_color='w')
nx.connected.is_connected(G)
# glasso
# sns.histplot(glasso)

# _sq(min_connected_filter(glasso)>0)#.shape, X.shape, animals.columns.shape
```

```{code-cell} ipython3

def get_mask(e_pmf,idx):
    return sprs.coo_array(_sq(e_pmf)*
                          np.multiply.outer(idx,idx))

def unroll_node_obs(X): 
    trirow, tricol = np.triu_indices(n=X.shape[1],k=1)
    return np.einsum('ij,ik->ijk', X, X)[:,trirow, tricol]

def obs_mask_dists(X,d):
    n = X.shape[0]    
    mask = unroll_node_obs(X)
    return np.ma.masked_array(np.multiply.outer(np.ones(n), _sq(d)), mask=~(mask.astype(bool)))

def masked_subset_mst(x_d):
    A = _sq(x_d.compressed())
    T = sprs.csgraph.minimum_spanning_tree(A).todense()
    x_t = np.zeros_like(x_d)
    x_t[~x_d.mask] = _sq(T)
    return x_t

# est_dists = bilinear_dists(sinkhorn(_norm_diag(coocur_prob(roll_X, pseudocts=0.5))))
# est_dists = bilinear_dists(mutual_information(X, pseudocts=0.5))
est_dists = np.abs(-np.log(ochiai(sprs.csr_array(X), pseudocts=0.5)))

# masked_dists = np.ma.array(np.multiply.outer(np.ones(roll_X.shape[0]), _sq(ochiai(roll_X))), mask=unroll_node_obs(~roll_X.astype(bool)))
# masked_dists=obs_mask_dists(X, est_dists)
# masked_subset_mst(masked_dists[0])
# for n,i in enumerate(masked_dists):
#     # print(i.count(), (_sq(np.multiply.outer(roll_X[n], roll_X[n]))).sum(), roll_X[n].sum())
#     print(masked_subset_mst(i)>0)
# _sq(sprs.csgraph.minimum_spanning_tree(_sq(masked_dists[0].compressed())).todense())
# unroll_node_obs(roll_X).shape, np.multiply.outer(np.ones(roll_X.shape[0]), _sq(est_dists)).shape
# masked_dists[1]
# (lambda x: (x,(x**2-x)/2.))(_outer(np.multiply, roll_X[2]).sum())
# (unroll_node_obs(roll_X)[1].astype(bool)).sum()
```

```{code-cell} ipython3
from affinis.associations import high_salience_skeleton
# d,pred = sprs.csgraph.shortest_path(est_dists, return_predecessors=True)
# E_obs = np.array([_sq(sprs.csgraph.reconstruct_path(d, p).astype(bool).toarray()) for p in pred])
# hss = (E_obs.sum(axis=0)+0.5)/(E_obs.shape[0]+1)
f,ax = plt.subplots(ncols=2, figsize=(12,4))
# hss = (E_obs.T*(X.sum(axis=0)/X.shape[0])).T.sum(axis=0)
hss = high_salience_skeleton(X, pseudocts=('zero-sum','min-connect'))
np.histogram(hss)
sns.heatmap(sinkhorn(hss*ochiai(X)), ax=ax[0], square=True)
sns.heatmap(sinkhorn(_sq(_sq(ochiai(X)))), ax=ax[1], square=True)
plt.tight_layout()
# X.sum(axis=0)/X.shape[0]
# np.histogram(E_obs.sum(axis=0), bins=range(15))
# np.array([_sq(sprs.csgraph.reconstruct_path(np.ones_like(est_dists), p).toarray()) for p in pred]).sum(axis=0)
```

```{code-cell} ipython3
plt.figure(figsize=(15,15))
# G = nx.from_pandas_adjacency(pd.DataFrame(_sq(~min_connected_filter(_sq(sinkhorn(_sq(hss)*ochiai(X)))).mask), index=animals.columns, columns=animals.columns))

G = nx.from_pandas_adjacency(pd.DataFrame(_sq(~min_connected_filter(_sq(sinkhorn(hss*ochiai(X)))).mask), index=animals.columns, columns=animals.columns))
# G = nx.from_pandas_adjacency(pd.DataFrame(hss>0.05, index=animals.columns, columns=animals.columns))

# pos_cos = nx.kamada_kawai_layout(G, dist = pd.DataFrame(-np.log(ochiai(X)), columns=animals.columns, index=animals.columns).to_dict())
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos=pos, node_color='w')
nx.connected.is_connected(G)
# np.abs(est_dists)
# np.histogram(E_obs.sum(axis=0), bins=range(15))
```

```{code-cell} ipython3
from affinis.distance import adjusted_forest_dists

from affinis.associations import SFD_edge_cond_prob, SFD_interaction_prob
evd_L = _sq(SFD_edge_cond_prob(sprs.csr_array(X), pseudocts=('zero-sum','min-connect')))
# mst_post = evd_L*X.shape[0]
post_L=evd_L*_sq(adjusted_forest_dists((lambda a: np.diag(a.sum(axis=0))-a)(_sq(evd_L)), beta=100))
# post_L = _sq(spanning_forests_edge_rate(X))
```

$P(y|x) = \frac{P(Y\bigcap X)}{P(X)}$

$P(x|y) = \frac{P(Y\bigcap X)}{P(Y)}$

so if $X\subset Y$, then $P(Y\bigcap X)=P(X)$ and $P(X|Y) = \frac{P(X)}{P(Y)}$ and 

$P(X) = P(X|Y)P(Y)$

let X be "probability of an interraction" and Y be "probability of a co-occurrence", we can measure that right side via spanning tree bootstraps, to estimate the left. 

Alternatively, if the co-occurrences are treated as rates (say, counts in a poisson or negative binomial), then we are alternatively deriving our estimate as the thinning parameter for each

```{code-cell} ipython3
f = plt.figure(figsize=(15,15))

# G = nx.from_pandas_adjacency(pd.DataFrame((minmax(sinkhorn(_sq(post_L)))>0.7).astype(int), index=animals.columns, columns=animals.columns))
# G = nx.from_pandas_adjacency(pd.DataFrame(_sq(_sq(top_tree_pct(_sq(post_L), mult=1.5))), index=animals.columns, columns=animals.columns))
Gthres = min_connected_filter(_sq(Gvals:=(_sq(evd_L))))
G = nx.from_pandas_adjacency(pd.DataFrame(_sq(Gthres.filled(0)), index=animals.columns, columns=animals.columns))
# Gneg = nx.from_pandas_adjacency(pd.DataFrame(
#     _sq(3*(threshold_edges_filter(_sq(Gvals), 0.9) - Gthres.min()).filled(0) + 0.5*Gthres.filled(0)),
#     index = animals.columns, columns = animals.columns,
# ))
# pos = nx.kamada_kawai_layout(G, dist=pd.DataFrame(_sq(-np.log(evd_L)), columns=animals.columns, index=animals.columns).to_dict())
pos = nx.kamada_kawai_layout(G)
# pos = nx.spring_layout(G, iterations=1000, k=2)

nx.draw_networkx(G, pos=pos, node_color='w')
# nx.draw_networkx_edges(G, pos=pos)
# nx.draw_networkx_labels(G, pos=pos, bbox = dict(facecolor = "xkcd:cement", edgecolor=None, joinstyle='round'))
# plt.savefig('animals.svg')
# list(G.neighbors('spider'))
nx.connected.is_connected(G)
# fig
```

```{code-cell} ipython3
from affinis.distance import adjusted_forest_dists, generalized_graph_dists
sns.heatmap(np.exp(-generalized_graph_dists(edge_mask_to_laplacian(min_connected_filter(evd_L)), beta=2)))
```

```{code-cell} ipython3
sns.heatmap(ochiai(X))
```

```{code-cell} ipython3
Ttest = sprs.csgraph.minimum_spanning_tree(sprs.dok_array(_sq(Gthres.filled(0)))).todok()

Ttest.nonzero()

# list(zip(*sprs.coo_array(X).nonzero()))
```

```{code-cell} ipython3
row,col = Ttest.nonzero()
Ttest[col,row] = Ttest[row,col]
Ttest.nonzero()
```

```{code-cell} ipython3
# Ttest.keys()
# Ttest[0].values()
# sprs.dok_matrix(X)[0]
# Ttest
from itertools import combinations
completes = list((np.array(list(combinations(i.nonzero()[1],2))).T for i in sprs.coo_matrix(X).todok()))
completes[0]
```

```{code-cell} ipython3
# sprs.coo_array((est_dists[completes[0][0],completes[0][1]], (completes[0][0], completes[0][1])), shape=(5,5)).toarray().round(2)
```

```{code-cell} ipython3
# %timeit _sq(sprs.csgraph.minimum_spanning_tree(_sq(est_dists[completes[0][0],completes[0][1]])).toarray())>0
```

```{code-cell} ipython3
# %timeit _sq(sprs.csgraph.minimum_spanning_tree(_sq(_sq(est_dists)*_sq(np.multiply.outer(X[0], X[0])))))
```

```{code-cell} ipython3
spX = sprs.coo_matrix(X).todok()
x = spX[:100,0]
%timeit (x@x.T)>0

%timeit np.multiply.outer(X[:100,0], X[:100,0])>0

%timeit np.array(list(combinations(x.nonzero()[1],2))).T
# plt.spy(X, aspect=0.01)
# (spX[:1000,0]@spX[:1000,0].T)>0
```

```{code-cell} ipython3
from tqdm import tqdm
from affinis.utils import _outer
def _filter_edgeweight_to_tree(e):
    return _sq(sprs.csgraph.minimum_spanning_tree(_sq(e)).toarray())>0

def better_sfep(X, prior=ochiai, pseudocts=0.5):
    est_dists = -np.log(prior(X, pseudocts=pseudocts))
    big_N = X.shape[1]>=1e3
    if sprs.issparse(X) and (not big_N):
        X_obs = X.toarray() 
    elif big_N:
        X_obs = sprs.coo_matrix(X).todok()
    else: 
        X_obs = X
    # X_obs =  if ( else X.toarray()
    # X_obs = .todok() if  else X
    # idx_gen = tqdm(np.array(list(combinations(x.nonzero()[1],2))).T for x in sprs.coo_matrix(X).todok() if x.sum()>1)
    idx_gen = (sprs.coo_array(est_dists*((x.T@x if big_N else _outer(np.multiply, x)))) for x in X_obs)
    E_obs = sprs.coo_array([_sq((sprs.csgraph.minimum_spanning_tree(a)>0).todense()) for a in idx_gen])
    # row_col = np.concatenate([idx.T[filter_edgeweight_to_tree(est_dists[idx[0],idx[1]])] for idx in idx_gen if idx.shape[1]>1]).T
    # return sprs.coo_array((np.ones(row_col.shape[1]), (row_col[0],row_col[1])), shape=(X.shape[1],X.shape[1]))
    return E_obs
    # row_col = [idx.T[filter_edgeweight_to_tree(est_dists[idx[0],idx[1]])] for idx in idx_gen]
    # row_col = list(idx_gen)
    # row_col = [(idx, est_dists[idx[0],idx[1]]) for idx in idx_gen if idx.shape[1]>1]
    # return row_col
E_obs = better_sfep(X)
# E_obs
```

```{code-cell} ipython3
# (E_obs.T@E_obs)@(spX.T@E_obs).
from affinis.utils import sparse_adj_to_incidence, _norm_diag
from affinis.associations import _spanning_forests_obs_bootstrap
# n1,n2 = np.triu(nx.adjacency_matrix(G).todense()).nonzero()#,_sq(np.triu_indices_from(X.T@X, k=1)[0])
# e = np.ma.nonzero(_sq(nx.adjacency_matrix(G).todense()))[0]
# B = sprs.coo_array((np.concatenate([ones:=np.ones(e.shape[0]),-ones]), (np.concatenate([e,e]),np.concatenate([n1,n2]))), shape=(_sq(nx.adjacency_matrix(G).todense()).shape[0], X.shape[1]))
def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
E_obs = _spanning_forests_obs_bootstrap(X)
n1, n2 = np.triu(_sq(evd_L)).nonzero()
# print(n1.shape)
e = np.ma.nonzero(evd_L)[0]
print(e.shape, n1.shape, n2.shape)
B = sprs.coo_array((np.concatenate([evd_L, -evd_L]), (np.concatenate([e,e]),np.concatenate([n1,n2]))), shape=(e.shape[0], X.shape[1]))

# np.diag((B.T@B).toarray())==np.diag(nx.laplacian_matrix(G).toarray()).round(1)
Xest=(E_obs@(np.abs(B))).toarray()

# ((np.abs(B).T@E_obs.T>0).T.toarray().astype(int)!=X.astype(int)).sum(axis=0)
# _norm_diag((B.T@B).toarray())
```

```{code-cell} ipython3
# _sq(np.diag((E_obs.astype(int).T@E_obs).toarray()))

# _sq(E_obs.sum(axis=0))

# Xest=(E_obs@(B!=0)).astype(int)
sns.histplot(_sq(Xest.T@Xest+1), log_scale=True)
sns.histplot(_sq(X.T@X+1), log_scale=True)
```

```{code-cell} ipython3
f,ax = plt.subplots(ncols=2, figsize=(12,4))
sns.heatmap(Xest[:, np.lexsort(-(Xest>0).astype(int)[::-1])], ax=ax[0])
sns.heatmap(X[:, np.lexsort(-X[::-1])], ax=ax[1])
np.lexsort(-X[::-1])
# np.argsort(Xest.sum(axis=0))
```

```{code-cell} ipython3
# sns.heatmap(ochiai(X))
f,ax = plt.subplots(ncols=2, figsize=(12,4))
sns.heatmap(_sq(_sq(X.T@X)), ax=ax[0], square=True)
sns.heatmap(_sq(_sq(Xest.T@Xest)), ax=ax[1], square=True)
plt.tight_layout()
```

```{code-cell} ipython3
# sns.heatmap(np.cov(Xest))
sns.heatmap(_sq(min_connected_filter(_sq(sinkhorn(_sq(_sq(ochiai(Xest)))))).filled(0)))
# sns.heatmap(_sq(min_connected_filter(_sq(ochiai(Xest))).filled(0)))
```

```{code-cell} ipython3
# spanning_forests_edge_prob(Xest>)
# sns.histplot(np.ma.masked_less_equal(Xest, 0).flatten())

# E_obs
```

```{code-cell} ipython3
f = plt.figure(figsize=(15,15))

# G = nx.from_pandas_adjacency(pd.DataFrame((minmax(sinkhorn(_sq(post_L)))>0.7).astype(int), index=animals.columns, columns=animals.columns))
# G = nx.from_pandas_adjacency(pd.DataFrame(_sq(_sq(top_tree_pct(_sq(post_L), mult=1.5))), index=animals.columns, columns=animals.columns))
Gest = min_connected_filter(_sq(sinkhorn(_sq(_sq(Xest.T@Xest)))))
G = nx.from_pandas_adjacency(pd.DataFrame(_sq(Gest.filled(0)), index=animals.columns, columns=animals.columns))
# Gneg = nx.from_pandas_adjacency(pd.DataFrame(
#     _sq(3*(threshold_edges_filter(_sq(Gvals), 0.9) - Gthres.min()).filled(0) + 0.5*Gthres.filled(0)),
#     index = animals.columns, columns = animals.columns,
# ))
# pos = nx.kamada_kawai_layout(G, dist=pd.DataFrame(1-ochiai(Xest), columns=animals.columns, index=animals.columns).to_dict())
pos = nx.kamada_kawai_layout(G)
# pos = nx.spring_layout(G, iterations=1000, k=2)

nx.draw_networkx(G, pos=pos, node_color='w')
# nx.draw_networkx_edges(G, pos=pos_cos)
# mx.
# nx.draw_networkx_labels(G, pos=pos_cos, bbox = dict(facecolor = "xkcd:cement", edgecolor=None, joinstyle='round'))
# plt.savefig('animals.svg')
# list(G.neighbors('spider'))
nx.connected.is_connected(G)
```

```{code-cell} ipython3
Xfilt = (E_obs.todense()[:,~Gthres.mask]@nx.incidence_matrix(G).T > 0).astype(int)
it_L = spanning_forests_edge_prob(Xfilt)
```

```{code-cell} ipython3
f,ax = plt.subplots(ncols=2, figsize=(12,4))
sns.heatmap(it_L, ax=ax[0])
sns.heatmap(_sq(evd_L), ax=ax[1])
```

```{code-cell} ipython3
from affinis.plots import hinton

hinton(_sq(evd_L))
```

```{code-cell} ipython3
np.linalg.norm((X-(Xest>0)),ord='fro'), np.linalg.norm(X,ord='fro'), np.linalg.norm(Xest>0,ord='fro')
```

```{code-cell} ipython3
import yappi
yappi.clear_stats()
yappi.set_clock_type("wall") # Use set_clock_type("wall") for wall time
yappi.start()
better_sfep(X)
yappi.stop()
yappi.get_func_stats().print_all()
yappi.get_thread_stats().print_all()
```

```{code-cell} ipython3
# %timeit spanning_forests_edge_prob(X)
yappi.clear_stats()
yappi.set_clock_type("wall")
yappi.start()
spanning_forests_edge_prob(X)
yappi.stop()
yappi.get_func_stats().print_all()
yappi.get_thread_stats().print_all()
```

```{code-cell} ipython3
sns.heatmap(_sq((E_obs.sum(axis=0)+0.5)/_sq(X.T@X+1)))
# _sq(E_obs.todense())
```

```{code-cell} ipython3
# mst_post/X.shape[1], 
import seaborn.objects as so
from affinis.utils import edge_mask_to_laplacian
# _sq(coocur_prob(X,pseudocts=0.5))*(E_obs.sum(axis=0)+0.5)/(unroll_node_obs(X).sum(axis=0)+1)
# sns.heatmap(_norm_diag((lambda a: np.diag(a.sum(axis=0))-a)(_sq(mst_post))))
# sns.histplot(np.ma.masked_less( -_sq(_norm_diag((lambda a: np.diag(a.sum(axis=0))-a)(_sq(mst_post)))),0.1))
# sns.histplot(mst_post,discrete=True)
# mst_post

# for scores in [
so.Plot(pd.DataFrame({
    'ochiai': _sq(ochiai(X, pseudocts=0.5)),
    'P_edge': evd_L,
    # mst_post,
    'P_e|m': post_L,
}).melt(), x='value', color='variable'
).add(
    so.Bars(), 
    so.Hist(stat='proportion'), 
    # so.Stack() 
).scale(x='log')
# ]:
    # sns.histplot(scores, stat='density', log_scale=True, element='step', fill=False)
    # sns.histplot(np.ma.masked_less(evd_L, 0.01), stat='density', log_scale=True)
# sns.histplot()
# sns.histplot(np.ma.masked_less(post_L, 0.01), stat='density', log_scale=True)
# sns.histplot(_sq(ochiai(X, pseudocts=0.5)), stat='density', log_scale=True)
# sns.histplot(np.ma.masked_less(_sq(_norm_diag(forest(edge_mask_to_laplacian(np.ma.masked_less(evd_L, 0.1)), beta=1))), 0.1), stat='density')
```

```{code-cell} ipython3
# np.sort(X.shape[0]*post_L)
so.Plot(pd.DataFrame({
    'co-occur': X.shape[0]*_sq(coocur_prob(X, pseudocts=0.5)),
    'ends-in': X.shape[0]*post_L,
    'mst_post': mst_post,
    # 'same-tree': X.shape[0]*_sq(_norm_diag(forest((lambda a: np.diag(a.sum(axis=0))-a)(_sq(evd_L)), beta=1))),
}).melt(), x='value', color='variable'
).add(
    so.Bars(), 
    so.Hist(stat='proportion'), 
    # so.Stack() 
).scale(x='log')

# sns.histplot(_sq(X.T@X), log_scale=True)
# sns.histplot(X.shape[0]*post_L, log_scale=True, fill=False)
# sns.histplot(X.shape[0]*_sq(_norm_diag(forest((lambda a: np.diag(a.sum(axis=0))-a)(_sq(evd_L)), beta=5))))
```

```{code-cell} ipython3
sns.clustermap(pd.DataFrame(_sq(evd_L), columns=animals.columns, index=animals.columns), 
               # mask = _sq(min_connected_filter(evd_L).mask)
              )
# sns.heatmap(_norm_diag(1-forest(edge_mask_to_laplacian(np.ma.masked_less(evd_L, 0.1)), beta=1)))
# sns.heatmap(forest(edge_mask_to_laplacian(np.ma.masked_less(evd_L, 0.1)), beta=1)/np.sqrt(1-_outer(np.multiply, 1-np.diag(forest(edge_mask_to_laplacian(np.ma.masked_less(evd_L, 0.1)), beta=1)))))
```

```{code-cell} ipython3
# pd.DataFrame(_sq(evd_L), columns=animals.columns, index=animals.columns).to_dict()
# pd.Series.to_dict()
```

```{code-cell} ipython3
A=_sq(_sq(X.T@X))

A.argmax(axis=0)
A.max(axis=0)
```

```{code-cell} ipython3
nx.draw_networkx_nodes(
    K:=nx.karate_club_graph(), pos=(testpos:=nx.spring_layout(K)), 
    node_color='white', edgecolors='k', 
    node_size=100, linewidths=0.5,
)
nx.draw_networkx_edges(
    K, pos=testpos,
)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# _sq((threshold_edges_filter(_sq(Gvals), 0.8) - Gthres.min()).filled(0))
nx.kamada_kawai_layout(G)
```

```{code-cell} ipython3
# f.savefig('animals.svg')
from netgraph import Graph, InteractiveGraph
from affinis.filter import threshold_edges_filter
plt.figure(figsize=(10,10))
Graph(
    # _sq(min_connected_filter(_sq(Gvals:=(_sq(evd_L)))).filled(0)),
    Gneg,
    # _sq((threshold_edges_filter(_sq(Gvals), 0.9) - Gthres.min()).filled(0)),
    # node_labels=dict(enumerate(animals.columns)),
    node_labels=True,
    # scale = (3,3),
    node_layout=nx.kamada_kawai_layout(G), 
    # node_layout_kwargs=dict(k=0.1),
    node_size=5, 
    # total_iterations=100,
    # node_labels=True,
    node_label_offset = 0.001, 
    node_edge_color='w',
    # node_shape='o',
    edge_cmap='viridis_r'
    # edge_layout='curved', 
    # edge_layout_kwargs=dict(k=0.05),
    # edge_layout='bundled'

)
plt.show()
```

```{code-cell} ipython3
from pyvis.network import Network
nt = Network('500px','500px')
nt.from_nx(G)
nt.show('nx.html')
```

```{code-cell} ipython3
Graph(
    K, 
    scale=(2,2),
    node_layout="spring", 
    node_layout_kwargs=dict(k=0.5)
)
```

```{code-cell} ipython3
from mpl_toolkits.mplot3d import Axes3D

# The graph to visualize
# G = nx.cycle_graph(20)

# 3d spring layout
pos3 = nx.spring_layout(G, dim=3)
# Extract node and edge positions from the layout
node_xyz = np.array([pos3[v] for v in sorted(G)])
edge_xyz = np.array([(pos3[u], pos3[v]) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


_format_axes(ax)
fig.tight_layout()
# plt.show()
```

```{code-cell} ipython3
import scipy.stats as ss
# ss.rv_histogram(
cts,vals=np.histogram(unroll_node_obs(X).sum(axis=0), density=False, 
# bins=np.arange(unroll_node_obs(X).sum(axis=0).max())
)

plt.plot(cts[1:], robbins_cooc:=((vals[1:-1]+1)*cts[:-1]+0.5)/(cts[1:]+1), ls='', marker='o') #robbins estimate
    # , density=False).pdf(range(1000))

cts,vals=np.histogram(E_obs.sum(axis=0), density=False, 
# bins=np.arange(E_obs.sum(axis=0).max())
)
# plt.plot(cts[1:], robbins_intr:=((vals[1:-1]+1)*cts[:-1]+0.5)/(cts[1:]+1), ls='', marker='o') #robbins estimate
# plt.yscale('log')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from scipy.stats import poisson, nbinom, dirichlet
from scipy.special import lambertw

# _sq(1-poisson(mst_post).pmf(0))/forest(_norm_diag((lambda a: np.diag(a.sum(axis=0))-a)(_sq(mst_post))), beta=10)

# m = X.shape[1]*(E_obs.sum(axis=0)+0.5)/_sq(X.T@X+1)
# m = E_obs.sum(axis=0)/np.ma.masked_less(_sq(X.T@X),1)
# s = m/((E_obs>0).sum(axis=0)/np.ma.masked_less(_sq(X.T@X), 1))
# lam = (lambda s: lambertw(-s*np.exp(-s))+s)(s)
# 1-s/np.real(lam)   
# s, lam

# lambertw(-mst_post*np.exp(-mst_post))+mst_post
# mst_post
# E_obs.mean(axis=0)/E_obs.nonzero().mean(axis=0)
# (E_obs>0).mean(axis=0)
# m/((E_obs>0).sum(axis=0)/np.ma.masked_less(_sq(X.T@X), 1))
# np.real(lam)
# m
E_obs_ma = np.ma.masked_array(E_obs.todense(), mask=~(unroll_node_obs(X).astype(bool)))

bss = poisson(1).rvs((E_obs.shape[0], 1000))
m = (poisson(1).rvs((E_obs.shape[0], 1000)).sum(axis=1)*E_obs_ma.T).T.mean(axis=0)/1000
s = m/(m>0).sum(axis=0)
lam = (lambda s: lambertw(-s*np.exp(-s))+s)(s)
# 1-s/np.real(lam)   
# sns.histplot(np.real(lam))
sns.histplot(unroll_node_obs(X).sum(axis=0)/X.shape[0], stat='density')
sns.histplot(E_obs.sum(axis=0)[E_obs.sum(axis=0)>5]/X.shape[0], stat='density')
# plt.plot(poisson(unroll_node_obs(X).sum(axis=0).mean()/X.shape[0]).pmf(np.arange(200)))
# plt.plot(nbinom(7, 0.5).pmf(np.arange(50)), color='r')
# sns.histplot(E_obs.sum(axis=0)[E_obs.sum(axis=0)>0], stat='probability', discrete=True)
# lam

# sns.histplot(nbinom(4, 0.1).mean(axis=0)>0], stat='probability', discrete=True)
# lam

nbinom(7, 0.5).mean(), unroll_node_obs(X).sum(axis=0).mean()/X.shape[0]
# m
# sns.histplot((E_obs.sum(axis=0)+0.5)/(unroll_node_obs(X).sum(axis=0)+1), stat='density')
# (unroll_node_obs(X).astype(bool)).shape, E_obs.shape
# plt.yscale('log')
```

```{code-cell} ipython3
# sns.histplot(_sq(ochiai(X.shape[0]*dirichlet(4*np.ones(X.shape[0])).rvs().T*X))
ochiai(rng.choice(X, axis=0, size=1000, p=dirichlet(4*np.ones(X.shape[0])).rvs()[0]), pseudocts=0.5)
# dirichlet(4*np.ones(X.shape[0])).rvs()[0]

# ochiai(X)
```

```{code-cell} ipython3
for i in range(100):
    plt.plot(np.sort(_sq(
        ochiai(rng.choice(X, axis=0, size=1000, p=dirichlet(4*np.ones(X.shape[0])).rvs()[0]), pseudocts=0.5)
        ))[-60:], 
        marker='_', ls='', color='k', alpha=0.1
    )
plt.plot(np.sort(_sq(ochiai(X, pseudocts=0.5)))[-60:], marker='_', ls='', color='r')
```

```{code-cell} ipython3
# from affinis.plots import hinton
from affinis.utils import minmax
from affinis.associations import _contingency_prob

a,b,c,d = _contingency_prob(X, pseudocts=0.5)

d/b, d/c, a.shape
# sns.heatmap(1/_outer(np.add, 1/X.sum(axis=0)))
# sns.heatmap(sinkhorn(_sq(post_L)))
# sns.histplot(post_L)
# sns.histplot(minmax(_sq(sinkhorn(_sq(post_L)))))
sns.heatmap((1/_outer(np.add, 1/X.sum(axis=0)))/(np.sqrt(_outer(np.multiply, X.sum(axis=0)))/2))
```

```{code-cell} ipython3
sns.heatmap(np.sqrt(_outer(np.multiply, X.sum(axis=0)))/2)
```

```{code-cell} ipython3
# sns.histplot(post_L, stat='density')
# sns.histplot(np.where(mst_post>0.1, mst_post, np.nan), stat='density')
# _sq(mst_post).shape
# plt.xscale('log')
# mst_post
```

```{code-cell} ipython3
sns.histplot(post_L/_sq(ochiai(X, pseudocts=0.5)), log_scale=True)
```

```{code-cell} ipython3
# matplotlib.use('nbagg')

from netgraph import Graph, InteractiveGraph, EditableGraph
# %matplotlib widget
plt.figure(figsize=(15,15))

gplot=InteractiveGraph(
    _sq(min_connected_filter(_sq(Gvals)).filled(0)), 
    node_labels=True, 
    node_layout='dot', 
    # node_label_offset=0.05,
    node_size=2,
    directed=False,
)
plt.show()
# dict(zip(map(tuple, np.vstack(np.triu_indices_from(Gvals)).T.tolist()), 1-min_connected_filter(_sq(Gvals)).filled(0)))

# generalized_graph_dists(edge_mask_to_laplacian(min_connected_filter(_sq(Gvals))))
```

```{code-cell} ipython3
# sns.displot(mst_post)
# _sq(mst_post>0.1)
plt.figure(figsize=(15,15))

# G = nx.from_pandas_adjacency(pd.DataFrame((minmax(sinkhorn(_sq(post_L)))>0.7).astype(int), index=animals.columns, columns=animals.columns))
# G = nx.from_pandas_adjacency(pd.DataFrame(_sq(_sq(top_tree_pct(_sq(post_L), mult=1.5))), index=animals.columns, columns=animals.columns))

G = nx.from_pandas_adjacency(pd.DataFrame(_sq(~min_connected_filter(_sq(Gvals:=(_sq(post_L)))).mask), index=animals.columns, columns=animals.columns))
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos=pos, node_color='w')
# nx.draw_networkx_edges(G, pos=pos, edge_cmap='viridis')
```

```{code-cell} ipython3
from affinis.utils import _outer
def unroll_node_obs(X): 
    trirow, tricol = np.triu_indices(n=X.shape[1],k=1)
    return np.einsum('ij,ik->ijk', X, X)[:,trirow, tricol]
    
np.ma.array(np.multiply.outer(np.ones(roll_X.shape[0]), _sq(ochiai(roll_X))), mask=unroll_node_obs(~roll_X.astype(bool)))
# x_mask= np.ma.masked_less(X[0], 1e-3)
# x_umsk= X[0]
# %timeit _outer(np.multiply,x_mask.compressed())
# %timeit _outer(np.multiply, x_umsk)
# x_mask.

# unroll_node_obs(roll_X)[0].sum()
# np.ma.masked_array(X.astype(bool))
```

```{code-cell} ipython3
from scipy.sparse.csgraph import minimum_spanning_tree
plt.figure(figsize=(15,15))
Gtree = nx.from_pandas_adjacency(pd.DataFrame(minimum_spanning_tree(-Gvals).todense()<0, index=animals.columns, columns=animals.columns))
pos = nx.spring_layout(Gtree)
nx.draw_networkx(Gtree, pos=pos, node_color='w', edge_color='r')
nx.draw_networkx_edges(nx.difference(G, Gtree), 
    pos=pos, edge_color='g', alpha=0.8,
    connectionstyle="arc3,rad=0.01"  # <-- THIS IS IT
)
# nx.difference()
```

```{code-cell} ipython3
from toolz import sliding_window
# df.query('category=="animals"').groupby(['listnum','item'])['item'].rolling(2).count()
# animals.rolling(2).sum()
(df
 .reset_index(level=1, drop=True)
 .set_index(
    df
    .groupby(level=0)
    .cumcount()
    .rename('tokenid'), 
    append=True
    )['item']
#  .groupby(['item'])['item']
 .rolling(2)
 .apply(str.join(', '))
#  .count()
#  .rolling(2)
#  .count()
#  .unique()
)
```

```{code-cell} ipython3
# ochiai(roll_X)
np.cross(roll_X, roll_X)
```
