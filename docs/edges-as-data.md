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

# Exploratory Graphoid Analysis

> Toward Principled Graphical Model Recovery

## Modeling Edges...as Data?

```{code-cell} ipython3
import networkx as nx
import pandas as pd
from scipy import sparse, linalg, stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```{code-cell} ipython3
rng = np.random.default_rng(seed=2)
```

First, let's model a common "social network analysis" task: co-citation networks. 

1. What is our **goal**? 
    > we want to see how authors are "connected" to each other, based on how they write together. 
2. How does our data get created? 
    > Based on how "connected" authors are, they will be more likely to write papers together. Let's say a friendship _causes_ one author to have a higher probability of co-authoring a given paper, if their friend is an author. 
    >
    >We directly observe the resulting indicators (authors co-authored a given paper, for each paper in our dataset). 
3. How is our goal related to our data?
    > Note that we now have an _inverse problem_: what set of friendships would lead to a set of authorships that we observe?  

+++

### Synthesize the Data    

Let's directly create a "relaistic" friendship graph.
A reasonable limitation is that people are either pairwise friends or part of a friend "group". 
In other words, if you start with 3 friends, and two of those friends know each other and they both a 4th person, you would also know that 4th person. 

> This is the set of single-vertex-separator graphs, block-graphs, or "clique" unions. 

It happens we can exactly represent the set of block graphs as the line graphs of trees. 
> Both of these classes of graphs are important for the Hammersly-Clifford Theorem, which will become critical, later. 

```{code-cell} ipython3
n_authors=20
author_idx = pd.CategoricalIndex((f'author_{i}' for i in range(1,n_authors+1)))

friendships = nx.random_tree(n_authors+1, seed=2) # real events... what "happens" as evidence of a relationship
author_rel = nx.relabel.relabel_nodes(nx.convert_node_labels_to_integers(nx.line_graph(friendships)),dict(zip(range(n_authors),author_idx.categories.tolist())))  # inferred structure
# author_rel.
```

```{code-cell} ipython3
f = plt.figure(figsize=(5,4)).patch.set_alpha(0.)
nx.draw(author_rel, pos=nx.layout.kamada_kawai_layout(author_rel), 
        node_color='xkcd:puce', edge_color='grey')
plt.title('Author Friendships', color='grey')
```

### What have we observed? 

This is an Incidence Structure, where a friendship is a set (of cardinality 2), and an author is a line. 
Note that the incidence representation of the data is a matrix where the features are 'nodes' and the observations are 'edges'. 

```{code-cell} ipython3
# nx.incidence_matrix(author_rel)
incidence = (
    pd.Series.sparse.from_coo(nx.incidence_matrix(author_rel, oriented=True).tocoo())#.set_index(author_idxlevel=0)%%!
    .pipe(lambda df: df.set_axis(df.index.set_levels(author_idx, level=0).swaplevel()))
    .sort_index()
    .astype('Sparse[int]')
)
incidence.head()
# sparse.csc_array.to
```

```{code-cell} ipython3
B = incidence.unstack().fillna(0.)
B.head()
```

This is different from the adjacency representation, where the sparse representation comes from an "edgelist" view:

```{code-cell} ipython3
edgelist = (
    nx.to_pandas_edgelist(author_rel)
    .astype(dict(source=author_idx.dtype, target=author_idx.dtype))
)
edgelist.head()
```

However, the two are _related_. 

The Laplacian is defined as the grammian of the _oriented_ incidence matrix: 

```{code-cell} ipython3
L = B.T@B
# L = incidence.sparse.to_coo()[0].T@incidence.sparse.to_coo()[0]
sns.heatmap(L, mask=L==0)
```

### Hang on, the Grammian? That sounds familiar...

Indeed, from an edge-observational point of vieew, the Laplacian would just be the unscaled scatter matrix of our sample data: 

$$S = \frac{1}{N-1} X^TX = \frac{1}{N-1}L$$

In turn, our data covariance would be the _centered_ version of our data. 
> NOTE that for _undirected graphs_, we are implicitly observing two copies of each "oriented edge" (one for each direction). So, we can easily show that the centered data matrix
> $$B_c = \begin{bmatrix} B \\-B\end{bmatrix}$$
> should have exactly $L_c = 2 L = (N-1)\frac{1}{2}S$

Thus undirected Laplacians are the unscaled covariance matrices of our edge observations, and we can be on our merry way using graphs and their laplacians to analyze our social network...right? 

+++

### Some problems: what's in an edge? 

The Laplacian is typically used as a continuous-time analog for dynamic systems. 
For instance, it's the discrete _Laplace Operator_, a green's function for discrete Poisson equation: 
$$\frac{d x}{d t} = -kLx$$

It's also called the _Kirchoff Matrix_, due to its relation with the flow of current in an electrical network. It encodes the flow of energy from each node to the nodes it is connected to. 
> This explains the "oriented" part! Every edge means a unit of energy can flow "from" one node "to" another, according to the conductance between them!

+++

But.... This has some major implications for our original model. Remember: 
> we want to connect authors if they have a friendship that corresponds to increased conditional probability of co-authorship. 

Nothing is "flowing". Our observation of authorship doesn't take away some amount of authorship from their friend at some subsequent timestep. 

> These kinds of issues are _everywhere_ in network analysis... ask yourself: **what do our edges measure?**

+++

Instead of a friendship implying _negative correlation_ (see the negatives in all off-diagonals in $L$?), we probably want our friendships to indicate positive correlations. 

Do this by using the unsigned laplacian $L_S$, which we can get from the _unoriented_ incidence matrix. 

```{code-cell} ipython3
Ls = np.abs(B).T@np.abs(B)  # signless Laplacian
sns.heatmap(Ls, mask=Ls==0)
# sns.heatmap(np.linalg.inv(Ls), mask=np.abs(np.linalg.inv(Ls))<=0.1)
# sns.heatmap(np.linalg.pinv(2*L/(B.shape[0]-1)))
# sns.heatmap(np.linalg.inv(np.eye(n_authors)+0.05*L).dot(L))
```

Better, but recall that for our statistical modeling desires, empirical covariance must be centered (which only _happend_ to work out for us in the _oriented_ case. 

What does our graphical scatter look like now? 

```{code-cell} ipython3
M = np.abs(B).cov()
sns.heatmap(M, mask=M==0)
```

Something is weird. 

There are negative values? 
Folks that weren't friends are now friends, and folks that didnt know each other at all... kinda oppose each other? 

> If you're not with me... _you're my enemy_. 

(Thanks a lot, ~~Anakin~~Darth Vader)

Indeed, the sample covariance matrix is a _biased_ intrinsic estimator. 
Estimating the true covariance from data is the problem of _Covariance Estimation_, which usually involves sparsity constraints and optimization of the determinant (Graphical Lasso). 

+++

This is all _before we've even gotten to our actual observational model_, which will further obscure what exactly should and should not be real edges. 

Remember, what we want to _recover_ is a graph like the one at the top, where we see "real" friendships. even if our data is a set of observations of _only friends, exclusively, co-authoring a single paper per mutual friendship_, we would still be way off in determining friendships due to the bias inherent in the signless laplacian, and the normal laplacian is not at all going to model what we are after in the first place. 

> We need to recover the true covariance struture with better techniques than the co-occurrence laplacian, even if our data is _perfect_. 

And, trust me, it most certainly is _not_. 

+++

## Modeling Observation Data as...Edges?

Let's use our gaussian probabilistic graphical model (it's a Markov Random Field) to generate some fake data. 
For each week in the next, say, 4 or so years, let's ask our set of vaugely-associated authors whether the wrote a paper or not. 

```{code-cell} ipython3
papers = pd.DataFrame(
    np.where(rng.multivariate_normal(np.zeros(n_authors), M, size=200)>0.3, 1,0),
    columns=author_idx
).rename_axis('week')
papers.melt(ignore_index=False).replace(0,np.nan).dropna()
```

```{code-cell} ipython3
# Scatter Matrix
sns.heatmap(papers.T@papers)
plt.title('co-occurrence counts')
```

```{code-cell} ipython3
# Now the Covariance
sns.heatmap(papers.cov())
plt.title('covariance')
```

Ok, what's going on...??

Recall that, before, we counted edges as observations. 
Edges were indeed in the node-feature-space $\mathbb{R}^n$, but they only ever had a source and a target (marginal sum was 2). 
Now we are allowing much bigger "edges". 

In fact, this observational data is still an incidence structure, but it is a _hypergraph_. 
Hypergraphs have 1-to-1 corresponcence with _bipartite graphs_, and the _biadjacency_ matrix is the matrix we just saw.
By taking the _inner product_ of all pairs of edges in this bipartite graph, we have now performed what's called a _linear bipartite projection_ of the hypergraph into a graph on only authors. 

Linear projections make the (rather presumptuous) assertion that all hyperedges/sets are equivalent to linear combinations of their subsets --- specifically, a union of all pairwise relations. 

```{code-cell} ipython3
obs_incidence = (
    papers.melt(ignore_index=False, var_name='author')
    .reset_index()
    .astype({'author':author_idx.dtype, 'value':'Sparse[int]'})
    .set_index(['week','author'])
)['value']

obs_incidence
```

```{code-cell} ipython3
sns.heatmap(obs_incidence.unstack().T@obs_incidence.unstack())
plt.title('co-occurrence counts')
```

```{code-cell} ipython3
sns.heatmap(obs_incidence.unstack().cov())
plt.title('covariance')
```

Co-Occurrence Covariance and our "unrolled" edges' grammian: the two are _the same_

```{code-cell} ipython3
(papers.cov()-obs_incidence.unstack().cov()).max() # to machine precision
```

### An Aside: Modeling Complex Systems --- Torres et al (2021)

A more precise description of what's going on here is that our observations _are_ hyperedges. 
However, to simply calculate an empirical covariance (which is a centered/scaled version of the co-occurrence Laplacian), whe are implicitly forming a downward-closure of our hyperedges, turning them into **$k$-simplices**. 

+++

Now recall that a hyperedge (publication, in our data) with $k$ authors being turned into a $k$-simplex makes it identical to another simplex on the same authors. 
Think of this as "forgetting" that documents are independent observations of relationships in the hyperedge. 
The inner product will treat any co-occurrence in any document as equivalent to another co-occurrence anywhere else. 

> We have implicitly removed conditional independence from our model

+++

Then, to get to a graph again, we assume that each inner-product can stand in for its own, pairwise relationship... its own edge weight. 
So, it's impossible to tell if a given projected edge came about because of participation in one _huge_ paper, a bunch of smaller papers, or maybe a mixture. 
This is implied by the inner product itself, due to associativity of addition/multiplication. 

> We have implicitly removed higher-order interractions from our model 

+++

### I'm Lost in the weeds

One way of building intuition on this is to ask yourself: Is it more likely, given a paper's list of authors ...?
> for all authors to have been equally friends with every other author, a priori

or

> for authors to have been asked by a single friend, with the ability to ask one other friend. 

Both are probably a _stretch_, with respect to how papers are really going to get written (there's probably a lot more drama!). 
But it should be clear that, as we move away from the $n=2$ case (where the two options are equivalent), to higher numbers of co-authors, the situation where you know everyone else equally _under our conditional probability definition of friendship_ becomes far more unlikely that not. 


+++

### Sounds good, but _why_?
Why is it less likely? 
After all, isn't a model where every pair is assigned equal probability an... uninformative prior, of sorts? There's fewer decisions I, as an analyst must make (read: fewer parameters), so shouldn't the complexity be lower, somehow? 

It depends on your prior, much like our deference to sparse models (even if they need tuning). 
If we believe a priori that, say, any two authors in the set are some probability $p<1$ to be friends, then the chance of observing the mutual friendship of $k$ authors is distributed as a binomial distribution on ${k \choose 2}$ trials (one for each edge).  

Even if every author has equal odds of knowing any other author (which should ring massive alarms, given our block-graph model of the world), then the prior likelihood of a clique should be


```{code-cell} ipython3
from scipy.stats import binom
plt.stem(x:=np.arange(2,11), binom.pmf(1., x,0.5))#.pmf(np.arange(10))
plt.gca().set(xlabel='# authors', ylabel='clique prior probability ($p=0.5$)')
```

In reality $p$ should be a lot lower, and _certainly_ infuenced by our domain knowledge. 

+++

### TODO
Stop here...

```{code-cell} ipython3
Ds_inv=np.diag(1./np.diag(Ls))
Σ = stats.Covariance.from_precision(cal_L:=linalg.sqrtm(Ds_inv)@Ls@linalg.sqrtm(Ds_inv))
sns.heatmap(cal_L)
```

```{code-cell} ipython3
sns.heatmap(Σ.covariance)
```

```{code-cell} ipython3
stats.multivariate_normal(cov=Σ, seed=rng).rvs(400)

# eps=0.01*np.eye(n_authors)

# sns.heatmap(sparse.linalg.inv(sparse.csc_array(L+eps)).todense())
```

```{code-cell} ipython3

# Create sparse, symmetric PSD matrix S
A = rng.standard_normal(size=(n_authors, n_authors))  # Unit normal gaussian distribution.
A[sparse.rand(n_authors, n_authors, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
Strue = A.dot(A.T) + 0.05 * np.eye(n_authors)  # Force strict pos. def.

# Create the covariance matrix associated with S.
R = np.linalg.inv(Strue)

# Create samples y_i from the distribution with covariance R.
y_sample = linalg.sqrtm(R).dot(rng.standard_normal((n_authors, 30)))

# Calculate the sample covariance matrix.
Y = np.cov(y_sample)
```

```{code-cell} ipython3
sns.heatmap(Strue, mask=Strue==0)
```

```{code-cell} ipython3
sns.heatmap(R, mask=R<=0.05)
```

```{code-cell} ipython3
sns.heatmap(np.linalg.inv(R), mask=np.linalg.inv(R)<=0.05)
```

```{code-cell} ipython3

```
