import networkx as nx
import numpy as np
import sparse

def generate_bipartite_matrix(n_cols, n_rows, p=0.25, seed=2):
    """
    Generates a binary feature matrix from a random bipartite graph.
    
    Args:
        n_cols: Number of features (columns)
        n_rows: Number of observations (rows)
        p: Probability of edge creation
        seed: Random seed for reproducibility
        
    Returns:
        A sparse.COO matrix representing the biadjacency matrix.
    """
    B = nx.bipartite.random_graph(n_cols, n_rows, p, seed=seed)
    # Nodes in the first set are 0 to n_cols-1
    # Nodes in the second set are n_cols to n_cols + n_rows - 1
    nodes_second_set = list(B.nodes)[n_cols:]
    
    # biadjacency_matrix returns a scipy.sparse.coo_matrix
    adj = nx.bipartite.biadjacency_matrix(B, nodes_second_set)
    
    # Ensure it's a sparse.COO and cast to int to avoid operand issues
    return sparse.COO.from_scipy_sparse(adj).astype(int)

def get_benchmark_data(scale="small"):
    """
    Provides standardized matrices for benchmarking.
    """
    configs = {
        "small": (15, 30),
        "medium": (500, 1000),
        "large": (1000, 10000),
    }
    
    if scale not in configs:
        raise ValueError(f"Scale must be one of {list(configs.keys())}")
        
    n_cols, n_rows = configs[scale]
    return generate_bipartite_matrix(n_cols, n_rows)
