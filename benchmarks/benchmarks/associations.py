import os
import sparse
import affinis.associations as aff

class AssociationsBench:
    filename = 'large_matrix.npz'

    def setup_cache(self):
        import networkx as nx
        n_cols=1000
        n_rows=15000
        B = nx.bipartite.random_graph(n_cols,n_rows, .0005, seed=2)
        n = list(B.nodes)[n_cols:]
        X = sparse.COO.from_scipy_sparse(
            nx.bipartite.biadjacency_matrix(B, n)
        ).astype(bool)
        sparse.save_npz(self.filename, X)
        

    def setup(self):
        # Use absolute path to avoid asv temp dir issues
        # path = os.path.join(os.path.dirname(__file__), 'large_matrix.npz')
        self.X = sparse.load_npz(self.filename).to_scipy_sparse()

    def time_coocur_prob(self, *args):
        return aff.coocur_prob(self.X)

    def time_ochiai(self, *args):
        return aff.ochiai(self.X)

    def time_odds_ratio(self, *args):
        return aff.odds_ratio(self.X)

    def time_mutual_information(self, *args):
        return aff.mutual_information(self.X)

    def time_yule_y(self, *args):
        return aff.yule_y(self.X)

    def time_yule_q(self, *args):
        return aff.yule_q(self.X)

    def time_hyperbolic_project(self, *args):
        return aff.hyperbolic_project(self.X)

    def time_resource_project(self, *args):
        return aff.resource_project(self.X)

    def time_high_salience_skeleton(self, *args):
        return aff.high_salience_skeleton(self.X)

    def time_doubly_stochastic_filter(self, *args):
        return aff.doubly_stochastic_filter(self.X)

    def time_forest_pursuit(self, *args):
        return aff.forest_pursuit(self.X)
