import numpy as np
from cns.midend.graph_gen import subgraph, gen_dense_connect_edges, GraphGenerator


class GraphGeneratorNoCluster(GraphGenerator):
    def __init__(self, target_points):
        self.num_points = len(target_points)
        self.target_points = target_points

        inc_idx: np.ndarray = np.arange(self.num_points)
        self.l0_dense_edge_index: np.ndarray = np.stack([inc_idx, inc_idx], axis=0)
        self.l1_dense_edge_index = gen_dense_connect_edges(inc_idx.copy())
        self.l0_to_l1_edge_index = self.l0_dense_edge_index.copy()

        self.cluster_centers_index = inc_idx.copy()
        self.belong_cluster_index = inc_idx.copy()

    def get_graph(self, missing_node_indices=None):
        l0_dense_edge_index = self.l0_dense_edge_index
        l1_dense_edge_index = self.l1_dense_edge_index
        l0_to_l1_edge_index = self.l0_to_l1_edge_index

        num_nodes = len(self.target_points)
        num_clusters = num_nodes
        if (missing_node_indices is not None) and len(missing_node_indices):
            l0_dense_edge_index, node_mask, _ = subgraph(l0_dense_edge_index, missing_node_indices, on="both", num_nodes=num_nodes)
            l0_to_l1_edge_index, _, _ = subgraph(l0_to_l1_edge_index, missing_node_indices, on="both", num_nodes=num_clusters)
            l1_dense_edge_index, cluster_mask, _ = subgraph(l1_dense_edge_index, missing_node_indices, on="both", num_nodes=num_clusters)
        else:
            node_mask = np.ones(num_nodes, dtype=bool)
            cluster_mask = np.ones(num_clusters, dtype=bool)

        return (
            l0_dense_edge_index, l1_dense_edge_index, l0_to_l1_edge_index, 
            node_mask, cluster_mask
        )

    @property
    def num_clusters(self):
        return self.num_points


if __name__ == "__main__":
    gg = GraphGeneratorNoCluster(np.random.randn(5, 2))
    data = gg.get_data(np.random.randn(5, 2))
    print(data)
    print(data.x_cur)

