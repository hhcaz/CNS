import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.cluster import AffinityPropagation, KMeans


def gen_dense_connect_edges(node_indices):
    """Create NxN edges on N nodes.

    Arguments:
    - node_indices: shape = (N,)

    Returns:
    - edge_index: shape = (2, NxN)
    """
    num_nodes = len(node_indices)
    node_indices = np.asarray(node_indices, dtype=np.int64)
    edge_index_j = np.repeat(node_indices, num_nodes)
    edge_index_i = np.tile(node_indices, num_nodes)
    edge_index = np.stack([edge_index_j, edge_index_i], axis=0)
    return edge_index


def subgraph(edge_index, remove_node_indices, on="j", num_nodes=None):
    """
    Arguments:
    - edge_index: shape = (2, E0), [edge_index_j, edge_index_i]
    - remove_node_indices: list or array of index of nodes to remove
    - on: "j", crop on edge_index_j; "i", crop on edge_index_i; "both", on both

    Returns:
    - edge_index: shape = (2, E1), masked edge_index where connection contains the remove nodes are cropped
    - node_mask: shape = (num_nodes,), node mask to filter nodes
    - edge_mask: shape = (num_nodes,), edge mask to filter edges
    """ 
    on = on.strip().lower()
    assert on in ["j", "i", "both"]

    if num_nodes is None:
        num_nodes = 1
    
    if edge_index.size == 0:
        node_mask = np.ones(num_nodes, dtype=bool)
        edge_mask = np.ones(0, dtype=bool)
        return edge_index, node_mask, edge_mask

    if on == "j":
        num_nodes = max(edge_index[0].max() + 1, num_nodes)
        node_mask = np.ones(num_nodes, dtype=bool)
        node_mask[remove_node_indices] = False
        edge_mask = node_mask[edge_index[0]]
    elif on == "i":
        num_nodes = max(edge_index[1].max() + 1, num_nodes)
        node_mask = np.ones(num_nodes, dtype=bool)
        node_mask[remove_node_indices] = False
        edge_mask = node_mask[edge_index[1]]
    else:
        num_nodes = max(edge_index.max() + 1, num_nodes)
        node_mask = np.ones(num_nodes, dtype=bool)
        node_mask[remove_node_indices] = False
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    
    edge_index = edge_index[:, edge_mask]

    return edge_index, node_mask, edge_mask


class GraphData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, **kwargs):

        if "new_scene" not in kwargs.keys():
            kwargs.update({"new_scene": torch.tensor(False).bool()})
        if "tPo_norm" not in kwargs.keys():
            kwargs.update({"tPo_norm": torch.tensor(1.0).float()})

        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key: str, value, *args, **kwargs):
        for candidate in [
            "l0_dense_edge_index", 
            "l0_to_l1_edge_index_j",
            "cluster_centers_index",
        ]:
            if candidate in key:
                return self.num_nodes
        
        for candidate in [
            "l1_dense_edge_index", 
            "l0_to_l1_edge_index_i", 
            "belong_cluster_index"
        ]:
            if candidate in key:
                return self.num_clusters
        
        return super().__inc__(key, value, *args, **kwargs)
    
    def start_new_scene(self, is_new=True):
        setattr(self, "new_scene", torch.tensor(is_new).bool())
    
    def set_distance_scale(self, dist_scale):
        setattr(self, "tPo_norm", torch.tensor(dist_scale).float())


def cluster_points(points):
    model = AffinityPropagation(damping=0.9)
    yhat = model.fit_predict(points)
    cluster_ids, cluster_points_count = np.unique(yhat, return_counts=True)

    if len(cluster_ids) < 4:
        model = KMeans(n_clusters=4)
        yhat = model.fit_predict(points)
        cluster_ids, cluster_points_count = np.unique(yhat, return_counts=True)
    return yhat, cluster_ids, cluster_points_count


def try_cluster_points(points, max_tries=1):
    for i in range(max_tries):
        try:
            ret = cluster_points(points)
        except Exception as e:
            print("[ERR ] Error encountered in cluster points:")
            print(e)
            
            if i >= max_tries - 1:
                raise e
            else:
                print("[INFO] Left tries: {}/{}".format(max_tries-i-1, max_tries))
        else:
            break
    return ret


class GraphGenerator(object):
    def __init__(self, target_points):
        self.num_points = len(target_points)
        self.target_points = target_points

        if self.num_points > 16:
            self.yhat, self.cluster_ids, self.cluster_points_count = \
                try_cluster_points(target_points, max_tries=3)
        else:
            self.yhat = np.arange(self.num_points)
            self.cluster_ids = self.yhat.copy()
            self.cluster_points_count = np.ones_like(self.cluster_ids)
        
        self.cluster_centers_index = []  # record index of each cluster's center point
        self.belong_cluster_index = np.zeros(len(self.target_points), dtype=np.int64)
        self.cluster_points_index = []  # record points indices of each cluster

        l0_dense_edge_index = []  # increment by num_points
        # l1_dense_edge_index = []  # increment by num_clusters
        l0_to_l1_edge_index_j = []  # increment by num_points
        l0_to_l1_edge_index_i = []  # increment by num_clusters
        to_center = np.zeros_like(target_points)

        for i, c_id in enumerate(self.cluster_ids):
            mask = (self.yhat == c_id)
            cluster_points = target_points[mask]
            cluster_points_index = np.nonzero(mask)[0]
            cluster_median_pos = np.median(cluster_points, axis=0)

            dist2median = np.linalg.norm(cluster_points - cluster_median_pos, axis=-1)
            min_dist_idx = np.argmin(dist2median)
            cluster_center_idx = cluster_points_index[min_dist_idx]

            l0_dense_edge_index.append(gen_dense_connect_edges(cluster_points_index))
            l0_to_l1_edge_index_j.append(cluster_points_index)
            l0_to_l1_edge_index_i.append(np.ones_like(cluster_points_index) * i)
            to_center[cluster_points_index] = target_points[cluster_center_idx] - target_points[cluster_points_index]
            self.belong_cluster_index[cluster_points_index] = i

            self.cluster_centers_index.append(cluster_center_idx)
            self.cluster_points_index.append(cluster_points_index)
        
        l0_to_l1_edge_index_j = np.concatenate(l0_to_l1_edge_index_j)
        l0_to_l1_edge_index_i = np.concatenate(l0_to_l1_edge_index_i)
        
        self.cluster_centers_index = np.asarray(self.cluster_centers_index, dtype=np.int64)
        self.l0_dense_edge_index = np.concatenate(l0_dense_edge_index, axis=-1)
        self.l1_dense_edge_index = gen_dense_connect_edges(np.arange(len(self.cluster_ids)))
        self.l0_to_l1_edge_index = np.stack([l0_to_l1_edge_index_j, 
                                             l0_to_l1_edge_index_i], axis=0)
        self.to_center = to_center

        # Example:
        # query node index = 5
        # node2j_index: 3 4 2 7 0 [1] 9 8 6 5  // find the element in index 5, value is 1
        # l0_to_l1_edge_index_j: 4 [5] 2 0 1 9 8 3 7 6  // find the element in index 1, value is 5, 
        #                                                  which is equal to the value of query node index
        # l0_to_l1_edge_index_i: 0 [0] 0 1 1 1 1 2 2 2  // find the element in index 1, value is 0, 
        #                                                  which is the corresponding cluster index of query node
        # cluster index is 0
        self.node2j_index = np.empty_like(l0_to_l1_edge_index_j)
        self.node2j_index[l0_to_l1_edge_index_j] = np.arange(len(l0_to_l1_edge_index_j))
        # l0_to_l1_edge_index_j_index = self.node2j_index[node_index]
        # cluster_index = l0_to_l1_edge_index_i[l0_to_l1_edge_index_j_index]

    
    def _get_graph(self, missing_node_indices=None):
        l0_dense_edge_index = self.l0_dense_edge_index
        l1_dense_edge_index = self.l1_dense_edge_index
        l0_to_l1_edge_index = self.l0_to_l1_edge_index

        num_nodes = len(self.target_points)
        num_clusters = len(self.cluster_ids)

        if (missing_node_indices is not None) and len(missing_node_indices):
            l0_dense_edge_index, node_mask, _ = subgraph(l0_dense_edge_index, missing_node_indices, on="both", num_nodes=num_nodes)
            l0_to_l1_edge_index, _, _ = subgraph(l0_to_l1_edge_index, missing_node_indices, on="j", num_nodes=num_clusters)

            cluster_indices = self.l0_to_l1_edge_index[1][self.node2j_index[missing_node_indices]]
            cluster_points_count = self.cluster_points_count.copy()
            np.add.at(cluster_points_count, cluster_indices, -1)
            missing_cluster_indices = np.nonzero(cluster_points_count == 0)[0]
            
            if len(missing_cluster_indices):
                l1_dense_edge_index, cluster_mask, _ = subgraph(l1_dense_edge_index, missing_cluster_indices, on="both", num_nodes=num_clusters)
                l0_to_l1_edge_index, _, _ = subgraph(l0_to_l1_edge_index, missing_cluster_indices, on="i", num_nodes=num_clusters)
            else:
                cluster_mask = np.ones(num_clusters, dtype=bool)
        else:
            node_mask = np.ones(num_nodes, dtype=bool)
            cluster_mask = np.ones(num_clusters, dtype=bool)
        
        return (
            l0_dense_edge_index, l1_dense_edge_index, l0_to_l1_edge_index, 
            node_mask, cluster_mask
        )


    @property
    def num_clusters(self):
        return len(self.cluster_ids)


    def get_data(
        self, 
        current_points,
        missing_node_indices=None,
        current_features=None, 
        target_features=None, 
        mismatch_mask=None,
    ) -> GraphData:
        if current_features is None:
            # current_features = current_points
            current_features = self.default_feature(current_points)
        if target_features is None:
            # target_features = self.target_points
            target_features = self.default_feature(self.target_points)
        
        l0_dense_edge_index, l1_dense_edge_index, l0_to_l1_edge_index, \
            node_mask, cluster_mask = self._get_graph(missing_node_indices)
        
        if mismatch_mask is None:
            mismatch_mask = np.zeros_like(node_mask)  # bool
        
        data = GraphData(
            num_nodes=self.num_points,
            num_clusters=torch.tensor(self.num_clusters).long(),

            x_cur=torch.from_numpy(current_features).float(),                           # (N, num_x_feats)
            pos_cur=torch.from_numpy(current_points).float(),                           # (N, num_pos_feats)
            l1_dense_edge_index_cur=torch.from_numpy(l1_dense_edge_index).long(),       # (2, E11), inc by num_nodes
            l0_to_l1_edge_index_j_cur=torch.from_numpy(l0_to_l1_edge_index[0]).long(),  # (E01,), inc by num_nodes
            l0_to_l1_edge_index_i_cur=torch.from_numpy(l0_to_l1_edge_index[1]).long(),  # (E01,), inc by num_clusters

            x_tar=torch.from_numpy(target_features).float(),
            pos_tar=torch.from_numpy(self.target_points).float(),
            l1_dense_edge_index_tar=torch.from_numpy(self.l1_dense_edge_index).long(), 

            node_mask=torch.from_numpy(node_mask).bool(),
            cluster_mask=torch.from_numpy(cluster_mask).bool(),                         # (C,)
            cluster_centers_index=torch.from_numpy(self.cluster_centers_index).long(),  # (C,), inc by num_nodes
            belong_cluster_index=torch.from_numpy(self.belong_cluster_index).long(),    # (N,), inc by num_clusters
        )

        return data
    
    def default_feature(self, points):
        return points
        # return np.concatenate([points, self.target_points - points], axis=-1)
