import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from cns.sim.dataset import DataLoader
from .graph_vs_no_cluster import GraphVS_NoCluster
from .graph_gen_no_cluster import GraphGeneratorNoCluster
from cns.train_gvs_short_seq import train


class DataLoader_NoCluster(DataLoader):
    def __init__(self, camera_config, batch_size, train=True, num_trajs=100, env="Point"):
        super().__init__(camera_config, batch_size, train, num_trajs, env)

        for dataset in self.datasets:
            dataset.graph_gen_type = GraphGeneratorNoCluster
        
        if self.train:
            for dataset in self.datasets:
                dataset.env.obs_num_range = (4, 128)
                dataset.env.init()


if __name__ == "__main__":
    train(
        batch_size=20,
        device=torch.device("cuda:0"),
        model_class=GraphVS_NoCluster,
        data_class=DataLoader_NoCluster,
        suffix="no_cluster",
        save="posix" in os.name
    )
