import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from cns.sim.dataset import DataLoader, ObsAug
from cns.sim.sampling import gen_virtual_points_uniformed
from cns.train_gvs_short_seq import train


class DataLoader_Uniform(DataLoader):
    def __init__(self, camera_config, batch_size, train=True, num_trajs=100, env="Point"):
        super().__init__(camera_config, batch_size, train, num_trajs, env)

        if self.train:
            for dataset in self.datasets:
                dataset.aug = ObsAug.RandomMismatch | ObsAug.RandomDiscard | ObsAug.PosJitter
                # override the original sampling strategy
                dataset.env.sample_points_func = gen_virtual_points_uniformed
                # re-initialize environment
                dataset.env.init()


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        data_class=DataLoader_Uniform,
        suffix="data_all_uniform",
        save="posix" in os.name
    )
