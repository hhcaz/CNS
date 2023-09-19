import torch
import torch.nn as nn
import torch.nn.functional as F
from cns.models.graph_vs import MLP, GraphVS


class DummyPTConv(nn.Module):
    def __init__(self, in_channels, out_channels, pos_nn):
        super(DummyPTConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        self.lin = nn.Linear(in_channels[0], out_channels, bias=False)
    
    def forward(self, x, pos_j, pos_i):
        x = self.lin(x)
        delta = self.pos_nn(pos_i - pos_j)
        return x + delta


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, pos_dim):
        super(Encoder, self).__init__()

        self.in_enc = MLP([input_dim, output_dim])
        self.att_clu_conv = DummyPTConv(
            in_channels=output_dim,
            out_channels=output_dim,
            pos_nn=MLP([pos_dim, output_dim, output_dim])
        )
        self.out_enc = MLP([output_dim*2, output_dim])
    
    def get_clu(self, x, x_ref, pos, pos_ref, 
                l0_to_l1_edge_index, centers_index):
        
        x_clu = F.relu(self.att_clu_conv(x, pos, pos_ref))
        return x_clu
    
    def forward(self, x_cur, x_tar, pos_cur, pos_tar, cluster_mask, 
                l0_to_l1_edge_index, centers_index):
        
        x_cur = F.relu(self.in_enc(x_cur))
        x_tar = F.relu(self.in_enc(x_tar))
        x_cur_clu = self.get_clu(
            x_cur, x_tar, pos_cur, pos_tar, 
            l0_to_l1_edge_index, centers_index
        )
        x_tar_clu = self.get_clu(
            x_tar, x_tar, pos_tar, pos_tar, 
            l0_to_l1_edge_index, centers_index
        )
        x_clu = torch.cat([x_cur_clu, x_tar_clu - x_cur_clu], dim=-1)
        x_clu = F.relu(self.out_enc(x_clu)) * cluster_mask.float().unsqueeze(-1)
        return x_clu


class GraphVS_NoCluster(GraphVS):
    def __init__(self, input_dim, pos_dim, hidden_dim=128, regress_norm=False):
        super(GraphVS_NoCluster, self).__init__(input_dim, pos_dim, hidden_dim, regress_norm)
        # override original encoder structure
        self.encoder = Encoder(input_dim, hidden_dim, pos_dim)


if __name__ == "__main__":
    from cns.sim.dataset import DataLoader
    from .graph_gen_no_cluster import GraphGeneratorNoCluster

    dataloader = DataLoader(
        None, 
        batch_size=2, train=True, num_trajs=100, 
        env="Point"
    )

    for dataset in dataloader.datasets:
        dataset.graph_gen_type = GraphGeneratorNoCluster
    
    if dataloader.train:
        for dataset in dataloader.datasets:
            dataset.env.obs_num_range = (4, 128)
            dataset.env.init()

    net = GraphVS_NoCluster(
        input_dim=dataloader.num_node_features,
        pos_dim=dataloader.num_pos_features,
    )

    hidden = None
    for data in dataloader:
        if getattr(data, "new_scene").any():
            hidden = None

        vel, hidden = net(data, hidden)
        dataloader.feedback(data.vel)

        print(vel)


