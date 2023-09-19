import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from cns.models.graph_vs import MLP, GraphVS


class EdgeConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvGRUCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_gates = EdgeConv(
            nn=MLP([(in_channels+out_channels)*2, out_channels*2]),
            aggr="max"
        )

        self.conv_candi = EdgeConv(
            nn=MLP([(in_channels+out_channels)*2, out_channels]),
            aggr="max"
        )
    
    def init_hidden(self, size0):
        zeros = torch.zeros(size0, self.out_channels)
        return zeros
    
    def forward(self, h, x, edge_index_gate, edge_index_cand):
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.conv_gates(combined, edge_index_gate))
        reset_gate, update_gate = torch.chunk(gates, chunks=2, dim=-1)

        combined = torch.cat([x, h*reset_gate], dim=-1)
        ht = torch.tanh(self.conv_candi(combined, edge_index_cand))

        h_next = (1 - update_gate) * h + update_gate * ht
        return h_next


class Backbone(nn.Module):
    def __init__(self, hidden_dim):
        super(Backbone, self).__init__()

        self.l1_conv0 = EdgeConv(MLP([hidden_dim*2, hidden_dim]), "max")
        self.temporal_aggr = EdgeConvGRUCell(hidden_dim, hidden_dim)
        self.l1_conv1 = EdgeConv(MLP([hidden_dim*2, hidden_dim]), "max")
    
    def forward(self, hidden, x_clu, pos_clu, 
                l1_dense_edge_index_cur, l1_dense_edge_index_tar, batch_clu):
        # pos_clu & batch_clu are preserved arguments to maintain
        # the same interface as Backbone used in GraphVS
        x_clu = F.relu(self.l1_conv0(x_clu, l1_dense_edge_index_cur))
        hidden = self.temporal_aggr(
            hidden, x_clu, 
            edge_index_gate=l1_dense_edge_index_tar,
            edge_index_cand=l1_dense_edge_index_cur
        )
        x_clu = F.relu(self.l1_conv1(hidden, l1_dense_edge_index_cur))
        return hidden, x_clu


class GraphVS_EdgeConv(GraphVS):
    def __init__(self, input_dim, pos_dim, hidden_dim=128, regress_norm=False):
        super(GraphVS_EdgeConv, self).__init__(input_dim, pos_dim, hidden_dim, regress_norm)

        del self.backbone
        self.backbone = Backbone(hidden_dim)


if __name__ == "__main__":
    from cns.sim.dataset import DataLoader

    dataloader = DataLoader(
        None, 
        batch_size=2, train=True, num_trajs=100, 
        env="Point"
    )

    net = GraphVS_EdgeConv(
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
