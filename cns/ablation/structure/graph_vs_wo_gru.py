import torch
import torch.nn as nn
import torch.nn.functional as F
from cns.models.graph_vs import MLP, PointEdgeConv, GraphVS


class DummyConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, pos_channels):
        super(DummyConvGRUCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_channels = pos_channels

        self.conv = PointEdgeConv(
            in_channels=in_channels,
            out_channels=out_channels,
            msg_nn=MLP([in_channels*2 + pos_channels, out_channels])
        )

    def init_hidden(self, *args):
        """Preserve same interface as PEConvGRU"""
        return torch.tensor(0.).float()
    
    def forward(self, h, x, pos, edge_index_gate, edge_index_cand):
        """
        Preserve same interface as PEConvGRU.
        Therefore, `h` and `edge_index_gate` are not used.
        """
        x = F.relu(self.conv(x, pos, edge_index_cand))
        return x


class GraphVS_woGRU(GraphVS):
    def __init__(self, input_dim, pos_dim, hidden_dim=128, regress_norm=False):
        super(GraphVS_woGRU, self).__init__(input_dim, pos_dim, hidden_dim, regress_norm)

        # replace original PEConvGRUCell with DummyConvGRUCell
        conv_gru = self.backbone.temporal_aggr
        self.backbone.temporal_aggr = DummyConvGRUCell(
            in_channels=conv_gru.in_channels,
            out_channels=conv_gru.out_channels,
            pos_channels=conv_gru.pos_channels
        )
        del conv_gru


if __name__ == "__main__":
    from cns.sim.dataset import DataLoader

    dataloader = DataLoader(
        None, 
        batch_size=2, train=True, num_trajs=100, 
        env="Point"
    )

    net = GraphVS_woGRU(
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

