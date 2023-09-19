import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from cns.models.graph_vs import GraphVS


class SimpleConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, pos_channels):
        super(SimpleConvGRUCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_channels = pos_channels

        self.fc = nn.Linear(in_channels+out_channels, out_channels)
        self.conv = GatedGraphConv(
            out_channels=out_channels,
            num_layers=1
        )
    
    def init_hidden(self, size0):
        """Preserve same interface as PEConvGRU"""
        zeros = torch.zeros(size0, self.out_channels)
        return zeros
    
    def forward(self, h, x, pos, edge_index_gate, edge_index_cand):
        """
        Preserve same interface as PEConvGRU.
        Therefore, `pos` and `edge_index_cand` are not used.
        """
        x = torch.cat([x, h], dim=-1)
        x = F.relu(self.fc(x))
        h_next = self.conv(x, edge_index_gate)
        return h_next


class GraphVS_SimpleGRU(GraphVS):
    def __init__(self, input_dim, pos_dim, hidden_dim=128, regress_norm=False):
        super(GraphVS_SimpleGRU, self).__init__(input_dim, pos_dim, hidden_dim, regress_norm)

        # replace original PEConvGRUCell with SimpleConvGRUCell
        conv_gru = self.backbone.temporal_aggr
        self.backbone.temporal_aggr = SimpleConvGRUCell(
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

    net = GraphVS_SimpleGRU(
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

