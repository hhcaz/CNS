import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from .graph_vs_edge_conv import GraphVS_EdgeConv
from cns.train_gvs_short_seq import train


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        model_class=GraphVS_EdgeConv,
        suffix="edge_conv",
        save="posix" in os.name
    )
