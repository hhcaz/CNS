import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from .graph_vs_simple_gru import GraphVS_SimpleGRU
from cns.train_gvs_short_seq import train


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        model_class=GraphVS_SimpleGRU,
        suffix="simple_gru",
        save="posix" in os.name
    )
