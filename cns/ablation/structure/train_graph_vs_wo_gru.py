import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from .graph_vs_wo_gru import GraphVS_woGRU
from cns.train_gvs_short_seq import train


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        model_class=GraphVS_woGRU,
        suffix="wo_gru",
        save="posix" in os.name,
        steps_for_update=1,  # Changes to 1 here !!! since there's no temporal modeling
    )
