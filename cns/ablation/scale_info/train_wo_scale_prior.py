import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from cns.models.graph_vs import GraphVS
from cns.midend.graph_gen import GraphData
from cns.train_gvs_short_seq import train
from cns.models.functions import postprocess, objectives


class GraphVS_woScalePrior(GraphVS):

    @classmethod
    def postprocess(cls, raw_pred, *args):
        return postprocess(raw_pred, None, post_scale=False)
    
    @classmethod
    def objectives(cls, raw_pred, data: GraphData):
        # weight = torch.tensor([1]*3 + [0.53]*3).float().to(raw_pred[0].device)
        weight = 1
        return objectives(raw_pred, data, gt_si=False, weight=weight)


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        model_class=GraphVS_woScalePrior,
        suffix="wo_scale_prior",
        save="posix" in os.name
    )
