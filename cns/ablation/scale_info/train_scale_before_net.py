import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from cns.sim.dataset import DataLoader
from cns.models.graph_vs import GraphVS
from cns.midend.graph_gen import GraphData
from cns.train_gvs_short_seq import train
from cns.models.functions import postprocess, objectives


class GraphVS_ScaleBeforeNet(GraphVS):

    # def preprocess(self, data: GraphData):
    #     scale = getattr(data, "tPo_norm")
    #     batch = getattr(data, "batch", None)
    #     if batch is not None:
    #         scale = scale[batch].view(-1, 1)
        
    #     for attr in ["x_cur", "x_tar", "pos_cur", "pos_tar"]:
    #         setattr(data, attr, getattr(data, attr) * scale)
    #     return data
    
    @classmethod
    def preprocess(cls, data: GraphData):
        scale: torch.Tensor = getattr(data, "tPo_norm")
        batch = getattr(data, "batch", None)
        if batch is not None:
            scale = scale[batch]
        scale = scale.view(-1, 1)
        
        for attr in ["x_cur", "x_tar"]:
            attr_val: torch.Tensor = getattr(data, attr)
            if scale.size(0) == 1:
                scale = scale.repeat(attr_val.size(0), 1)
            attr_val = torch.cat([attr_val, scale], dim=-1)
            setattr(data, attr, attr_val)
        return data
    
    @classmethod
    def postprocess(self, raw_pred, *args):
        return postprocess(raw_pred, None, post_scale=False)
    
    @classmethod
    def objectives(cls, raw_pred, data: GraphData):
        # weight = torch.tensor([1]*3 + [0.53]*3).float().to(raw_pred[0].device)
        weight = 1
        return objectives(raw_pred, data, gt_si=False, weight=weight)


class DataLoader_ScaleBeforeNet(DataLoader):
    @property
    def num_node_features(self):
        return super().num_node_features + 1


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        model_class=GraphVS_ScaleBeforeNet,
        data_class=DataLoader_ScaleBeforeNet,
        suffix="scale_before_net",
        save="posix" in os.name
    )
