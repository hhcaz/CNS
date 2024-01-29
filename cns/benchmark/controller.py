import json
import torch
import numpy as np
from typing import Union, Dict
from ..models.graph_vs import GraphVS
from ..midend.graph_gen import GraphData
from ..ablation.ibvs.ibvs import IBVS


class GraphVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        self.device = torch.device(device)
        # self.net: GraphVS = torch.load(ckpt_path, map_location=self.device)["net"]
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if hasattr(ckpt, "net") and isinstance(ckpt["net"], torch.nn.Module):
            self.net: GraphVS = ckpt["net"]
        else:
            self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
            self.net.load_state_dict(ckpt)
        self.net.eval()
        self.hidden = None

    def __call__(self, data: GraphData) -> np.ndarray:
        with torch.no_grad():
            data = data.to(self.device)
            if hasattr(self.net, "preprocess"):
                data = self.net.preprocess(data)

            if getattr(data, "new_scene").any():
                print("[INFO] Got new scene, set hidden state to zero")
                self.hidden = None

            raw_pred = self.net(data, self.hidden)
            self.hidden = raw_pred[-1]
            vel = self.net.postprocess(raw_pred, data)

        vel = vel.squeeze(0).cpu().numpy()
        return vel


class IBVSController(object):
    def __init__(self, config_path: str):
        with open(config_path, "r") as fp:
            use_mean = json.load(fp)["use_mean"]
        self.ibvs = IBVS(use_mean)

    def __call__(self, data: GraphData) -> np.ndarray:
        return self.ibvs(data)


class ImageVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        from ..ablation.ibvs.raft_ibvs import RaftIBVS
        from ..reimpl import ICRA2018, ICRA2021
        
        self.device = torch.device(device)
        self.net: Union[ICRA2018, ICRA2021, RaftIBVS] = \
            torch.load(ckpt_path, map_location=self.device)["net"]
        self.net.eval()
        self.tar_feat = None
    
    def __call__(self, data: Dict) -> np.ndarray:
        with torch.no_grad():
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)
            
            if data.get("new_scene", True):
                self.tar_feat = None
            
            data["tar_feat"] = self.tar_feat
            raw_pred = self.net(data)
            self.tar_feat = data["tar_feat"]
            vel = self.net.postprocess(raw_pred, data)
        
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()
        vel = vel.flatten()
        return vel
