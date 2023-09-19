# import time
import cv2
import torch
import numbers
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.optical_flow._utils import make_coords_grid
from cns.utils.perception import CameraIntrinsic
from .flow_vis import flow_to_color


def ibvs_Lmean_torch(
    fp_cur: torch.Tensor,
    Z_cur: torch.Tensor,
    fp_tar: torch.Tensor,
    Z_tar: torch.Tensor
) -> torch.Tensor:
    """Image-based visual servo controller.

    Arguments:
    - fp_cur: (N, 2), 2 represents (x, y), 
            feature points in current normalized camera frame
    - Z_cur: (N,), depth of feature points in current camera frame
    - fp_tar: (N, 2), feature points in target normalized camera frame
    - Z_tar: (N,), depth of feature points in target camera frame

    Returns:
    - vel: (6,), [tx, ty, tz, wx, wy, wz], camera velocity in current camera frame
    """

    assert fp_cur.size() == fp_tar.size(), "number of feature points not match"
    num_fp = fp_cur.shape[0]

    x_cur = fp_cur[:, 0]
    y_cur = fp_cur[:, 1]
    # build interaction matrix at current camera frame
    L_cur = torch.zeros((num_fp * 2, 6), device=fp_cur.device)
    L_cur[0::2, 0] = -1. / Z_cur
    L_cur[0::2, 2] = x_cur / Z_cur
    L_cur[0::2, 3] = x_cur * y_cur
    L_cur[0::2, 4] = -(1 + x_cur * x_cur)
    L_cur[0::2, 5] = y_cur
    L_cur[1::2, 1] = -1. / Z_cur
    L_cur[1::2, 2] = y_cur / Z_cur
    L_cur[1::2, 3] = 1 + y_cur * y_cur
    L_cur[1::2, 4] = -x_cur * y_cur
    L_cur[1::2, 5] = -x_cur

    x_tar = fp_tar[:, 0]
    y_tar = fp_tar[:, 1]
    # build interaction matrix at target camera frame
    L_tar = torch.zeros((num_fp * 2, 6), device=fp_tar.device)
    L_tar[0::2, 0] = -1. / Z_tar
    L_tar[0::2, 2] = x_tar / Z_tar
    L_tar[0::2, 3] = x_tar * y_tar
    L_tar[0::2, 4] = -(1 + x_tar * x_tar)
    L_tar[0::2, 5] = y_tar
    L_tar[1::2, 1] = -1. / Z_tar
    L_tar[1::2, 2] = y_tar / Z_tar
    L_tar[1::2, 3] = 1 + y_tar * y_tar
    L_tar[1::2, 4] = -x_tar * y_tar
    L_tar[1::2, 5] = -x_tar

    error = torch.zeros(num_fp * 2, device=fp_cur.device)
    error[0::2] = x_tar - x_cur
    error[1::2] = y_tar - y_cur

    # invL = torch.linalg.pinv((L_cur + L_tar) / 2.)
    # vel = torch.dot(invL, error)

    vel = torch.linalg.lstsq((L_cur + L_tar) / 2., error).solution
    return vel


class RaftIBVS(nn.Module):
    def __init__(self, ransac=True):
        super().__init__()

        weights = Raft_Large_Weights.DEFAULT
        self.raft = raft_large(weights=weights)
        self.transforms = weights.transforms()
        self.ransac = ransac
    
    @classmethod
    def preprocess(cls, image: torch.Tensor):
        _, _, H, W = image.size()

        Hmod8 = H % 8
        pt = Hmod8 // 2
        pb = Hmod8 - pt

        Wmod8 = W % 8
        pl = Wmod8 // 2
        pr = Wmod8 - pl

        if Hmod8 > 0 or Wmod8 > 0:
            image = F.pad(image, (pl, pr, pt, pb), "constant", 0)
        return image

    def forward(self, data: dict):
        tar_img: torch.Tensor = data.get("tar_img")
        cur_img: torch.Tensor = data.get("cur_img")

        tar_img = self.preprocess(tar_img)
        cur_img = self.preprocess(cur_img)
        tar_img, cur_img = self.transforms(tar_img, cur_img)

        list_of_flows = self.raft(tar_img, cur_img)

        return list_of_flows
    
    def grid_sample(self, pos: torch.Tensor):
        # pos: (B, 2, H, W)
        B, _, H, W = pos.size()
        # grid = make_coords_grid(B, H//10, W//10, pos.device)
        # grid = grid / grid.max(dim=1, keepdim=True).values * 2 - 1
        # print("grid min = {}, grid max = {}".format(grid.min().item(), grid.max().item()))
        # grid = rearrange(grid, "b c h w -> b h w c")
        # sampled_grid = F.grid_sample(pos, grid, align_corners=False)
        sampled_grid = F.interpolate(pos, scale_factor=0.1, mode="bilinear", align_corners=False)
        # sampled_grid = F.interpolate(pos, scale_factor=0.1, mode="area")
        # print("sample_grid size = {}".format(sampled_grid.size()))
        return sampled_grid
    
    def postprocess(self, raw_pred, data: dict):
        flows: torch.Tensor = raw_pred[-1]  # (B, 2, H, W)
        intrinsic: CameraIntrinsic = data.get("intrinsic")
        dist_scale: list = data.get("dist_scale")

        flow_image = flow_to_color(flows[-1]).permute(1, 2, 0).cpu().numpy() / 255
        cv2.imshow("flow", flow_image)
        cv2.waitKey(1)

        B, _, H, W = flows.size()
        pos_tar = make_coords_grid(B, H, W, flows.device)  # (B, 2, H, W)
        pos_cur = pos_tar + flows  # (B, 2, H, W)

        self.ransac = True
        if self.ransac:
            # print("[INFO] Do ransac!")
            pos_tar = self.grid_sample(pos_tar)
            pos_cur = self.grid_sample(pos_cur)

        pos_tar = rearrange(pos_tar, "b c h w -> b (h w) c")
        pos_cur = rearrange(pos_cur, "b c h w -> b (h w) c")

        # pixel to norm camera plane
        cxy = torch.tensor([intrinsic.cx, intrinsic.cy]).to(flows)
        fxy = torch.tensor([intrinsic.fx, intrinsic.fy]).to(flows)
        pos_tar = (pos_tar - cxy) / fxy
        pos_cur = (pos_cur - cxy) / fxy

        if isinstance(dist_scale, numbers.Number):
            dist_scale = [dist_scale] * B

        # t0 = time.time()
        vels = []
        for b in range(B):
            # self.ransac = False
            if self.ransac:
                mask = self.ransac_pos(pos_cur[b], pos_tar[b])
                v = ibvs_Lmean_torch(
                    fp_cur=pos_cur[b][mask],
                    Z_cur=dist_scale[b],
                    fp_tar=pos_tar[b][mask],
                    Z_tar=dist_scale[b]
                )
            else:
                v = ibvs_Lmean_torch(
                    fp_cur=pos_cur[b],
                    Z_cur=dist_scale[b],
                    fp_tar=pos_tar[b],
                    Z_tar=dist_scale[b]
                )
            vels.append(v)
        vels = torch.stack(vels, axis=0)  # (B, 6)
        # t1 = time.time()
        # print("[INFO] IBVS time: {:.3f}s".format(t1-t0))
        return vels
    
    def ransac_pos(self, pos_cur: torch.Tensor, pos_tar: torch.Tensor):
        pos_cur = pos_cur.detach().cpu().numpy()
        pos_tar = pos_tar.detach().cpu().numpy()

        E, mask = cv2.findEssentialMat(pos_tar, pos_cur, np.eye(3), 
            cv2.RANSAC, 0.999, 3.0)
        mask = mask.ravel().astype(bool)
        # print("--------mask--------")
        # print(mask.sum())
        mask = torch.from_numpy(mask).bool()
        return mask


if __name__ == "__main__":
    # from tqdm import tqdm

    # with torch.no_grad():
    #     A = torch.randn(512*512, 6).to("cuda:0")
    #     b = torch.randn(512*512).to("cuda:0")

    #     x = torch.linalg.lstsq(A, b)
    #     for _ in tqdm(range(100)):
    #         x = torch.linalg.lstsq(A, b)
    #     print(x)

    raft = RaftIBVS()

    grid = make_coords_grid(1, 3, 5)
    print(grid[0])
    print(grid.size())
