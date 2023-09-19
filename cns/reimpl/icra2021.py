import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.models.efficientnet import efficientnet_b3


class ICRA2021(nn.Module):
    def __init__(self):
        super().__init__()
        # use EfficientNetB3 as mentioned in section V.A
        self.feature_extractor = efficientnet_b3(weights="IMAGENET1K_V1")
        self.velocity_regressor = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )
        self.criterion = MultiTaskLoss()
    
    @torch.no_grad()
    def preprocess(self, img: torch.Tensor):
        _, _, H, W = img.size()
        if H >= W:
            pl = (H - W) // 2
            pr = H - W - pl
            pt = pb = 0
        else:
            pt = (W - H) // 2
            pb = W - H - pt
            pl = pr = 0

        # First pad to square
        if H != W:
            img = F.pad(img, (pl, pr, pt, pb), "constant", 0)
        
        # Next scale to 224x224, according to section V.A
        desired_image_size = 224
        if max(H, W) != desired_image_size:
            img = F.interpolate(img, size=(desired_image_size, desired_image_size), 
                mode="bilinear", align_corners=False)
        return img
    
    def _efficient_forward(self, x: torch.Tensor):
        with torch.inference_mode():
            self.feature_extractor.eval()
            x = TVF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            x = self.feature_extractor.features(x)
            x = self.feature_extractor.avgpool(x)
            x = torch.flatten(x, 1)
        return x.clone()
    
    def forward(self, data: dict):
        """
        data: Dict contains key of:
        - cur_img: torch.Tensor, (B, 3, H, W)
        - one of tar_img and tar_feat
        """
        cur_img: torch.Tensor = data.get("cur_img")
        tar_img: torch.Tensor = data.get("tar_img", None)
        tar_feat: torch.Tensor = data.get("tar_feat", None)

        if tar_feat is None:
            assert tar_img is not None, (
                "At leat one of tar_img and tar_feat should be provided")

        cur_feat = self._efficient_forward(self.preprocess(cur_img))
        if tar_feat is None:
            tar_feat = self._efficient_forward(self.preprocess(tar_img))
            data["tar_feat"] = tar_feat

        x = cur_feat - tar_feat
        v = self.velocity_regressor(x)
        return v
    
    def trainable_parameters(self):
        return self.velocity_regressor.parameters()
    
    def postprocess(raw_pred, *args, **kwargs):
        return raw_pred


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_vxy = nn.Parameter(torch.tensor(0.).float())
        self.scale_vz  = nn.Parameter(torch.tensor(0.).float())
        self.scale_wxy = nn.Parameter(torch.tensor(0.).float())
        self.scale_wz  = nn.Parameter(torch.tensor(0.).float())

    def forward(self, pred_v, gt_v):
        sigma2_vxy = torch.exp(self.scale_vxy)
        sigma2_vz  = torch.exp(self.scale_vz)
        sigma2_wxy = torch.exp(self.scale_wxy)
        sigma2_wz  = torch.exp(self.scale_wz)

        vxy_loss = F.mse_loss(pred_v[..., 0:2], gt_v[..., 0:2], reduction="mean")
        vz_loss  = F.mse_loss(pred_v[..., 2:3], gt_v[..., 2:3], reduction="mean")
        wxy_loss = F.mse_loss(pred_v[..., 3:5], gt_v[..., 3:5], reduction="mean")
        wz_loss  = F.mse_loss(pred_v[..., 5:6], gt_v[..., 5:6], reduction="mean")

        regular_loss = self.scale_vxy + self.scale_vz + self.scale_wxy + self.scale_wz
        total_loss = vxy_loss / sigma2_vxy + vz_loss / sigma2_vz + \
                     wxy_loss / sigma2_wxy + wz_loss / sigma2_wz + regular_loss
        no_weight_loss = vxy_loss + vz_loss + wxy_loss + wz_loss

        return {
            "vxy_loss": vxy_loss,
            "vz_loss": vz_loss,
            "wxy_loss": wxy_loss,
            "wz_loss": wxy_loss,
            "regular_loss": regular_loss,
            "total_loss": total_loss,
            "no_weight_loss": no_weight_loss
        }


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = ICRA2021().to(device)
    criterion = MultiTaskLoss().to(device)
    cur_img = torch.randn((4, 3, 224, 224)).to(device)
    tar_img = torch.randn((4, 3, 224, 224)).to(device)

    pred_vel, _ = model(cur_img, tar_img)
    gt_vel = torch.randn_like(pred_vel)
    loss = criterion(pred_vel, gt_vel)

    print(pred_vel)
    print(loss)

