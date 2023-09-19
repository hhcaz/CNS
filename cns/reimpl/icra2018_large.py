import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from scipy.spatial.transform import Rotation as R
from torchvision.models.alexnet import alexnet


class ICRA2018Large(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = alexnet(weights="IMAGENET1K_V1")
        del self.feature_extractor.avgpool
        del self.feature_extractor.classifier

        self.post_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(start_dim=1)
        )

        self.pose_regressor = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 6),
        )
        self.criterion = WeightMSELoss()

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
        desired_image_size = 448
        if max(H, W) != desired_image_size:
            img = F.interpolate(img, size=(desired_image_size, desired_image_size), 
                mode="bilinear", align_corners=False)
        return img

    def _alex_forward(self, x: torch.Tensor):
        with torch.inference_mode():
            self.feature_extractor.eval()
            x = TVF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            x = self.feature_extractor.features(x)  # (B, 256, H, W)
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

        cur_feat = self._alex_forward(self.preprocess(cur_img))
        if tar_feat is None:
            tar_feat = self._alex_forward(self.preprocess(tar_img))
            data["tar_feat"] = tar_feat

        return {
            "cur_rela_pose": self.pose_regressor(self.post_conv(cur_feat)),
            "tar_rela_pose": self.pose_regressor(self.post_conv(tar_feat))
        }

    def trainable_parameters(self):
        # return self.pose_regressor.parameters()
        params = []
        for m in [
            self.post_conv,
            self.pose_regressor
        ]:
            params.extend(list(m.parameters()))
        return params

    def postprocess(self, raw_pred, *args, **kwargs):
        """Raw network predictions to velocity."""
        for key in ["cur_rela_pose", "tar_rela_pose"]:
            if isinstance(raw_pred[key], torch.Tensor):
                raw_pred[key] = raw_pred[key].detach().cpu().numpy()
        
        assert raw_pred["cur_rela_pose"].ndim == 2  # (B, 6)
        B = raw_pred["cur_rela_pose"].shape[0]

        cur_rela_pose = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)  # (B, 4, 4)
        cur_rela_pose[:, :3, :3] = R.from_rotvec(raw_pred["cur_rela_pose"][:, 3:6]).as_matrix()
        cur_rela_pose[:, :3, 3] = raw_pred["cur_rela_pose"][:, 0:3]

        tar_rela_pose = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)  # (B, 4, 4)
        tar_rela_pose[:, :3, :3] = R.from_rotvec(raw_pred["tar_rela_pose"][:, 3:6]).as_matrix()
        tar_rela_pose[:, :3, 3] = raw_pred["tar_rela_pose"][:, 0:3]

        pred_vel = []
        for b in range(B):
            # cur_abs_pose = ref_pose @ cur_rela_pose[b]
            # tar_abs_pose = ref_pose @ tar_rela_pose[b]
            # tcT = np.linalg.inv(tar_abs_pose) @ cur_abs_pose
            # # equivlent to:
            tcT = np.linalg.inv(tar_rela_pose[b]) @ cur_rela_pose[b]
            # calculate pbvs velocity
            u = R.from_matrix(tcT[:3, :3]).as_rotvec()
            v = -tcT[:3, :3].T @ tcT[:3, 3]
            w = -u
            vel = np.concatenate([v, w])
            pred_vel.append(vel)
        pred_vel = np.stack(pred_vel, axis=0)
        return pred_vel


class WeightMSELoss(nn.Module):
    def forward(self, raw_pred, gt_poses):
        """
        - raw_pred: raw_pred from ICRA2018 forward
        - gt_poses: (B, 2, 6), 2 represents [cur_rela_pose, tar_rela_pose]
        """
        cur_rela_pose = raw_pred["cur_rela_pose"]
        gt_cur_rela_pose = gt_poses[:, 0, :]

        # t_loss = F.mse_loss(cur_rela_pose[:, 0:3], gt_cur_rela_pose[:, 0:3], reduction="mean")
        # u_loss = F.mse_loss(
        #     cur_rela_pose[:, 3:6] / math.pi * 180,  # in degrees, see section II.C
        #     gt_cur_rela_pose[:, 3:6] / math.pi * 180, 
        #     reduction="mean"
        # )
        # total_loss = t_loss + 0.01 * u_loss  # section II.C, Eq. 7


        t_loss = F.mse_loss(cur_rela_pose[:, 0:3], gt_cur_rela_pose[:, 0:3], reduction="mean")
        u_loss = F.mse_loss(cur_rela_pose[:, 3:6], gt_cur_rela_pose[:, 3:6], reduction="mean")
        total_loss = t_loss + 0.1 * u_loss

        return {
            "t_loss": t_loss,
            "u_loss": u_loss,
            "total_loss": total_loss
        }


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = ICRA2018Large().to(device)
    cur_img = torch.randn((4, 3, 224, 224)).to(device)
    tar_img = torch.randn((4, 3, 224, 224)).to(device)

    pred_vel, _ = model(cur_img, tar_img)
    gt_vel = torch.randn_like(pred_vel)
    loss = model.criterion(pred_vel, gt_vel)

    print(pred_vel)
    print(loss)

