import cv2
import torch
import numpy as np

from ..utils.perception import CameraIntrinsic
from ..utils.image_transform import scale_to_fit, pad_to_ratio, rot90
from .superglue_utils import get_matching, detect_feat, sort_and_trim_feat, add_suffix
from .utils import Correspondence, FrontendBase


class SuperGlue(FrontendBase):
    def __init__(
        self, 
        intrinsic: CameraIntrinsic, 
        detector="SuperGlue:0", 
        device="cuda:0",
        ransac=True
    ):
        self.intrinsic = intrinsic
        self.detector_str = detector
        _, self.rot90s = self._parse_detector(detector)

        self.device = device
        self.matching = get_matching(self.device)
        self.matching.eval()
        self.ransac = ransac

        self.tar_kp = None
        self.tar_img = None
        self.tar_img_changed = True
        self.tar_feat = None
    
    @classmethod
    def _parse_detector(cls, config: str):
        items = config.split(":")
        name = items[0]
        if len(items) == 1:
            rot90s = [0]
        else:
            rot90s = [int(c)%4 for c in items[1]]
            rot90s = list(set(rot90s))  # remove duplication
            rot90s.sort()
        
        return name, rot90s
    
    @torch.no_grad()
    def _extract_target(self, img0, mask=None):
        """
        Arguments:
        - img0: np.ndarray, (H, W, 3)
        
        Returns:
        - feat0: features computed by superglue
        - inv_s: function to inverse scaling for keypoints
        - inv_p: function to inverse padding for keypoints
        - inv_rots: function to inverse rotation for keypoints
        """
        # img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        if isinstance(mask, np.ndarray):
            img0[mask.astype(bool)] = 0
        box_size = int(round(np.sqrt(img0.shape[0] * img0.shape[1])))
        
        img0_s, _, inv_s = scale_to_fit(img0, (box_size, box_size))
        img0_p, _, inv_p = pad_to_ratio(img0_s, 1, fill=0.5)

        img_rots = []
        inv_rots = []

        for k in self.rot90s:
            if k == 0:
                img0_r = img0_p
                inv_r = lambda x: x  # identity mapping
            else:
                img0_r, _, inv_r = rot90(img0_p, k)
            
            img_rots.append(img0_r)
            inv_rots.append(inv_r)

        frame0_tensor = np.stack(img_rots, axis=0)
        frame0_tensor = torch.from_numpy(frame0_tensor).unsqueeze(1).float().to(self.device)
        feat0 = detect_feat(frame0_tensor, self.matching.superpoint)
        feat0 = sort_and_trim_feat(feat0)

        return feat0, inv_s, inv_p, inv_rots
    
    @torch.no_grad()
    def _extract_current(self, img1):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        box_size = int(round(np.sqrt(img1.shape[0] * img1.shape[1])))
        
        img1_s, _, inv_s = scale_to_fit(img1, (box_size, box_size))
        frame1_tensor = torch.from_numpy(img1_s[None, None]).float().to(self.device)
        feat1 = detect_feat(frame1_tensor, self.matching.superpoint)

        return feat1, inv_s

    @torch.no_grad()
    def _feature_match(self, cur_kp, cur_feat):
        data = dict(**add_suffix(self.tar_feat, "0"), 
                    **add_suffix(cur_feat, "1"))
        result = self.matching(data)

        offset = 0
        matches = [[], []]
        for m0 in result["matches0"].cpu().numpy():
            valid_index = np.nonzero(m0 > -1)[0]
            if len(valid_index):
                matches[0].append(valid_index + offset)
                matches[1].append(m0[valid_index])
            offset += len(m0)
        
        if len(matches[0]):
            matches[0] = np.concatenate(matches[0])
            matches[1] = np.concatenate(matches[1])
            matches = np.stack(matches, axis=-1)  # (M, 2)
        else:
            matches = np.zeros((0, 2), dtype=np.int64)
        
        conf0 = np.concatenate(
            list(result["matching_scores0"].cpu().numpy()), axis=0)
        confidence = conf0[matches[:, 0]]
        
        if self.ransac and len(matches) >= 4:
            tar_pts = self.tar_kp[matches[:, 0]]
            cur_pts = cur_kp[matches[:, 1]]

            if len(matches) == 4:
                M, mask = cv2.findHomography(tar_pts, cur_pts, cv2.RANSAC, 5.0)
            else:
                E, mask = cv2.findEssentialMat(tar_pts, cur_pts, self.intrinsic.K,
                    cv2.RANSAC, 0.999, 3.0)
            mask = mask.flatten().astype(bool)
            matches = matches[mask]
            confidence = confidence[mask]
        
        return matches, confidence

    def update_target_frame(self, image, mask=None):
        feat, inv_s, inv_p, inv_rots = self._extract_target(image, mask)
        kpts = []
        for i in range(len(feat["keypoints"])):
            kp = feat["keypoints"][i].cpu().numpy()  # (N, 2)
            kp = inv_rots[i](kp)
            kp = inv_s(inv_p(kp))
            kpts.append(kp)
        kpts = np.concatenate(kpts, axis=0)  # (N*n_rot, 2)

        self.tar_kp = kpts  # (0, 0) ~ (W-1, H-1)
        self.tar_img = image
        self.tar_img_changed = True
        self.tar_feat = feat
        return True

    def process_current_frame(self, image):
        if self.tar_feat is None:
            print("[INFO] Haven't initialize for target frame!")
            return None
        
        feat, inv_s = self._extract_current(image)
        kpts = feat["keypoints"][0].cpu().numpy()  # (N, 2)
        kpts = inv_s(kpts)

        matches, confidence = self._feature_match(kpts, feat)
        if len(matches) < 4:
            print("[INFO] Too few matches (={})".format(len(matches)))
            return None

        cur_kp_aligned = np.zeros_like(self.tar_kp)
        observed_mask = np.zeros(len(self.tar_kp), dtype=bool)
        # reorder kpts to align with target kpts
        cur_kp_aligned[matches[:, 0]] = kpts[matches[:, 1]]
        observed_mask[matches[:, 0]] = True

        corr = Correspondence(
            intrinsic=self.intrinsic,
            tar_img=self.tar_img,
            tar_pos=self.tar_kp,

            cur_img=image,
            cur_pos=kpts,

            match=matches,
            valid_mask=observed_mask,
            cur_pos_aligned=cur_kp_aligned, 

            detector_name=self.detector_str,
            tar_img_changed=self.tar_img_changed
        )
        # will be set to True when `update_target_frame` is called`
        self.tar_img_changed = False
        return corr
