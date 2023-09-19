import cv2
import torch
import numpy as np

from ..utils.image_transform import scale_to_fit, pad_to_ratio, rot90
from .superglue_utils import get_matching, detect_feat, sort_and_trim_feat, add_suffix


class Frontend(object):
    def __init__(self, K=np.eye(3), device="cuda:0"):
        """
        K: camera intrinsic:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        self.K = K

        self.device = device
        self.matching = get_matching(self.device)
        self.matching.eval()

        self.tar_kp = None
        self.tar_img = None
        self.tar_pos = None
        self.tar_feat = None
    
    @torch.no_grad()
    def _extract_target(self, img0):
        """
        Arguments:
        - img0: np.ndarray, (H, W, 3)
        
        Returns:
        - feat0: features computed by superglue
        - inv_s: function to inverse scaling for keypoints
        - inv_p: function to inverse padding for keypoints
        - inv_rots: function to inverse rotation for keypoints
        """
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        box_size = int(round(np.sqrt(img0.shape[0] * img0.shape[1])))
        # box_size = 640
        
        img0_s, _, inv_s = scale_to_fit(img0, (box_size, box_size))
        img0_p, _, inv_p = pad_to_ratio(img0_s, 1, fill=0.5)

        img_rots = []
        inv_rots = []

        img_rots.append(img0_p)
        for k in [1, 2, 3]:
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
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        box_size = int(round(np.sqrt(img1.shape[0] * img1.shape[1])))
        # box_size = 640
        
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
        
        if len(matches) >= 4:
            tar_pts = self.tar_kp[matches[:, 0]]
            cur_pts = cur_kp[matches[:, 1]]

            if len(matches) == 4:
                M, mask = cv2.findHomography(tar_pts, cur_pts, cv2.RANSAC, 5.0)
            else:
                E, mask = cv2.findEssentialMat(tar_pts, cur_pts, self.K,
                    cv2.RANSAC, 0.999, 3.0)
            mask = mask.flatten().astype(bool)
            matches = matches[mask]
            confidence = confidence[mask]
        
        return matches, confidence

    def update_target_frame(self, image):
        feat, inv_s, inv_p, inv_rots = self._extract_target(image)
        kpts = []
        for i in range(len(feat["keypoints"])):
            kp = feat["keypoints"][i].cpu().numpy()  # (N, 2)
            if i >= 1:
                kp = inv_rots[i-1](kp)
            kp = inv_s(inv_p(kp))
            kpts.append(kp)
        uv_img_tar = np.concatenate(kpts, axis=0)  # (N*n_rot, 2)
        uv_norm_tar = cv2.undistortPoints(uv_img_tar, self.K, None).reshape(-1, 2)

        self.tar_kp = uv_img_tar  # (0, 0) ~ (W-1, H-1)
        self.tar_img = image
        self.tar_feat = feat
        self.tar_pos = uv_norm_tar  # normalized plane
        return True

    def process_current_frame(self, image):
        """
        Arguments:
        - image: np.ndarray, (H, W, 3), dtype=uint8 or float32

        Returns:
        - tar_pos: (N, 2), keypoint points on normalized camera plane or image plane at desired pose
        - cur_pos: (N, 2), keypoint points on ... at current pose
            Note: if K==eye(3), returns coordinates on image plane, otherwise on normalized camera plane
        - confidence: (N,), float32, confidence score calculated by superglue
        - observed_mask: (N,), bool, whether the match of tar_pos[i] and cur_pos[i] is valid.
            Note: The cur_pos has been rearranged according to the correspondence, this mask also indicates
                whether i-th tar_pos is observed at current frame
        """
        if self.tar_feat is None:
            print("[INFO] Haven't initialize for target frame!")
            return None
        
        feat, inv_s = self._extract_current(image)
        kpts = feat["keypoints"][0].cpu().numpy()  # (N, 2)
        kpts = inv_s(kpts)
        uv_img_cur = kpts  # (0, 0) ~ (W-1, H-1)
        uv_norm_cur = cv2.undistortPoints(uv_img_cur, self.K, None).reshape(-1, 2)

        matches, confidence = self._feature_match(uv_img_cur, feat)
        print("[INFO] Num good matches = {}".format(len(matches)))
        if len(matches) < 4:
            print("[INFO] Too few matches (={})".format(len(matches)))
            return None

        cur_pos = np.zeros_like(self.tar_pos)
        observed_mask = np.zeros(len(self.tar_pos), dtype=bool)
        cur_pos[matches[:, 0]] = uv_norm_cur[matches[:, 1]]
        observed_mask[matches[:, 0]] = True
        
        return self.tar_pos, cur_pos, confidence, observed_mask


if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)
    frontend = Frontend(K=np.eye(3), device="cuda:0")

    key = 'n'
    while True:
        if key == 'n':
            ret, img0 = cap.read()
            frontend.update_target_frame(img0)
        elif key == 'q':
            break
        
        ret, img1 = cap.read()
        if ret:
            tar_pos, cur_pos, conf, mask = frontend.process_current_frame(img1)
            tar_pos, cur_pos, conf = tar_pos[mask], cur_pos[mask], conf[mask]

            H, W = img1.shape[:2]
            tar_pos = np.round(tar_pos).astype(np.int32).clip(0, [W-1, H-1])
            cur_pos = np.round(cur_pos).astype(np.int32).clip(0, [W-1, H-1])

            image = np.concatenate([img0, img1], axis=1)
            for tp, cp, ci in zip(tp, cp, conf):
                cv2.line(tuple(tp), tuple(cp + [W, 0]), color=[0, 255, 0])
            cv2.imshow("target | current", image)
            key = chr(cv2.waitKey(1) & 0xFF)

