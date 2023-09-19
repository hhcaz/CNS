import cv2
import numpy as np
from ..utils.image_transform import rot90
from ..utils.perception import CameraIntrinsic
from .utils import Correspondence, FrontendBase


class Classic(FrontendBase):
    def __init__(
        self, 
        intrinsic: CameraIntrinsic, 
        detector="SIFT:0", 
        ransac=True
    ):
        """
        Arguments
        - intrinsic: 3x3 numpy array, set to eye(3) is K is None
        - detector: str, can be SIFT, AKAZE, ORB, etc. The trailing number means
            rotate target image anti-clock-wise by n*90 degrees, e.g.: 
            SIFT:02 means using SIFT detector, and match the original target image
            and the image rotated by 180 degrees
        """
        self.intrinsic = intrinsic
        self.detector_str = detector.strip()
        self.detecor_name, self.rot90s = self._parse_detector(self.detector_str)
        self.detector = self._get_detector(self.detecor_name)
        self.ransac = ransac

        self.tar_img = None
        self.tar_img_changed = True
        self.tar_pos = None  # np.ndarray (N, 2)
        self.tar_des_rot = None  # np.ndarray (N, D)
        self.kdtree = None

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

    def _get_detector(self, detector: str = "SIFT"):
        create = detector.strip().upper() + "_create"
        if not hasattr(cv2, create):
            print("[INFO] Cannot load function {} from module cv2".format(create))
            print("[INFO] Fallback to SIFT_create")
            self.detector_str = "SIFT"
            create = "SIFT_create"
        create = getattr(cv2, create)
        return create()
    
    def _extract_kp_des(self, image, mask=None):
        if isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
        kp, des = self.detector.detectAndCompute(image, mask)
        success = (kp is not None) and (des is not None)
        if des is not None:
            des = des.astype(np.float32)  # opencv FLANN use float32
            success = len(des) >= 4

        if not success:
            print("[INFO] Extract feature points and descriptors failed.")
            return None, None, False
        else:
            return kp, des, success

    def _build_tree(self):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        self.kdtree = cv2.flann_Index()
        # FLANN use float32
        self.kdtree.build(self.tar_des_rot, index_params)

    def _feature_match(self, cur_pos, cur_des):
        indices, dists = self.kdtree.knnSearch(
            cur_des, knn=2, params=dict(checks=50)
        )

        good_mask = dists[:, 0] < 0.75 * dists[:, 1]
        # good_mask = dists[:, 0] < 1.2 * dists[:, 1]
        tar_index = indices[good_mask, 0]
        cur_index = np.nonzero(good_mask)[0]
        matches = np.stack([tar_index, cur_index], axis=-1)

        if self.ransac and len(matches) > 4:
            tar_pts = self.tar_pos[tar_index]
            cur_pts = cur_pos[cur_index]

            if len(matches) == 4:
                M, mask = cv2.findHomography(tar_pts, cur_pts, cv2.RANSAC, 5.0)
            else:
                E, mask = cv2.findEssentialMat(tar_pts, cur_pts, self.intrinsic.K,
                    cv2.RANSAC, 0.999, 3.0)
            mask = mask.ravel().astype(bool)
            matches = np.ascontiguousarray(matches[mask])

        return matches

    def update_target_frame(self, image, mask=None):
        tar_pos = []
        tar_des_rot = []

        success_any = False
        for k in self.rot90s:
            if k == 0:
                image_rot = image
                mask_rot = mask
                inv_r = lambda x: x
            else:
                image_rot, _, inv_r = rot90(image, k=k)
                if mask is not None:
                    mask_rot, _, _, rot90(mask, k=k)
                else:
                    mask_rot = None
            
            tar_kp_r, tar_des_r, success = \
                self._extract_kp_des(image_rot, mask_rot)
            success_any = success_any or success
            
            if success:
                # inv rotated points
                tar_pos.append(inv_r(np.float32([kp.pt for kp in tar_kp_r])))
                tar_des_rot.append(tar_des_r)

        if success_any:
            self.tar_img = image
            self.tar_img_changed = True
            self.tar_des_rot = np.concatenate(tar_des_rot)
            self.tar_pos = np.concatenate(tar_pos)
            self._build_tree()

        return success_any

    def _align_cur_to_tar(self, cur_pos, matches):
        tar_indices, cur_indices = matches.T
        cur_pos_aligned = np.zeros_like(self.tar_pos)
        cur_pos_aligned[tar_indices] = cur_pos[cur_indices]
        valid_mask = np.zeros(self.tar_pos.shape[0], dtype=bool)
        valid_mask[tar_indices] = True
        return cur_pos_aligned, valid_mask

    def process_current_frame(self, image):
        if self.tar_des_rot is None:
            print("[INFO] Haven't initialize for target frame!")
            return None

        cur_kp, cur_des, success = self._extract_kp_des(image)
        if not success:
            return None

        cur_pos = np.float32([kp.pt for kp in cur_kp])  # (N, 2)
        matches = self._feature_match(cur_pos, cur_des)
        if len(matches) < 3:
            print("[INFO] Too few matches (={})".format(len(matches)))
            return None

        cur_pos_aligned, valid_mask = self._align_cur_to_tar(cur_pos, matches)

        corr = Correspondence(
            intrinsic=self.intrinsic, 
            tar_img=self.tar_img,
            tar_pos=self.tar_pos,

            cur_img=image,
            cur_pos=cur_pos,

            match=matches,
            valid_mask=valid_mask, 
            cur_pos_aligned=cur_pos_aligned,
            
            detector_name=self.detector_str, 
            tar_img_changed=self.tar_img_changed
        )
        # will be set to True when `update_target_frame` is called`
        self.tar_img_changed = False
        return corr


if __name__ == "__main__":
    import time
    from .utils import FrontendConcatWrapper, plot_corr

    intrinsic = CameraIntrinsic.default()
    # frontend = Classic(intrinsic, "ORB", rot90=2)

    frontend = FrontendConcatWrapper(
        [
            Classic(intrinsic, "AKAZE:0"),
            # Classic(intrinsic, "ORB:2"),
        ]
    )


    key = 'n'
    cap = cv2.VideoCapture(0)
    while True:
        if key == 'n':
            ret, img0 = cap.read()
            frontend.update_target_frame(img0)
        elif key == 'q':
            break
        
        ret, img1 = cap.read()
        if ret:
            # img1, _, _ = rot90(img1, k=0)

            corr = frontend.process_current_frame(img1)
            if corr:
                t0 = time.time()
                image = plot_corr(corr, show_keypoints=True)
                t1 = time.time()
                cv2.imshow("target | current", image)
                key = chr(cv2.waitKey(1) & 0xFF)
                t2 = time.time()
                print("[INFO] Matching time = {:.3f}s, vis time = {:.3f}s"
                    .format(t1-t0, t2-t1))

