import cv2
import time
import json
import torch
import numpy as np
from enum import Flag, auto
from ..midend.corr2graph import Midend
from ..frontend.classic import Classic
try:
    from ..frontend.superglue import SuperGlue
except Exception as e:
    print("[INFO] SuperGlue not available. If you want to use it, "
          "please follow instructions in README.md to install it.")
from ..frontend.utils import FrontendConcatWrapper, show_corr

from ..utils.perception import CameraIntrinsic
from ..utils.visualize import show_graph, show_keypoints
from .controller import GraphVSController, IBVSController, ImageVSController
from ..ablation.cluster.train_graph_vs_no_cluster import GraphVS_NoCluster
from ..ablation.cluster.graph_gen_no_cluster import GraphGeneratorNoCluster


class VisOpt(Flag):
    NO = 0
    KP = auto()
    MATCH = auto()
    GRAPH = auto()
    ALL = KP | MATCH | GRAPH


def get_frontend(intrinsic: CameraIntrinsic, detectors: str, ransac=True):
    """
    - 'AKAZE:02+ORB:13' means AKAZE with rots=[0,2] and ORB with rots=[1,3]
    """
    frontends = []
    for config in detectors.split("+"):
        config = config.strip()
        if config.lower().startswith("superglue"):
            frontends.append(SuperGlue(intrinsic, config, ransac=ransac))
        else:
            frontends.append(Classic(intrinsic, config, ransac=ransac))
    
    if len(frontends) == 0:
        return frontends[0]
    else:
        return FrontendConcatWrapper(frontends)


class CorrespondenceBasedPipeline(object):
    REQUIRE_IMAGE_FORMAT = "BGR"

    def __init__(
        self, 
        detector: str, 
        ckpt_path: str, 
        intrinsic: CameraIntrinsic, 
        device="cuda:0", 
        ransac=True, 
        vis=VisOpt.NO
    ):
        self.device = torch.device(device)
        self.vis = vis
        
        self.frontend = get_frontend(intrinsic, detector, ransac)
        self.midend = Midend()

        if ckpt_path.endswith(".json"):
            self.control = IBVSController(ckpt_path)
        else:
            self.control = GraphVSController(ckpt_path, self.device)
            if isinstance(self.control.net, GraphVS_NoCluster):  # special case
                self.midend.graph_generator_class = GraphGeneratorNoCluster

        self.new_scene = True
        self.dist_scale = 1.0
        self.target_kwargs = dict()
    
    @classmethod
    def from_file(cls, config_path: str):
        with open(config_path, "r") as fp:
            config: dict = json.load(fp)
        
        intrinsic: dict = config.get("intrinsic", None)
        if intrinsic is None:
            intrinsic = CameraIntrinsic.default()
        else:
            intrinsic = CameraIntrinsic.from_dict(intrinsic)
        
        detector: str = config.get("detector", "SIFT")
        ckpt_path: str = config["checkpoint"]
        device: str = config.get("device", 
            "cuda:0" if torch.cuda.is_available() else "cpu")
        ransac = config.get("ransac", True)
        vis_opts: str = config.get("visualize", "NO").split("|")
        vis = getattr(VisOpt, vis_opts[0])
        if len(vis_opts) > 1:
            for opt in vis_opts[1:]:
                vis = vis | getattr(VisOpt, opt)
        return cls(detector, ckpt_path, intrinsic, device, ransac, vis)
    
    def set_target(self, image: np.ndarray, dist_scale: float, **kwargs):
        """
        Arguments
        - image: np.ndarray, bgr image (Note: here is bgr not rgb)
        - dist_scale: scalar representing the distance from camera to scene center
        """
        success = self.frontend.update_target_frame(image)
        self.new_scene = True
        self.dist_scale = dist_scale
        self.target_kwargs = kwargs
        return success
    
    def get_control_rate(self, image: np.ndarray):
        """
        Arguments:
        - image: np.ndarray, bgr image (Note: here is bgr not rgb)
        
        Returns:
        - vel: np.ndarray, shape=(6,), [vx, vy, vz, wx, wy, wz], 
            camera velocity in camera coordinate
        - data: a GraphData object for network inference or visualization, 
            would be None if no matched keypoints are detected
        - timing: dict, keys = ["frontend_time", "midend_time", 
            "backend_time", "total_time"]
        """
        timing = {
            "frontend_time": 0,
            "midend_time": 0, 
            "backend_time": 0,
            "total_time": 0
        }

        t0 = time.time()
        corr = self.frontend.process_current_frame(image)
        timing["frontend_time"] = time.time() - t0

        if corr is None:
            vel = np.zeros(6, dtype=np.float32)
            data = None
        else:
            t0 = time.time()
            data = self.midend.get_graph_data(corr)
            timing["midend_time"] = time.time() - t0

            if self.new_scene:
                data.start_new_scene()
            data.set_distance_scale(self.dist_scale)
            self.new_scene = False

            for k, v in self.target_kwargs.items():
                setattr(data, k, v)
            # add intrinsic
            setattr(data, "intrinsic", self.frontend.intrinsic)

            if self.vis & VisOpt.KP:
                show_keypoints(data)
            if self.vis & VisOpt.GRAPH:
                show_graph(data)
            if self.vis & VisOpt.MATCH:
                show_corr(corr, show_keypoints=True)

            t0 = time.time()
            vel = self.control(data)
            timing["backend_time"] = time.time() - t0

            # add numpy image for SSIM stop criterion afterwards
            setattr(data, "cur_img", image)
            setattr(data, "tar_img", corr.tar_img)

        timing["total_time"] = timing["frontend_time"] + timing["backend_time"]
        return vel, data, timing


class ImageBasedPipeline(object):
    REQUIRE_IMAGE_FORMAT = "RGB"

    def __init__(self, ckpt_path: str, device="cuda:0", vis=VisOpt.NO):
        self.device = torch.device(device)
        self.vis = vis

        self.control = ImageVSController(ckpt_path, device)
        self.tar_img_torch = None
        self.tar_img_np = None

        self.new_scene = True
        self.target_kwargs = dict()
    
    def set_target(self, image: np.ndarray, **kwargs):
        """
        Arguments
        - image: np.ndarray, rgb image (Note: here is rgb not bgr)
        - dist_scale: scalar representing the distance from camera to scene center
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        self.tar_img_np = image
        self.tar_img_torch = (
            torch.from_numpy(image).float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )  # (1, 3, H, W)
        self.new_scene = True
        self.target_kwargs = kwargs
    
    def get_control_rate(self, image):
        """
        - image: np.ndarray, rgb image (Note: here is rgb not bgr)
        """
        timing = {
            "total_time": 0
        }

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        cur_img_np = image
        cur_img_torch = (
            torch.from_numpy(image).float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )  # (1, 3, H, W)

        data = {
            "cur_img": cur_img_torch,
            "tar_img": self.tar_img_torch,
            "new_scene": self.new_scene
        }
        data.update(self.target_kwargs)

        if self.vis & VisOpt.ALL:
            tc_image = np.concatenate([self.tar_img_np, cur_img_np], axis=1)
            tc_image = cv2.cvtColor(tc_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("target | current", tc_image)
            cv2.waitKey(1)

        t0 = time.time()
        vel = self.control(data)
        timing["total_time"] = time.time() - t0
        self.new_scene = False

        data_for_stop = {"cur_img": cur_img_np, "tar_img": self.tar_img_np}

        return vel, data_for_stop, timing
