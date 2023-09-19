import os
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import cv2
import glob
import json
import time
import torch
import random
import skimage.io
import skimage.data
import numpy as np
import kornia as K
import kornia.augmentation as A
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from collections import namedtuple

from ..utils.perception import Camera, CameraIntrinsic
from ..utils.image_transform import scale_to_fit, pad_to_ratio
from ..sim.sampling import uniform_ball_sample, sample_camera_pose
from ..sim.environment import DebugAxes


class ImageEnv(object):
    ADD_OCCLU = False
    USE_ONE_IMG = True
    MAX_duz = 37  # degrees
    VOC_ROOT_CANDIDATES = [
        "E:/datasets/PASCAL_VOC_2012/VOC2012",
        "/mnt/data/chenaz/datasets/VOC2012",
    ]
    stars1953 = os.path.join(os.path.dirname(__file__), "stars1953.jpg")
    lenna = os.path.join(os.path.dirname(__file__), "Lenna.jpg")
    ONE_IMG_PATH = stars1953

    def __init__(self, camera_config, resample=True, auto_reinit=True, verbose=False, aug=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if camera_config is None:
            camera_intrinsic = CameraIntrinsic.default()
        else:
            with open(camera_config, "r") as j:
                config = json.load(j)
            camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        self.camera = Camera(camera_intrinsic)

        for VOC_ROOT in self.VOC_ROOT_CANDIDATES:
            if os.path.exists(VOC_ROOT):
                break
        else:
            # raise FileNotFoundError("Cannot find VOC root path.")
            print("[WARN] Cannot find VOC root path.")
        
        self.image_files = glob.glob(os.path.join(VOC_ROOT, "JPEGImages/*.jpg"))
        self.occlu_files = glob.glob(os.path.join(VOC_ROOT, "SegmentationObject/*.png"))

        self.obs_r = 0.2
        self.obs_h = 0.1

        self.target_pose_r = [0.6, 0.9]
        self.target_pose_phi = [70, 90]
        self.target_pose_drz_max = 15
        self.target_pose_dry_max = 1e-7
        self.target_pose_drx_max = 1e-7

        self.ref_pose_d = np.mean(self.target_pose_r)
        self.pos_std = self.ref_pose_d / 20
        # self.drxy_max = 10
        self.drxy_std = 10
        self.drz_std = 20
        
        self.ref_pose = np.array([
            [1,  0,  0, 0], 
            [0, -1,  0, 0,],
            [0,  0, -1, self.ref_pose_d],
            # 0.75 = (max_tar_pose_r + min_tar_pose_r) / 2. See environment.py for details
            [0,  0,  0, 1]
        ])

        self.dt = 1./50.  # 50Hz
        self.angle_eps = 1  # degree
        self.dist_eps = 0.002  # m
        self.max_steps = 200

        self.resample = resample
        self.auto_reinit = auto_reinit
        self.verbose = verbose
        self.aug = aug

        # state
        self.image_load = None
        self.image_target = None

        self.target_wcT = None
        self.initial_wcT = None
        self.current_wcT = None
        self.steps = 0

    # def refine_initial_pose(self):
    #     dT = np.linalg.inv(self.initial_wcT) @ self.target_wcT
    #     duz = R.from_matrix(dT[:3, :3]).as_rotvec()[-1]

    #     max_duz = self.MAX_duz / 180 * np.pi
    #     if np.abs(duz) > max_duz:
    #         if duz >= 0:
    #             compensate_uz = duz - max_duz
    #         else:
    #             compensate_uz = duz + max_duz
        
    #         compenstate_dT = np.eye(4)
    #         compenstate_dT[:3, :3] = R.from_rotvec(np.array([0., 0., compensate_uz])).as_matrix()
    #         self.initial_wcT = self.initial_wcT @ compenstate_dT
    #         # print("[INFO] Refine initial pose triggered")
    
    def get_warp_image(self, image_load: torch.Tensor, wcT, fill=torch.zeros(3)):
        # assume the loaded image is from the top down view at [0, 0, max_pose_r * 1.2]
        axisx = np.array([1, 0, 0])
        axisy = np.array([0, -1, 0])
        axisz = np.array([0, 0, -1])
        top_down_wcT = np.eye(4)
        top_down_wcT[:3, :3] = np.stack([axisx, axisy, axisz], axis=-1)
        # top_down_wcT[:3, 3] = np.array([0, 0, self.ref_pose_d * 1.2])
        top_down_wcT[:3, 3] = np.array([0, 0, self.ref_pose_d * 1.5])

        H, W = self.camera.intrinsic.height, self.camera.intrinsic.width
        B, C, H_, W_ = image_load.size()
        assert (H == H_) and (W == W_), "image size not consistent with camera intrinsic"

        image_pts_load = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]])
        space_pts_load = self.camera.inv_project(
            extrinsic=np.linalg.inv(top_down_wcT),
            uv=image_pts_load,
            Z=np.array([np.abs(top_down_wcT[2, 3])] * len(image_pts_load))
        )
        image_pts_query = self.camera.project(np.linalg.inv(wcT), space_pts_load)

        points_src = torch.from_numpy(image_pts_load).float().view(1, 4, 2)  # (1, 4, 2)
        points_dst = torch.from_numpy(image_pts_query).float().view(1, 4, 2)
        M = K.geometry.get_perspective_transform(points_src, points_dst).to(self.device)
        # image_query = K.geometry.warp_perspective(image_load, M, dsize=(H, W), fill_value=fill)
        image_query = K.geometry.warp_perspective(image_load, M, dsize=(H, W), padding_mode="fill", fill_value=fill)

        return image_query

    def load_image(self, path):
        if self.USE_ONE_IMG and hasattr(self, "one_img"):
            image = self.one_img
        else:
            if self.USE_ONE_IMG:
                image: np.ndarray = skimage.io.imread(self.ONE_IMG_PATH)
            else:
                image: np.ndarray = skimage.io.imread(path)

            H, W = self.camera.intrinsic.height, self.camera.intrinsic.width
            im_h, im_w = image.shape[:2]
            # rotate the source image if necessary to maximize the usage of image area
            if (H - W) * (im_h - im_w) < 0:
                image = np.rot90(image, k=1, axes=(0, 1))

            image, _, _ = scale_to_fit(image, box_wh=(W, H))
            image, _, _ = pad_to_ratio(image, wh_ratio=float(W)/float(H))

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(image)
            # plt.show()
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.
            image: torch.Tensor = K.image_to_tensor(image, keepdim=False).float().to(self.device)

            # ensure 3 color channels
            if image.size(1) == 1:
                image = image.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
            if image.size(1) == 4:
                image = image[:, :3].contiguous()  # (B, 4, H, W) -> (B, 3, H, W)
        
        if self.USE_ONE_IMG:
            self.one_img = image
        return image

    def load_occlusion(self, path: str, image_hw: tuple):
        """
        Arguments:
        - path: image segmentation label path
        - image_hw: (h, w), size of canvas image

        - Returns:
        - (rr, cc): (2, N), indices for canvas image indicating where to put occulsions
        - values: (C, N), values of occlusions, dtype=float32
        """
        
        occlu_label: np.ndarray = skimage.io.imread(path)
        occlu_image: np.ndarray = skimage.io.imread(
            path.replace("SegmentationObject", "JPEGImages").replace(".png", ".jpg"))
        if occlu_image.ndim == 2:
            occlu_image = np.stack([occlu_image]*3, dim=-1)  # (H, W) -> (H, W, 3)
        if occlu_image.ndim == 3 and occlu_image.shape[-1] == 1:
            occlu_image = np.concatenate([occlu_image]*3, dim=-1)  # (H, W, 1) -> (H, W, 3)
        if occlu_image.shape[-1] == 4:
            occlu_image = np.ascontiguousarray(occlu_image[:, :, :3])  # (H, W, 4) -> (H, W, 3)
        
        # filter the background label
        occlu_h, occlu_w = occlu_label.shape[:2]
        labels = np.unique(occlu_label.reshape(occlu_h*occlu_w, -1), axis=0)
        mask = ~((labels == np.array([0, 0, 0, 255])).all(axis=-1) | 
            (labels == np.array([224, 224, 192, 255])).all(axis=-1))
        labels = labels[mask]

        # random pick one label
        label = random.choice(labels)
        mask = (occlu_label == label).all(axis=-1)
        rr, cc = np.nonzero(mask)

        # crop out the object of selected label
        crop_h_start, crop_h_end = rr.min(), rr.max() + 1
        crop_w_start, crop_w_end = cc.min(), cc.max() + 1
        occlu_image = np.ascontiguousarray(
            occlu_image[crop_h_start:crop_h_end, crop_w_start:crop_w_end])
        mask = np.ascontiguousarray(
            mask[crop_h_start:crop_h_end, crop_w_start:crop_w_end])

        # scale the selected region to 30% if image size
        box_size = int(min(image_hw) * 0.3)
        occlu_image, _, _ = scale_to_fit(occlu_image, (box_size, box_size), minimize=True)
        mask, _, _ = scale_to_fit(mask, (box_size, box_size), order=0, minimize=True)

        if occlu_image.dtype == np.uint8:
            occlu_image = occlu_image.astype(np.float32) / 255.
        mask = mask > 0.5
        rr, cc = np.nonzero(mask)
        values = torch.from_numpy(occlu_image[mask]).float().permute(1, 0)  # (N, C) -> (C, N)
        values = values.to(self.device)

        image_h, image_w = image_hw
        offset_h = random.randint(0, image_h - mask.shape[0])
        offset_w = random.randint(0, image_w - mask.shape[1])

        rr = torch.from_numpy(rr + offset_h).long()
        cc = torch.from_numpy(cc + offset_w).long()

        return (rr, cc), values

    def aug_image(self, image: torch.Tensor, add_occlu=True):
        B, C, H, W = image.size()

        if add_occlu:
            # add occlusion
            occlu_label_file = random.choice(self.occlu_files)
            (rr, cc), values = self.load_occlusion(occlu_label_file, (H, W))
            
            image = image.detach().clone()  # (B, C, H, W), B = 0
            image[0, :, rr, cc] = values

        augs = [
            # A.RandomPlanckianJitter("blackbody", same_on_batch=False, keepdim=False, p=1.0),
            A.RandomPlasmaContrast(roughness=(0.1, 0.7), same_on_batch=False, keepdim=False, p=0.9)
        ]

        for aug in augs:
            image = aug(image)
        return image

    def sample_target_and_initial_pose(self):
        while True:
            self.target_wcT = self.sample_target_pose()
            self.initial_wcT = self.sample_initial_pose(self.target_wcT)
            self.current_wcT = self.initial_wcT.copy()
            # print(self.target_wcT, self.initial_wcT)
            if not self.abnormal_pose():
                break

    def init(self):
        if self.resample or (self.image_target is None):
            self.sample_target_and_initial_pose()
            
            while True:
                file_path = random.choice(self.image_files)
                image_load = self.load_image(file_path)
                # if image_load.mean() > 0.1:
                #     break
                break

            image_load_aug0 = self.aug_image(image_load, self.ADD_OCCLU) if self.aug else image_load.clone()
            image_target = self.get_warp_image(image_load_aug0, self.target_wcT, fill=torch.zeros(3))
            self.image_target = image_target

            # image_load_aug1 = self.aug_image(image_load, self.ADD_OCCLU) if self.aug else image_load
            # self.image_load = image_load_aug1
            self.image_load = image_load.clone()

        self.current_wcT = self.initial_wcT.copy()
        self.steps = 0

    def observation(self):
        fill = torch.rand(3) if self.aug else torch.zeros(3)
        # print("fill = {}".format(fill.cpu().numpy()))
        image_current = self.get_warp_image(self.image_load, self.current_wcT, fill=fill)
        if self.aug:
            image_current = self.aug_image(image_current, add_occlu=False)

        return (
            self.current_wcT, image_current,
            self.target_wcT, self.image_target,
            self.camera.intrinsic
        )

    def action(self, vel):
        dT = np.eye(4)
        dT[:3, :3] = R.from_rotvec(vel[3:] * self.dt).as_matrix()
        dT[:3, 3] = vel[:3] * self.dt
        self.current_wcT = self.current_wcT @ dT
        self.current_wcT[:3, :3] = R.from_matrix(self.current_wcT[:3, :3]).as_matrix()
        self.steps += 1

        if self.auto_reinit and self.need_reinit():
            self.init()
    
    def sample_target_pose(self):
        # return sample_camera_pose(
        #     r_min=self.target_pose_r[0],
        #     r_max=self.target_pose_r[1],
        #     phi_min=self.target_pose_phi[0],
        #     phi_max=self.target_pose_phi[1],
        #     drz_max=self.target_pose_drz_max,
        #     dry_max=self.target_pose_dry_max,
        #     drx_max=self.target_pose_drx_max
        # )
        return self.ref_pose.copy()
    
    def sample_initial_pose(self, ref_pose=None):
        if ref_pose is None:
            ref_pose = self.ref_pose

        drz = np.random.randn() * self.drz_std
        # dry = np.random.uniform(-self.drxy_max, self.drxy_max)
        # drx = np.random.uniform(-self.drxy_max, self.drxy_max)
        dry = np.random.randn() * self.drxy_std
        drx = np.random.randn() * self.drxy_std

        u = np.array([drx, dry, drz]) / 180.0 * np.pi
        dR = R.from_rotvec(u).as_matrix()
        dtxyz = np.random.randn(3) * self.pos_std

        dT = np.eye(4)
        dT[:3, :3] = dR
        dT[:3, 3] = dtxyz
        ini_wcT = ref_pose.copy() @ dT
        return ini_wcT
    
    def close_enough(self, T0, T1):
        dT = np.linalg.inv(T0) @ T1
        angle = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()) / np.pi * 180
        dist = np.linalg.norm(dT[:3, 3])
        return (angle < self.angle_eps) and (dist < self.dist_eps)
    
    def close_enough_(self):
        return self.close_enough(self.target_wcT, self.current_wcT)

    def exceeds_maximum_steps(self):
        return self.steps > self.max_steps

    def abnormal_pose(self):
        pos = self.current_wcT[:3, 3]
        if pos[-1] < self.ref_pose_d * 0.1:
            if self.verbose:
                print("[INFO] Camera under ground")
            return True

        dist = np.linalg.norm(pos)
        if dist > self.ref_pose_d * 2:
            if self.verbose:
                print("[INFO] Too far away")
            return True
        
        zaxis = self.current_wcT[:3, 2]
        # if (-zaxis[-1]) < np.cos((90 - min(self.initial_pose_phi) * 0.8) / 180 * np.pi):
        if (-zaxis[-1]) < 0.1:
            if self.verbose:
                print("[INFO] Camera z axis almost looks up")
            return True
        
        x0, y0, z0 = pos
        a, b, c = zaxis
        c = c * np.abs(c) / (np.abs(c) + 1e-8)  # avoid div 0
        x = (0-z0)/c*a + x0
        y = (0-z0)/c*b + y0
        if np.sqrt(x**2 + y**2) > self.ref_pose_d:
            print("[INFO] May raise feature loss")
            return True
        
        v2p = -pos / (np.linalg.norm(pos) + 1e-8)  # vec pointing from pos to O
        if np.dot(v2p, zaxis) < 0.2:
            print("[INFO] May raise feature loss")
            return True

        return False

    def need_reinit(self):
        # if self.close_enough(self.target_wcT, self.current_wcT):
        #     return True
        
        if self.exceeds_maximum_steps():
            if self.verbose:
                print("[INFO] Overceed maximum steps")
            return True
        
        if self.abnormal_pose():
            return True
        
        return False
    
    def print_pose_err(self):
        # calculate servo error and print
        dT = np.linalg.inv(self.current_wcT) @ self.target_wcT
        du = R.from_matrix(dT[:3, :3]).as_rotvec()
        dt = dT[:3, 3]
        print("[INFO] Servo error: |du| = {:.3f} degree, |dt| = {:.3f} mm"
            .format(np.linalg.norm(du)/np.pi*180, np.linalg.norm(dt)*1000))


class ImageEnvGUI(ImageEnv):
    def __init__(self, camera_config, resample=True, auto_reinit=True, verbose=False, aug=False):
        super().__init__(camera_config, resample, auto_reinit, verbose=True, aug=aug)
        if not hasattr(self, "client"):
            self.client = p.connect(p.SHARED_MEMORY)
            if self.client < 0:
                self.client = p.connect(p.GUI_SERVER)
            assert self.client >= 0, "Cannot connect Pybullet"
        
        print("[INFO] Client (id = {}) connected".format(self.client))
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(1.674, 70, -50.8, [0, 0, 0], physicsClientId=self.client)

        self.axes_cam_tar = DebugAxes(self.client)
        self.axes_cam_cur = DebugAxes(self.client)
        self.debug_items_displayed = False

    def init(self):
        super().init()
        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
    
    def init_debug_items(self):
        self.axes_cam_tar.update(self.target_wcT)
        self.axes_cam_cur.update(self.current_wcT)
        self.debug_items_displayed = True
        print("[INFO] Debug items initialzied for client {}".format(self.client))
    
    def clear_debug_items(self):
        self.axes_cam_tar.clear()
        self.axes_cam_cur.clear()
        self.debug_items_displayed = False
        print("[INFO] Debug items cleared for client {}".format(self.client))
    
    def observation(self):
        obs = super().observation()
        if not self.debug_items_displayed:
            self.init_debug_items()
        return obs
    
    def action(self, vel):
        super().action(vel)
        self.axes_cam_cur.update(self.current_wcT)


class SynObjEnvGUI(ImageEnvGUI):
    SceneObj = namedtuple("SceneObj", ["path", "scale", "pos", "orn"])
    YCB_ROOT = os.path.join(os.path.dirname(__file__), 
        "..", "thirdparty", "pybullet-object-models", "pybullet_object_models", "ycb_objects")

    def __init__(self, camera_config, resample=True, auto_reinit=True, verbose=False, aug=False):
        super().__init__(camera_config, resample, auto_reinit, verbose=True, aug=False)

        self.initial_pose_r = self.target_pose_r
        self.initial_pose_phi = self.target_pose_phi
        self.initial_pose_drz_max = self.target_pose_drz_max
        self.initial_pose_dry_max = self.target_pose_dry_max
        self.initial_pose_drx_max = self.target_pose_drx_max

        if not hasattr(self, "client"):
            self.client = p.connect(p.SHARED_MEMORY)
            print("[INFO] Client (id = {}) initialized".format(self.client))
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        models = glob.glob(os.path.join(self.YCB_ROOT, "*", "model.urdf"))
        models.sort()
        self.candidates = [(m, 1.0) for m in models]

        self.obj_ids = []
        self.objs = []

    def gen_scenes(self, num_objs, obs_r, obs_h):
        """Generate real objects randomly in a cylinder.

        Arguments:
        - num_objs: number of objects
        - obs_r: radius of observation space
        - obs_h: height of observation space
        """
        self.obj_ids.clear()
        self.objs.clear()
        self.clear_debug_items()

        p.resetSimulation(physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(1.674, 70, -50.8, [0, 0, 0], physicsClientId=self.client)
        p.loadURDF('plane.urdf', [0, 0, -self.obs_h*2], [0, 0, 0, 1], 
            globalScaling=1, physicsClientId=self.client)

        num_objs = np.clip(num_objs, 1, None)
        # uniformly distributed in a horizontal circle
        position = uniform_ball_sample(num_objs, 2) * obs_r
        # uniformly distributed vertically
        position = np.concatenate([position, (np.random.rand(num_objs, 1)-0.5)*obs_h], axis=-1)  # (N, 3)
        # overvall, uniformly distributed in a cyclinder with r=obs_r and h=obs_h

        for i in range(num_objs):
            rpy = np.random.uniform(-np.pi, np.pi, size=3)  # random pose
            orn = p.getQuaternionFromEuler(rpy)
            model, scale = self.candidates[np.random.choice(len(self.candidates))]
            obj_id = p.loadURDF(model, position[i], orn, globalScaling=scale, physicsClientId=self.client)
            print("[INFO] loaded id = {}, loaded model: {}".format(obj_id, model))
            self.obj_ids.append(obj_id)
            self.objs.append(
                self.SceneObj(path=model, scale=scale, pos=position[i], orn=orn))

    def init(self):
        if self.resample or (self.image_target is None):
            num_objs = 16
            self.gen_scenes(num_objs, self.obs_r, self.obs_h)

            while True:
                self.target_wcT = self.sample_target_pose()
                self.initial_wcT = self.sample_initial_pose()
                # if not self.close_enough(self.target_wcT, self.initial_wcT):
                #     break
                break
            
            target_cwT = np.linalg.inv(self.target_wcT)
            for _ in range(num_objs):
                p.stepSimulation(physicsClientId=self.client)
            frame = self.camera.render(target_cwT, self.client)

            rgb = frame.color_image()
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.
            rgb = K.image_to_tensor(rgb, keepdim=False).float()
            self.image_target = rgb

        self.current_wcT = self.initial_wcT.copy()
        self.steps = 0

        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
    
    def observation(self):
        current_cwT = np.linalg.inv(self.current_wcT)
        frame = self.camera.render(current_cwT, self.client)

        rgb = frame.color_image()
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.
        rgb = K.image_to_tensor(rgb, keepdim=False).float()

        if not self.debug_items_displayed:
            self.init_debug_items()

        return (
            self.current_wcT, rgb, 
            self.target_wcT, self.image_target, 
            self.camera.intrinsic
        )


if __name__ == "__main__":
    from cns.sim.supervisor import pbvs
    # p.connect(p.GUI_SERVER)

    # env = ImageEnvGUI("setup.json", resample=True, auto_reinit=False)
    env = ImageEnv(None, resample=True, auto_reinit=False, aug=True)
    env.init()

    input("[INFO] Press Enter to start: ")
    while True:
        current_wcT, current_image, target_wcT, target_image, intrinsic = env.observation()
        points = np.zeros((1, 3))

        # (C, H, W) -> (H, W, C)
        target_image = target_image[0].permute(1, 2, 0).cpu().numpy()
        current_image = current_image[0].permute(1, 2, 0).cpu().numpy()

        image = np.concatenate([target_image, current_image], axis=1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("target | current", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        vel = pbvs(current_wcT, target_wcT, points)
        vel = vel * 2

        print("[INFO] vel trans = {}, vel = {}".format(np.linalg.norm(vel[3:]), vel))
        env.action(vel)

        if env.need_reinit():
            input("[INFO] Reach goal! Use {} steps. Press Enter to reinit: ".format(env.steps))
            env.init()
            # input("[INFO] Press Enter to start: ")
        
        time.sleep(env.dt)

