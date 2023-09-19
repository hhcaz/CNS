import os
import glob
import numpy as np
import pybullet as p

from ..sim import environment as sim_env
from ..reimpl import environment as reimpl_env
from ..utils.perception import CameraIntrinsic, Camera
from .stat_scenes import stat_scenes


class BenchmarkEnvRender(sim_env.ImageEnvGUI):
    RETURN_IMAGE_FORMAT = "BGR"

    def __init__(self, scale=1.0, section="A"):
        if not hasattr(self, "client"):
            self.client = p.connect(p.GUI)
            print("[INFO] Client (id = {}) initialized".format(self.client))

        super().__init__(None, resample=True, auto_reinit=False)
        self.reinit(scale, section)
    
    def reinit(self, scale, section):
        self.scale = scale
        sections, self.scenes = stat_scenes()
        self.global_indices = sections[section].tolist()
        self.global_indices.sort()

        for scene in self.scenes:
            scene["target_wcT"] = np.array(scene["target_wcT"])
            scene["target_wcT"][:3, 3] *= scale

            scene["initial_wcT"] = np.array(scene["initial_wcT"])
            scene["initial_wcT"][:3, 3] *= scale

            scene["camera_nf"] = np.array(scene["camera_nf"]) * scale

            for obj in scene["objects"]:
                bias = 0 if obj["path"] == "plane.urdf" else 0.1
                obj["pos"] = (np.array(obj["pos"]) + [0, 0, bias]) * scale
                obj["scale"] *= scale

        # override original settings
        self.max_steps = 600
        self.dist_eps *= scale

    def skip_indices(self, indices=[]):
        residuals = set(self.global_indices) - set(indices)
        self.global_indices = list(residuals)

    def prefer_result_folder(self, prefix=None):
        folder = "scale={:.2f}".format(self.scale)
        if prefix is not None:
            folder = os.path.join(prefix, folder)
        return folder

    def prefer_result_fname(self, i):
        return "scene_gid={:0>4d}.npz".format(self.global_indices[i])

    @classmethod
    def get_global_indices_of_saved(self, folder: str):
        files = glob.glob(os.path.join(folder, "scene_gid=*"))
        global_indices = []
        for f in files:
            fname = os.path.split(f)[-1]
            fname_wo_ext = os.path.splitext(fname)[0]
            index = int(fname_wo_ext.replace("scene_gid=", ""))
            global_indices.append(index)
        return global_indices

    def __len__(self):
        return len(self.global_indices)

    def load_scene(self, i):
        p.resetSimulation(physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(1.5 * self.scale, 70, -50, [0, 0, 0], 
            physicsClientId=self.client)

        scene = self.scenes[self.global_indices[i]]
        self.target_wcT = scene["target_wcT"]
        self.initial_wcT = scene["initial_wcT"]
        self.current_wcT = self.initial_wcT.copy()

        intrinisc = CameraIntrinsic.from_dict(scene["camera_config"])
        self.camera = Camera(intrinisc, *scene["camera_nf"])

        self.plane_pos = (0, 0, 0)
        for obj in scene["objects"]:
            path = obj["path"] if obj["path"] == "plane.urdf" else os.path.join(
                self.YCB_ROOT, obj["path"])
            p.loadURDF(
                path, obj["pos"], obj["orn"], 
                globalScaling=obj["scale"], 
                physicsClientId=self.client
            )

            if obj["path"] == "plane.urdf":
                self.plane_pos = obj["pos"]
        
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        for _ in range(200):
            p.stepSimulation(physicsClientId=self.client)

    def init(self, i):
        self.load_scene(i)

        self.steps = 0
        target_cwT = np.linalg.inv(self.target_wcT)
        frame = self.camera.render(target_cwT, self.client)

        rgb = frame.color_image()
        self.tar_img = np.ascontiguousarray(rgb[:, :, ::-1])  # rgb to bgr

        # useless, set to zero
        self.points = np.zeros((1, 3))  # (N, 3)
        self.target_xy = np.zeros((1, 2))
        self.target_Z = np.ones(1)  # (N,)

        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
            self.pc.update(self.points)

        return self.tar_img.copy()

    def observation(self):
        current_cwT = np.linalg.inv(self.current_wcT)
        frame = self.camera.render(current_cwT, self.client)

        rgb = frame.color_image()
        cur_img = np.ascontiguousarray(rgb[:, :, ::-1])  # rgb to bgr

        if not self.debug_items_displayed:
            self.init_debug_items()

        return cur_img  # bgr image

    def camera_under_ground(self):
        current_z = self.current_wcT[2, 3]
        return current_z < self.plane_pos[-1]



class BenchmarkEnvAffine(reimpl_env.ImageEnvGUI):
    RETURN_IMAGE_FORMAT = "RGB"

    def __init__(self, scale=1.0, aug=False, one_image=None):
        self.client = p.connect(p.GUI)
        super().__init__(None, resample=True, auto_reinit=False, aug=aug)

        # basic configuration
        self.MAX_duz = 37  # degrees

        if one_image:
            self.USE_ONE_IMG = True
            self.ONE_IMG_PATH = one_image
        else:
            self.USE_ONE_IMG = False

        np.random.seed(2023)
        self.initial_poses = []
        self.target_poses = []
        for _ in range(150):
            self.sample_target_and_initial_pose()
            self.initial_poses.append(self.initial_wcT.copy())
            self.target_poses.append(self.target_wcT.copy())
        np.random.shuffle(self.image_files)

        self.global_indices = np.arange(len(self.target_poses)).tolist()
        self.scale = scale
        self.target_pose_r = [r*scale for r in self.target_pose_r]
        self.initial_pose_r = [r*scale for r in self.initial_pose_r]

        self.camera = Camera(
            self.camera.intrinsic,
            self.camera.near * scale,
            self.camera.far * scale
        )

        self.max_steps = 600
        self.dist_eps *= scale

    def skip_indices(self, indices=[]):
        residuals = set(self.global_indices) - set(indices)
        self.global_indices = list(residuals)

    def prefer_result_folder(self, prefix=None):
        folder = "scale={:.2f}".format(self.scale)
        if prefix is not None:
            folder = os.path.join(prefix, folder)
        return folder

    def prefer_result_fname(self, i):
        return "scene_gid={:0>4d}.npz".format(self.global_indices[i])

    @classmethod
    def get_global_indices_of_saved(self, folder: str):
        files = glob.glob(os.path.join(folder, "scene_gid=*"))
        global_indices = []
        for f in files:
            fname = os.path.split(f)[-1]
            fname_wo_ext = os.path.splitext(fname)[0]
            index = int(fname_wo_ext.replace("scene_gid=", ""))
            global_indices.append(index)
        return global_indices

    def __len__(self):
        return len(self.global_indices)

    def init(self, i) -> np.ndarray:
        self.initial_wcT = self.initial_poses[self.global_indices[i]].copy()
        self.target_wcT = self.target_poses[self.global_indices[i]].copy()
        self.initial_wcT[:3, 3] *= self.scale
        self.target_wcT[:3, 3] *= self.scale
        self.current_wcT = self.initial_wcT.copy()

        if self.USE_ONE_IMG:
            image_load = self.load_image(None)
        else:
            image_load = self.load_image(self.image_files[self.global_indices[i]])
        image_load_aug0 = self.aug_image(image_load) if self.aug else image_load
        image_target = self.get_warp_image(image_load_aug0, self.target_wcT)
        self.image_target = image_target

        image_load_aug1 = self.aug_image(image_load) if self.aug else image_load
        self.image_load = image_load_aug1
        self.steps = 0

        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
        
        return self.image_target[0].permute(1, 2, 0).cpu().numpy()

    def observation(self) -> np.ndarray:
        obs = super().observation()
        image_current = obs[1]
        return image_current[0].permute(1, 2, 0).cpu().numpy()  # rgb image
    
    def camera_under_ground(self):
        return self.current_wcT[2, 3] < 0

