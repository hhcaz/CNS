import cv2
import torch
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
# from .environment import ImageEnv, ImageEnvGUI, SynObjEnvGUI
from .environment_gau import ImageEnv, ImageEnvGUI, SynObjEnvGUI
from ..sim.supervisor import pbvs_straight


class Dataset(object):
    def __init__(self, camera_config, train=True, env="Image"):
        env = env.strip().lower()
        assert env in ["image", "imagegui", "synobj"], "Unknown environment."
        env_map = {
            "image": ImageEnv,
            "imagegui": ImageEnvGUI,
            "synobj": SynObjEnvGUI
        }
        self.env: ImageEnv = env_map[env](
            camera_config, resample=train, auto_reinit=False, aug=train)
        self.env.init()
        self.gt_vel = np.zeros(6)
    
    def mat2vec(self, T):
        u = R.from_matrix(T[:3, :3]).as_rotvec()
        t = T[:3, 3]
        vec = np.concatenate([t, u])
        return vec
    
    def get(self):
        current_wcT, image_current, target_wcT, image_target, \
            intrinsic = self.env.observation()

        # gt for icra2021
        gt_vel = pbvs_straight(current_wcT, target_wcT)
        self.gt_vel = gt_vel * 2
        gt_vel = torch.from_numpy(gt_vel).float()

        # gt for icra2018
        ref_pose = np.array([
            [1,  0,  0, 0], 
            [0, -1,  0, 0,],
            # [0,  0, -1, np.mean(self.env.target_pose_r)],
            [0,  0, -1, self.env.ref_pose_d],
            # 0.75 = (max_tar_pose_r + min_tar_pose_r) / 2. See environment.py for details
            [0,  0,  0, 1]
        ])  # top down view from middle tar_pose_r
        cur_rela_pose = np.linalg.inv(ref_pose) @ current_wcT
        tar_rela_pose = np.linalg.inv(ref_pose) @ target_wcT
        cur_rela_pose = torch.from_numpy(self.mat2vec(cur_rela_pose)).float()
        tar_rela_pose = torch.from_numpy(self.mat2vec(tar_rela_pose)).float()
        gt_rela_poses = torch.stack([cur_rela_pose, tar_rela_pose], dim=0)  # (2, 6)

        if isinstance(self.env, ImageEnvGUI):
            img_cur_np = image_current[0].permute(1, 2, 0).cpu().numpy()
            img_tar_np = image_target[0].permute(1, 2, 0).cpu().numpy()

            image = np.concatenate([img_tar_np, img_cur_np], axis=1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("target | current", image)
            cv2.waitKey(1)

        return image_current, image_target, gt_vel, gt_rela_poses

    def feedback(self, vel, norm=True):
        if isinstance(vel, torch.Tensor):
            vel = vel.detach().cpu().numpy()
        if norm:
            v, w = vel[:3], vel[3:]
            gt_v, gt_w = self.gt_vel[:3], self.gt_vel[3:]

            v = v / (np.linalg.norm(v) + 1e-8) * np.linalg.norm(gt_v)
            w = w / (np.linalg.norm(w) + 1e-8) * np.linalg.norm(gt_w)
            vel = np.concatenate([v, w])
        self.env.action(vel)


class DataLoader(object):
    def __init__(self, camera_config, batch_size, train=True, num_trajs=100, env="Image"):
        self.batch_size = batch_size
        if not train:
            np.random.seed(2022)
        self.datasets: List[Dataset] = [
            Dataset(camera_config, train, env="Image") for _ in range(batch_size - 1)]
        # only the last dataset can be set to PointGUI or ImageGUI
        self.datasets.append(Dataset(camera_config, train, env=env))
        self.num_samples = int(num_trajs) * self.datasets[0].env.max_steps
        self.num_batches = int(float(self.num_samples) / batch_size)
        self.num_batches = max(self.num_batches, 1)
        self.current_batch = 0
        self.train = train
    
    def get(self):
        data = [d.get() for d in self.datasets]
        image_current, image_target, gt_vel, gt_poses = zip(*data)
        image_current = torch.cat(image_current, dim=0)
        image_target = torch.cat(image_target, dim=0)
        gt_vel = torch.stack(gt_vel, dim=0)
        gt_poses = torch.stack(gt_poses, dim=0)
        return image_current, image_target, gt_vel, gt_poses
    
    def feedback(self, vel, norm=True):
        if isinstance(vel, torch.Tensor):
            vel = vel.detach().cpu().numpy()
        for i, d in enumerate(self.datasets):
            if d.env.abnormal_pose():
                d.feedback(d.gt_vel, norm=False)
            else:
                d.feedback(vel[i], norm)
    
    def resample_pose(self):
        for d in self.datasets:
            # d.env.initial_wcT = d.env.sample_initial_pose()
            # d.env.refine_initial_pose()
            d.env.initial_wcT = d.env.sample_initial_pose(d.env.target_wcT)
            d.env.current_wcT = d.env.initial_wcT
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.current_batch == 0) and (not self.train):
            for dataset in self.datasets:
                if dataset.env.steps > 0:
                    dataset.env.init()

        self.current_batch += 1
        if self.current_batch > self.num_batches:
            self.current_batch = 0

            gui_dataset = self.datasets[-1]
            if isinstance(gui_dataset.env, ImageEnvGUI):
                gui_dataset.env.clear_debug_items()

            raise StopIteration()

        for d in self.datasets:
            if d.env.need_reinit():
                d.env.init()

        return self.get()


if __name__ == "__main__":
    import time
    import pybullet as p

    p.connect(p.GUI_SERVER)
    # dataloader = DataLoader("setup.json", batch_size=1, train=True, num_trajs=100, env="ImageGUI")
    dataloader = DataLoader(None, batch_size=1, train=True, num_trajs=100, env="ImageGUI")
    for data in dataloader:
        image_current, image_target, gt_vel, gt_poses = data
        print(image_current.size(), image_target.size(), gt_vel.size())
        print("rela poses:")
        print(gt_poses.numpy())
        time.sleep(dataloader.datasets[-1].env.dt)
        dataloader.feedback(gt_vel * 2, norm=False)

