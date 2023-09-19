import torch
import numpy as np

from enum import Flag, auto
from torch_geometric.data import Batch
from typing import List, Optional, Union

from cns.midend.graph_gen import GraphGenerator, GraphData
from cns.utils.visualize import show_keypoints, show_graph
from cns.utils.perception import CameraIntrinsic
from .environment import PointEnv, PointEnvGUI, ImageEnvGUI
from .supervisor import supervisor_vel, Policy
from .sampling import ObservationSampler


class ObsAug(Flag):
    NoAug = 0
    RandomMismatch = auto()  # 10% points are mismatched
    RandomDiscard = auto()  # Random discard 10% points with equal probability
    AreaShift = auto()  # AreaShift will also discard points, but not equal probability
    PosJitter = auto()  # Add noise to keypoint coordinates
    Default = RandomMismatch | AreaShift | PosJitter


class Dataset(object):
    def __init__(
        self, 
        camera_config: Optional[Union[str, CameraIntrinsic]], 
        train=True, 
        env="Point", 
        aug=ObsAug.Default,
        # supervisor=Policy.PBVS_Center_Straight 
        supervisor=Policy.PBVS_Straight
    ):

        env = env.strip().lower()
        assert env in ["point", "pointgui", "imagegui"], "Unknown environment."
        env_map = {
            "point": PointEnv,
            "pointgui": PointEnvGUI,
            "imagegui": ImageEnvGUI
        }
        self.env: PointEnv = env_map[env](camera_config, resample=train, auto_reinit=False)
        self.supervisor = supervisor
        
        self.gt_vel = np.zeros(6)
        self.train = train
        self.aug = aug  # Note: augmentations are effective only when `train`=True

        self.graph_gen_type = GraphGenerator  # add alternatives for GraphGeneratorNoCluster
        self.graph_gen = None
        self.obs_sampler = None
        
        self.x_tar = None
        self.pos_tar = None
        self.env.init()


    def get(self):
        reinited = self.env.steps == 0
        current_xy, current_Z, target_xy, target_Z, intrinsic, \
            current_wcT, target_wcT, points, remove_index = self.env.observation()
        
        if isinstance(remove_index, list): remove_index = np.array(remove_index, dtype=np.int64)
        if remove_index is None: remove_index = np.array([], dtype=np.int64)

        N = len(points)
        H, W = intrinsic.height, intrinsic.width
        fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy

        # convert to normalized camera plane
        x1, y1 = intrinsic.pixel_to_norm_camera_plane(current_xy).T  # (N,), (N,)
        x2, y2 = intrinsic.pixel_to_norm_camera_plane(target_xy).T  # (N,), (N,)

        if reinited or (self.graph_gen is None):
            self.x_tar = np.stack([x2, y2], axis=-1)
            self.pos_tar = np.stack([x2, y2], axis=-1)
            # self.graph_gen = GraphGenerator(self.pos_tar)
            self.graph_gen = self.graph_gen_type(self.pos_tar)
            self.obs_sampler = ObservationSampler(self.pos_tar, tau_max=5)

        # Feature points mismatch augmentation
        mismatch_mask = np.zeros(N, dtype=bool)
        rand_perm = np.random.permutation(N)
        if N >= 10 and self.train and (self.aug & ObsAug.RandomMismatch):
            # 10% points are mismatched
            # sel_index = np.random.choice(N, int(N*0.2), replace=False)
            sel_index = rand_perm[:int(N*0.1)]
            rand_perm = rand_perm[int(N*0.1):]

            num_sel = len(sel_index)
            rand_x1 = (np.random.uniform(0, W, size=num_sel) - cx) / fx
            rand_y1 = (np.random.uniform(0, H, size=num_sel) - cy) / fy
            x1[sel_index] = rand_x1
            y1[sel_index] = rand_y1
            mismatch_mask[sel_index] = True
        
        # random noise augmentation
        if self.train and (self.aug & ObsAug.PosJitter):
            noise_scale = 0.002
            x1 = x1 + noise_scale * np.random.uniform(-1, 1, size=x1.shape)
            y1 = y1 + noise_scale * np.random.uniform(-1, 1, size=y1.shape)

        x_cur = np.stack([x1, y1], axis=-1)  # (N, num_features)
        pos_cur = np.stack([x1, y1], axis=1)

        obs_weight = None
        # random drop current observations for training
        if remove_index.size:
            pass  # pass if remove index is specified
        elif (N >= 64) and self.train and (self.aug & ObsAug.AreaShift):
            remove_index, obs_weight = self.obs_sampler(
                time=self.env.dt*self.env.steps, 
                current_wcT=current_wcT,
                dist_scale=np.mean(self.env.target_pose_r)
            )
        elif (N >= 10) and self.train and (self.aug & ObsAug.RandomDiscard):
            # 10% points are missing
            # note: missing points and mismatched points are not overlapped
            remove_index = rand_perm[:int(N*0.1)]
            rand_perm = rand_perm[int(N*0.1):]
        
        # if too much points are removed, preserve some
        if len(remove_index) >= 0.8 * N:
            remove_index = remove_index[:int(0.8*N)]
        # set zero features for missing nodes
        x_cur[remove_index] = 0
        pos_cur[remove_index] = 0
        
        data: GraphData = self.graph_gen.get_data(
            current_points=pos_cur,
            missing_node_indices=remove_index,
            mismatch_mask=mismatch_mask
        )

        if obs_weight is not None:
            obs_weight = obs_weight / (np.sum(obs_weight) + 1e-7)  # (N,)
            pbvs_points = np.sum(points * obs_weight[:, None], axis=0, keepdims=True)  # (1, 3)
        else:
            pbvs_points = points[getattr(data, "node_mask").numpy()]

        # pbvs_points = points
        # pbvs_points = points[getattr(data, "node_mask").numpy()]

        vel, (tPo_norm, vel_si) = supervisor_vel(
            self.supervisor,
            current_xy, current_Z, target_xy, target_Z, intrinsic, 
            current_wcT, target_wcT, pbvs_points)
        self.gt_vel = vel.copy()

        # append extra data for training
        if reinited:
            data.start_new_scene()
        data.set_distance_scale(tPo_norm)
        setattr(data, "vel", torch.from_numpy(vel[None, :]).float())        # (1, 6)
        setattr(data, "vel_si", torch.from_numpy(vel_si[None, :]).float())  # (1, 6)

        # append extra data for visualizaiton
        setattr(data, "intrinsic", self.env.camera.intrinsic)
        setattr(data, "walker_centers", self.obs_sampler.walker_centers if self.train else None)

        if isinstance(self.env, PointEnvGUI):
            show_keypoints(data)
            show_graph(data)

        return data

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
    def __init__(self, camera_config, batch_size, train=True, num_trajs=100, env="Point"):
        self.batch_size = batch_size
        if not train:
            np.random.seed(2022)
        self.datasets: List[Dataset] = [
            Dataset(camera_config, train, env="Point") for _ in range(batch_size - 1)]
        # only the last dataset can be set to PointGUI or ImageGUI
        self.datasets.append(Dataset(camera_config, train, env=env, aug=ObsAug.Default))
        self.num_samples = int(num_trajs) * self.datasets[0].env.max_steps
        self.num_batches = int(float(self.num_samples) / batch_size)
        self.num_batches = max(self.num_batches, 1)
        self.current_batch = 0
        self.train = train
    
    @property
    def num_node_features(self):
        data = self.datasets[0].get()
        return data["x_cur"].size(-1)
    
    @property
    def num_pos_features(self):
        data = self.datasets[0].get()
        return data["pos_cur"].size(-1)
    
    def get(self):
        data = [d.get() for d in self.datasets]
        data = Batch.from_data_list(data)
        return data
    
    def feedback(self, vel, norm=True):
        if isinstance(vel, torch.Tensor):
            vel = vel.detach().cpu().numpy()
        for i, d in enumerate(self.datasets):
            if d.env.abnormal_pose():
                d.feedback(d.gt_vel, norm=False)
            else:
                d.feedback(vel[i], norm)
    
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
            if isinstance(gui_dataset.env, PointEnvGUI):
                gui_dataset.env.clear_debug_items()

            raise StopIteration()
        
        need_reinit = [d.env.need_reinit() for d in self.datasets]
        if np.mean(need_reinit) > 0.5:
            print("[INFO] Re-initialize all environment")
            for d in self.datasets:
                d.env.init()

        return self.get()



if __name__ == "__main__":
    import time
    import pybullet as p
    import matplotlib.pyplot as plt

    p.connect(p.GUI_SERVER)

    dataloader = DataLoader(
        None, 
        batch_size=4, train=True, num_trajs=100, 
        env="PointGUI"
    )

    # dataloader = DataLoader(
    #     None,
    #     batch_size=1, train=False, num_trajs=100, num_prev_frames=3, 
    #     env="ImageGUI"
    # )
    # dataloader.datasets[-1].env.resample = True

    vel_trajs = []

    for data in dataloader:
        # show_graph(data, batch=-1)

        if getattr(data, "new_scene").any() and len(vel_trajs):
            vel_trajs = torch.stack(vel_trajs, dim=0).numpy()
            # vel_trajs = vel_trajs[-10:]
            plt.figure()
            for i, label in enumerate(["vx", "vy", "vz", "wx", "wy", "wz"]):
                plt.plot(vel_trajs[:, i], label=label)
            plt.legend()
            plt.tight_layout()
            plt.show()
            vel_trajs = []

        dataloader.feedback(data.vel * 2, norm=False)
        vel_trajs.append(data.vel_si[-1])
        time.sleep(dataloader.datasets[-1].env.dt)

