import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import cv2
import glob
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from collections import namedtuple
from typing import List, Union, Optional

from .sampling import sample_camera_pose
from .sampling import uniform_ball_sample, gen_virtual_points, gen_virtual_points_uniformed
from cns.utils.perception import Camera, CameraIntrinsic


def gen_observation(num_pts, obs_r, obs_h):
    if np.random.uniform(0, 1) < 0.5:
        return gen_virtual_points(num_pts, obs_r, obs_h)
    else:
        return gen_virtual_points_uniformed(num_pts, obs_r, obs_h)


class PointEnv(object):
    def __init__(
        self, 
        camera_config: Optional[Union[str, CameraIntrinsic]], 
        resample=True, 
        auto_reinit=True, 
        verbose=False
    ):
        if camera_config is None:
            camera_intrinsic = CameraIntrinsic.default()
        elif isinstance(camera_config, str):
            with open(camera_config, "r") as j:
                config = json.load(j)
            camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        elif isinstance(camera_config, CameraIntrinsic):
            camera_intrinsic = camera_config

        self.camera = Camera(camera_intrinsic)
        self.sample_points_func = gen_observation

        self.obs_r = 0.2
        self.obs_h = 0.1
        self.obs_num_range = (4, 512)

        self.target_pose_r = [0.5, 0.9]
        self.target_pose_phi = [70, 90]
        self.target_pose_drz_max = 15
        self.target_pose_dry_max = 5
        self.target_pose_drx_max = 5

        self.initial_pose_r = [0.5, 0.9]
        self.initial_pose_phi = [30, 90]
        self.initial_pose_drz_max = 60
        self.initial_pose_dry_max = 10
        self.initial_pose_drx_max = 10

        self.dt = 1./50.  # 50Hz
        self.angle_eps = 1  # degree
        self.dist_eps = 0.002  # m
        self.max_steps = 200

        self.resample = resample
        self.auto_reinit = auto_reinit
        self.verbose = verbose

        # state
        self.points = None
        self.target_wcT = None
        self.initial_wcT = None
        self.current_wcT = None
        self.steps = 0
    
    def init(self):
        if self.resample or (self.points is None):
            num_pts = np.random.randint(self.obs_num_range[0], self.obs_num_range[1]+1)
            self.points = self.sample_points_func(
                num_pts=num_pts, obs_r=self.obs_r, obs_h=self.obs_h)  # (N, 3)

            while True:
                self.target_wcT = self.sample_target_pose()
                self.initial_wcT = self.sample_initial_pose()
                if not self.close_enough(self.target_wcT, self.initial_wcT):
                    break
        self.current_wcT = self.initial_wcT.copy()
        self.steps = 0

    def observation(self):
        W, H = self.camera.intrinsic.width, self.camera.intrinsic.height
        target_cwT = np.linalg.inv(self.target_wcT)
        target_xy = self.camera.project(target_cwT, self.points)
        target_Z = (self.points @ target_cwT[:3, :3].T + target_cwT[:3, 3])[:, -1]  # (N,)

        current_cwT = np.linalg.inv(self.current_wcT)
        current_xy = self.camera.project(current_cwT, self.points)
        current_Z = (self.points @ current_cwT[:3, :3].T + current_cwT[:3, 3])[:, -1]  # (N,)
        remove_index = []

        return (
            current_xy, current_Z, target_xy, target_Z, self.camera.intrinsic,
            self.current_wcT, self.target_wcT, self.points, remove_index
        )
    
    def action(self, vel):
        try:
            dT = np.eye(4)
            dT[:3, :3] = R.from_rotvec(vel[3:] * self.dt).as_matrix()
            dT[:3, 3] = vel[:3] * self.dt
            self.current_wcT = self.current_wcT @ dT
        except Exception as e:
            self.current_wcT = self.initial_wcT
            print("[INFO] dT = {}".format(dT))
            print("[INFO] vel = {}".format(vel))
            print("[ERR ] error in action:")
            print(e)
            raise(e)
        self.current_wcT[:3, :3] = R.from_matrix(self.current_wcT[:3, :3]).as_matrix()
        self.steps += 1

        if self.auto_reinit and self.need_reinit():
            self.init()
    
    def sample_target_pose(self):
        return sample_camera_pose(
            r_min=self.target_pose_r[0],
            r_max=self.target_pose_r[1],
            phi_min=self.target_pose_phi[0],
            phi_max=self.target_pose_phi[1],
            drz_max=self.target_pose_drz_max,
            dry_max=self.target_pose_dry_max,
            drx_max=self.target_pose_drx_max
        )
    
    def sample_initial_pose(self):
        return sample_camera_pose(
            r_min=self.initial_pose_r[0],
            r_max=self.initial_pose_r[1],
            phi_min=self.initial_pose_phi[0],
            phi_max=self.initial_pose_phi[1],
            drz_max=self.initial_pose_drz_max,
            dry_max=self.initial_pose_dry_max,
            drx_max=self.initial_pose_drx_max
        )
    
    def close_enough(self, T0, T1):
        dT = np.linalg.inv(T0) @ T1
        angle = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()) / np.pi * 180
        dist = np.linalg.norm(dT[:3, 3])
        return (angle < self.angle_eps) and (dist < self.dist_eps)

    def exceeds_maximum_steps(self):
        return self.steps > self.max_steps

    def abnormal_pose(self):
        pos = self.current_wcT[:3, 3]
        # if pos[-1] < 0:
        if pos[-1] < self.initial_pose_r[0] * np.sin(self.initial_pose_phi[0]) * 0.5:
            if self.verbose:
                print("[INFO] Camera under ground")
            return True
        
        zaxis = self.current_wcT[:3, 2]
        if zaxis[-1] > -0.1:
            if self.verbose:
                print("[INFO] Camera almost look up")
            return True

        dist = np.linalg.norm(pos)
        if dist > self.target_pose_r[-1] * 2:
            if self.verbose:
                print("[INFO] Too far away")
            return True
        
        wPo = np.mean(self.points, axis=0)  # (3,)
        current_cwT = np.linalg.inv(self.current_wcT)
        cPo = current_cwT[:3, :3] @ wPo + current_cwT[:3, 3]
        if cPo[-1] < 0:
            if self.verbose:
                print("[INFO] Objects moving behind camera")
            return True
        
        return False
    
    def need_reinit(self):
        if self.close_enough(self.target_wcT, self.current_wcT):
            return True
        
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

        du_deg = np.linalg.norm(du)/np.pi*180
        dt_mm = np.linalg.norm(dt)*1000
        print("[INFO] Servo error: |du| = {:.3f} degree, |dt| = {:.3f} mm"
            .format(du_deg, dt_mm))
        return du_deg, dt_mm


class DebugAxes(object):
    """Visualize axes, red for x axis, green for y axis, blue for z axis"""
    def __init__(self, client=0):
        self.uids = [-1, -1, -1]
        self.client = client
    
    def update(self, pose):
        pos = pose[:3, 3]
        rot3x3 = pose[:3, :3]
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0],
            replaceItemUniqueId=self.uids[0], physicsClientId=self.client)
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], 
            replaceItemUniqueId=self.uids[1], physicsClientId=self.client)
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1],
            replaceItemUniqueId=self.uids[2], physicsClientId=self.client)
    
    def clear(self):
        p.removeUserDebugItem(self.uids[0], physicsClientId=self.client)
        p.removeUserDebugItem(self.uids[1], physicsClientId=self.client)
        p.removeUserDebugItem(self.uids[2], physicsClientId=self.client)
        self.uids = [-1, -1, -1]


class DebugPoints(object):
    """Visulaize points"""
    def __init__(self, client=0):
        self.uid = -1
        self.client = client
    
    def update(self, pos):
        colors = np.repeat(np.array([[1, 0, 0]]), len(pos), axis=0)  # red
        self.uid = p.addUserDebugPoints(pos, colors, 5, 
            replaceItemUniqueId=self.uid, physicsClientId=self.client)
    
    def clear(self):
        p.removeUserDebugItem(self.uid, physicsClientId=self.client)
        self.uid = -1


class PointEnvGUI(PointEnv):
    def __init__(self, camera_config, resample=True, auto_reinit=True):
        super().__init__(camera_config, resample, auto_reinit, verbose=True)

        if not hasattr(self, "client"):
            self.client = p.connect(p.SHARED_MEMORY)
            print("[INFO] Client (id = {}) initialized".format(self.client))
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(1.674, 70, -50.8, [0, 0, 0], physicsClientId=self.client)

        self.axes_cam_tar = DebugAxes(self.client)
        self.axes_cam_cur = DebugAxes(self.client)
        self.pc = DebugPoints(self.client)
        self.debug_items_displayed = False

    def init(self):
        super().init()
        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
            self.pc.update(self.points)
    
    def init_debug_items(self):
        self.axes_cam_tar.update(self.target_wcT)
        self.axes_cam_cur.update(self.current_wcT)
        self.pc.update(self.points)
        self.debug_items_displayed = True
        print("[INFO] Debug items initialzied for client {}".format(self.client))
    
    def clear_debug_items(self):
        self.axes_cam_tar.clear()
        self.axes_cam_cur.clear()
        self.pc.clear()
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


class ImageEnvGUI(PointEnvGUI):
    SceneObj = namedtuple("SceneObj", ["path", "scale", "pos", "orn"])
    YCB_ROOT = os.path.join(os.path.dirname(__file__), 
        "..", "thirdparty", "pybullet-object-models", "pybullet_object_models", "ycb_objects")

    def __init__(self, camera_config, resample=True, auto_reinit=True):
        super().__init__(camera_config, resample, auto_reinit)

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

        self.tar_img = None
        self.tar_kp = None
        self.tar_des = None
        self.obj_ids = []
        self.objs: List[self.SceneObj] = []
        self.few_match_occur = False
    
    def extract_kp_desc(self, image):
        if hasattr(cv2, "SIFT_create"):
            sift_create = cv2.SIFT_create
        else:
            sift_create = cv2.xfeatures2d.SIFT_create
        
        sift = sift_create()
        kp, des = sift.detectAndCompute(image, None)

        if des is None:
            print("[INFO] No keypoints and descriptors detected")
            plt.figure()
            plt.imshow(image)
            plt.show()

        des = des.astype(np.float32)
        return kp, des

    def feature_match(self, src_kp, src_des):
        assert (self.tar_kp is not None) and (self.tar_des is not None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.tar_des, src_des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        
        # tar_pts = np.float32([self.tar_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        # src_pts = np.float32([src_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)

        # if len(good) < 5:
        #     M, mask = cv2.findHomography(tar_pts, src_pts, cv2.RANSAC, 5.0)
        # else:
        #     E, mask = cv2.findEssentialMat(tar_pts, src_pts, self.camera.intrinsic.K,
        #         cv2.RANSAC, 0.999, 1.0)
        
        # mask = mask.ravel().tolist()
        # good = [g for g, m in zip(good, mask) if m]

        return good
    
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

        # plane_pos, plane_orn = [0, 0, -self.obs_h*2], [0, 0, 0, 1]
        plane_pos, plane_orn = [0, 0, 0], [0, 0, 0, 1]
        obj_id = p.loadURDF('plane.urdf', plane_pos, plane_orn, 
            globalScaling=1, physicsClientId=self.client)
        self.obj_ids.append(obj_id)
        self.objs.append(
            self.SceneObj(path='plane.urdf', scale=1, pos=plane_pos, orn=plane_orn))

        num_objs = np.clip(num_objs, 1, None)
        # uniformly distributed in a horizontal circle
        position = uniform_ball_sample(num_objs, 2) * obs_r
        # uniformly distributed vertically
        position = np.concatenate([position, (np.random.rand(num_objs, 1)-0.5)*obs_h], axis=-1)  # (N, 3)
        # overvall, uniformly distributed in a cyclinder with r=obs_r and h=obs_h
        position[:, -1] += self.obs_h*2

        for i in range(num_objs):
            rpy = np.random.uniform(-np.pi, np.pi, size=3)  # random pose
            orn = p.getQuaternionFromEuler(rpy)
            model, scale = self.candidates[np.random.choice(len(self.candidates))]
            obj_id = p.loadURDF(model, position[i], orn, globalScaling=scale, physicsClientId=self.client)
            print("[INFO] loaded id = {}, loaded model: {}".format(obj_id, model))
            self.obj_ids.append(obj_id)
            self.objs.append(
                self.SceneObj(path=model, scale=scale, pos=position[i], orn=orn))
        
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        for _ in range(200):
            p.stepSimulation(physicsClientId=self.client)

    def init(self):
        if self.resample or (self.points is None):
            # num_objs = np.random.randint(
            #     int(np.log2(self.obs_num_range[0] + 1)) + 1,
            #     int(np.log2(self.obs_num_range[1] + 1)) + 1) * 4

            num_objs = 16
            self.gen_scenes(num_objs, self.obs_r, self.obs_h)

            while True:
                self.target_wcT = self.sample_target_pose()
                self.initial_wcT = self.sample_initial_pose()
                if not self.close_enough(self.target_wcT, self.initial_wcT):
                    break
            
            target_cwT = np.linalg.inv(self.target_wcT)
            frame = self.camera.render(target_cwT, self.client)

            rgb = frame.color_image()
            self.tar_img = np.ascontiguousarray(rgb[:, :, ::-1])  # rgb to bgr
            self.tar_kp, self.tar_des = self.extract_kp_desc(self.tar_img)

            # map 2d feature points to 3d points in world frame, and stores in self.points
            W, H = self.camera.intrinsic.width, self.camera.intrinsic.height
            pc = np.asarray(frame.point_cloud().points).reshape(H, W, 3)  # (H, W, 3)
            kp_numpy = np.array([kp.pt for kp in self.tar_kp])
            target_ji = np.clip(np.round(kp_numpy).astype(np.int32), [0, 0], [W-1, H-1])  # (N, 2)
            cc, rr = target_ji.T
            self.points = np.ascontiguousarray(pc[rr, cc])  # (N, 3)

            # maybe we need sparse fps sampling if points are too much
            # TODO: farthest point sampling

            self.target_xy = kp_numpy
            self.target_Z = (self.points @ target_cwT[:3, :3].T + target_cwT[:3, 3])[:, -1]  # (N,)

        self.current_wcT = self.initial_wcT.copy()
        self.steps = 0
        self.few_match_occur = False

        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
            self.pc.update(self.points)
    
    def observation(self):
        W, H = self.camera.intrinsic.width, self.camera.intrinsic.height
        current_cwT = np.linalg.inv(self.current_wcT)
        frame = self.camera.render(current_cwT, self.client)

        rgb = frame.color_image()
        src_img = np.ascontiguousarray(rgb[:, :, ::-1])  # rgb to bgr
        src_kp, src_des = self.extract_kp_desc(src_img)

        matches = self.feature_match(src_kp, src_des)
        print("[INFO] Number of matches = {}".format(len(matches)))
        self.few_match_occur = len(matches) < 4

        current_xy = np.zeros_like(self.target_xy)
        observed_mask = np.zeros(len(self.target_xy), dtype=bool)
        for m in matches:
            current_xy[m.queryIdx] = src_kp[m.trainIdx].pt
            observed_mask[m.queryIdx] = True

        current_Z = (self.points @ current_cwT[:3, :3].T + current_cwT[:3, 3])[:, -1]  # (N,)
        remove_index = np.nonzero(~observed_mask)[0]

        if not self.debug_items_displayed:
            self.init_debug_items()
        
        image = cv2.drawMatches(self.tar_img, self.tar_kp, src_img, src_kp, matches, None)
        cv2.imshow("target|current", image)
        cv2.waitKey(1)

        return (
            current_xy, current_Z, self.target_xy, self.target_Z, self.camera.intrinsic,
            self.current_wcT, self.target_wcT, self.points, remove_index
        )

    def need_reinit(self):
        return self.few_match_occur or super().need_reinit()


if __name__ == "__main__":
    from .supervisor import pbvs
    p.connect(p.GUI_SERVER)

    # env = PointEnvGUI(None, resample=True, auto_reinit=False)
    env = ImageEnvGUI(None, resample=True, auto_reinit=False)
    env.init()

    input("[INFO] Press Enter to start: ")
    while True:
        current_xy, current_Z, target_xy, target_Z, intrinsic, \
            current_wcT, target_wcT, points, remove_index = env.observation()

        vel = pbvs(current_wcT, target_wcT, points)
        vel = vel * 2

        # print("[INFO] vel trans = {}, vel = {}".format(np.linalg.norm(vel[3:]), vel))
        env.action(vel)
        env.print_pose_err()

        if env.need_reinit():
            input("[INFO] Reach goal! Use {} steps. Press Enter to reinit: ".format(env.steps))
            print("-----------------------------------------------------------")
            env.init()
            # input("[INFO] Press Enter to start: ")
        
        time.sleep(env.dt)

