import time
import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
from ..sim.sampling import sample_camera_pose
from .stream import Stream


def vel_transform(
    BvQ = np.zeros(3),
    BwQ = np.zeros(3),
    AvB = np.zeros(3),
    AwB = np.zeros(3),
    ABR = np.eye(3),
    BQ = np.zeros(3)
):
    """
    Ref: https://zhuanlan.zhihu.com/p/155829972
    章节: 线速度和角速度同时存在的情况
    (^A V_Q) = (^A V_{BORG}) + (^A_B R) \cdot (^B V_Q) + 
                               (^A \Omega_B) \cross (^A_B R) \cdot (^B Q)
    (^A W_Q) = (^A W_B) + (^A_B R) \cdot (^B W_Q)

    Arguments:
    - BvQ: Q的线速度在B坐标系下的表示;
    - BwQ: Q的角速度在B坐标系下的表示;
    - AvB: 坐标系B的原点的线速度在A坐标系下的表示
    - AwB: 坐标系B的角速度在A坐标系下的表示
    - ABR: 坐标系B的姿态的旋转部分在坐标系A下的表示
    - BQ: B坐标系原点指向Q的向量在B坐标系下的表示

    Returns:
    - AvQ: Q的线速度在A坐标系下的表示;
    - AwQ: Q的角速度在A坐标系下的表示.
    """
    AvQ = AvB + ABR @ BvQ + np.cross(AwB, ABR @ BQ)
    AwQ = AwB + ABR @ BwQ
    return AvQ, AwQ


class RealEnv(object):

    tcp_cam_T_path = 'tcp_cam_T.npy'

    def __init__(self, resample=True, auto_reinit=False):
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.101")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.101")
        self.stream = Stream()

        # 目标场景的中心点在UR5机械臂基坐标系下的坐标
        self.target_scene_origin = np.array([0.5, 0.0, 0.0])  # under base frame

        # 理想目标相机的位姿的z轴始终指向场景中心，同时相机y轴与观测的半球相切并向下
        self.target_pose_r = [0.25, 0.3]  # 理想目标相机位姿的采样半径范围
        self.target_pose_phi = [70, 90]  # 理想目标相机坐标与场景中心的连线与基坐标系xOy平面的夹角
                                         # 90°意味着相机是垂直向下看的
        self.target_pose_drz_max = 15  # 到达理想目标位姿后沿着相机坐标系的y轴的旋转扰动，单位degree
        self.target_pose_dry_max = 5
        self.target_pose_drx_max = 5

        self.initial_pose_r = [0.4, 0.5]
        self.initial_pose_phi = [80, 90]
        self.initial_pose_drz_max = 15
        self.initial_pose_dry_max = 5
        self.initial_pose_drx_max = 5

        self.resample = resample
        self.auto_reinit = auto_reinit

        self.target_base_cam_T = None
        self.initial_base_cam_T = None
        self.current_base_cam_T = None
        self.steps = 0

        self.tcp_cam_T = np.load(self.tcp_cam_T_path)
        self.dt = 0.02

    def sample_target_pose(self):
        base_cam_T = sample_camera_pose(
            r_min=self.target_pose_r[0],
            r_max=self.target_pose_r[1],
            phi_min=self.target_pose_phi[0],
            phi_max=self.target_pose_phi[1],
            drz_max=self.target_pose_drz_max,
            dry_max=self.target_pose_dry_max,
            drx_max=self.target_pose_drx_max
        )
        base_cam_T[:3, 3] += self.target_scene_origin
        return base_cam_T
    
    def sample_initial_pose(self):
        base_cam_T = sample_camera_pose(
            r_min=self.initial_pose_r[0],
            r_max=self.initial_pose_r[1],
            phi_min=self.initial_pose_phi[0],
            phi_max=self.initial_pose_phi[1],
            drz_max=self.initial_pose_drz_max,
            dry_max=self.initial_pose_dry_max,
            drx_max=self.initial_pose_drx_max
        )
        base_cam_T[:3, 3] += self.target_scene_origin
        return base_cam_T

    def generate_poses(self, num, seed=None):
        target_poses = np.zeros((num, 4, 4))
        initial_poses = np.zeros((num, 4, 4))
        if seed is not None:
            np.random.seed(seed)
        for i in range(num):
            target_poses[i] = self.sample_target_pose()
            initial_poses[i] = self.sample_initial_pose()
        return target_poses, initial_poses

    def observation(self):
        bgr, depth = self.stream.get()
        dist_scale = np.median(depth)
        return bgr, dist_scale

    def transform_vel(self, cam_vel_cam):
        cam_vel_cam = np.asarray(cam_vel_cam)
        cam_v_cam, cam_w_cam = cam_vel_cam[:3], cam_vel_cam[3:]

        # Q as cam, B as cam, A as base
        base_cam_R = self.get_current_base_cam_T()[:3, :3]
        base_v_cam, base_w_cam = vel_transform(
            BvQ=cam_v_cam,
            BwQ=cam_w_cam,
            ABR=base_cam_R,
            BQ=np.zeros(3)
        )

        # Q as tcp, B as cam, A as base
        cam_t_tcpORG = np.linalg.inv(self.tcp_cam_T)[:3, 3]
        base_v_tcp, base_w_tcp = vel_transform(
            AvB=base_v_cam,
            AwB=base_w_cam,
            ABR=base_cam_R,
            BQ=cam_t_tcpORG
        )

        base_vel_tcp = np.concatenate([base_v_tcp, base_w_tcp])
        return base_vel_tcp

    def action(self, cam_vel_cam):
        max_vel = 0.2
        max_acc = 0.4

        vel_norm = np.linalg.norm(cam_vel_cam)
        if vel_norm > max_vel:
            cam_vel_cam = cam_vel_cam / (vel_norm + 1e-7) * max_vel

        base_vel_tcp = self.transform_vel(cam_vel_cam)
        if np.any(np.abs(base_vel_tcp) > 1e-5):
            self.rtde_c.speedL(base_vel_tcp.tolist(), max_acc, self.dt)
        return cam_vel_cam

    def is_safe_pose(self):
        base_cam_T = self.get_current_base_cam_T()
        z_axis = base_cam_T[:3, 2]
        proj = z_axis @ np.array([0, 0, -1])
        if proj < 0.6:
            print("[INFO] Camera almost parallel to the ground")
            return False

        box_size = 0.6
        tx, ty, tz = base_cam_T[:3, 3]
        ox, oy, oz = self.target_scene_origin

        if tx < ox - box_size / 2. or tx > ox + box_size / 2.:
            print("[INFO] tx (={:.2f}) should in [{:.2f}, {:.2f}]"
                  .format(tx, ox - box_size / 2., ox + box_size / 2.))
            return False

        if ty < oy - box_size / 2. or ty > oy + box_size / 2.:
            print("[INFO] ty (={:.2f}) should in [{:.2f}, {:.2f}]"
                  .format(ty, oy - box_size / 2., oy + box_size / 2.))
            return False

        min_z = min(*self.target_pose_r, *self.initial_pose_r) * 0.5
        max_z = max(*self.target_pose_r, *self.initial_pose_r) * 1.2
        if tz < min_z or tz > max_z:
            print("[INFO] tz (={:.2f}) should be in [{:.2f}, {:.2f}]"
                  .format(tz, min_z, max_z))
            return False

        return True

    def stop_action(self):
        dt = 0.3
        acceleration = 0.5
        for _ in range(3):
            joint_speed = [0.0001] * 6
            start = time.time()
            self.rtde_c.speedJ(joint_speed, acceleration, dt)
            end = time.time()
            duration = end - start
            if duration < dt:
                time.sleep(dt - duration)
        self.rtde_c.speedStop(100)

    def get_current_base_tcp_T(self):
        x, y, z, rx, ry, rz = self.rtde_r.getActualTCPPose()
        base_tcp_T = np.eye(4)
        base_tcp_T[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
        base_tcp_T[:3, 3] = [x, y, z]
        return base_tcp_T

    def get_current_base_cam_T(self):
        base_cam_T = self.get_current_base_tcp_T() @ self.tcp_cam_T
        return base_cam_T

    def close_enough(self, T0, T1):
        dT = np.linalg.inv(T0) @ T1
        rot_vec = R.from_matrix(dT[:3, :3]).as_rotvec()
        angle = np.linalg.norm(rot_vec) / np.pi * 180  # degree
        dist = np.linalg.norm(dT[:3, 3]) * 1000  # mm
        return (angle < 3) and (dist < 3)

    def move_to(self, base_cam_T):
        self.stop_action()
        # base_cam_T: camera pose under base frame
        base_tcp_T = base_cam_T @ np.linalg.inv(self.tcp_cam_T)

        xyz = base_tcp_T[:3, 3]
        rot = R.from_matrix(base_tcp_T[:3, :3]).as_rotvec()
        pose = np.concatenate([xyz, rot]).tolist()
        self.rtde_c.moveL(pose, 0.1, 0.2, True)

        while not self.close_enough(base_cam_T, self.get_current_base_cam_T()):
            time.sleep(self.dt)
        time.sleep(0.5)
        self.rtde_c.stopL(10)

    def pbvs_move_to(self, base_cam_T):
        while True:
            cur_bcT = self.get_current_base_cam_T()
            if self.close_enough(cur_bcT, base_cam_T):
                break
            tcT = np.linalg.inv(base_cam_T) @ cur_bcT
            u = R.from_matrix(tcT[:3, :3]).as_rotvec()
            v = -tcT[:3, :3].T @ tcT[:3, 3]
            w = -u
            cam_vel_cam = np.concatenate([v, w])
            self.action(cam_vel_cam)
        self.stop_action()

    def recover_end_joint_pose(self):
        self.stop_action()
        while True:
            cur_joint_pos = self.rtde_r.getActualQ()
            if np.abs(cur_joint_pos[-1]) > 1e-3:
                tar_joint_pos = cur_joint_pos.copy()
                tar_joint_pos[-1] = 0.0
                self.rtde_c.moveJ(tar_joint_pos, 1.05, 1.4, True)
                dt = np.abs(cur_joint_pos[-1]) / 1.05 + 0.5
                time.sleep(dt)
            else:
                self.rtde_c.stopJ(8)
                break

    def go_home(self, z=0.2):
        self.recover_end_joint_pose()
        target_cam_xyz = [0.5, 0.0, z]
        base_cam_T = np.eye(4)
        base_cam_T[:3, 0] = [0, -1, 0]
        base_cam_T[:3, 1] = [-1, 0, 0]
        base_cam_T[:3, 2] = [0, 0, -1]
        base_cam_T[:3, 3] = target_cam_xyz
        self.move_to(base_cam_T)


if __name__ == "__main__":
    env = RealEnv(resample=True, auto_reinit=False)

    test_target_cam_xyzs = [
        [0.6, 0.1, 0.4],
        [0.6, -0.1, 0.4],
        [0.4, -0.1, 0.4],
        [0.4, 0.1, 0.4],
    ]

    env.go_home()
    for _ in range(4):
        tar_bcT = env.sample_target_pose()
        env.pbvs_move_to(tar_bcT)
    env.go_home()
