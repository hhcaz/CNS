import numpy as np
from enum import Enum, auto
from cns.utils.perception import CameraIntrinsic
from scipy.spatial.transform import Rotation as R


def ibvs(
    fp_cur: np.ndarray,
    Z_cur: np.ndarray,
    fp_tar: np.ndarray,
    Z_tar: np.ndarray
) -> np.ndarray:
    """Image-based visual servo controller.

    Arguments:
    - fp_cur: (N, 2), 2 represents (x, y), 
            feature points in current normalized camera plane
    - Z_cur: (N,), depth of feature points in current camera frame
    - fp_tar: (N, 2), feature points in target normalized camera frame
    - Z_tar: (N,), depth of feature points in target camera frame

    Returns:
    - vel: (6,), [tx, ty, tz, wx, wy, wz], camera velocity in current camera frame
    """

    assert fp_cur.shape == fp_tar.shape, "number of feature points not match"
    num_fp = fp_cur.shape[0]

    x_cur = fp_cur[:, 0]
    y_cur = fp_cur[:, 1]
    # build interaction matrix at current camera frame
    L_cur = np.zeros((num_fp * 2, 6))
    L_cur[0::2, 0] = -1. / Z_cur
    L_cur[0::2, 2] = x_cur / Z_cur
    L_cur[0::2, 3] = x_cur * y_cur
    L_cur[0::2, 4] = -(1 + x_cur * x_cur)
    L_cur[0::2, 5] = y_cur
    L_cur[1::2, 1] = -1. / Z_cur
    L_cur[1::2, 2] = y_cur / Z_cur
    L_cur[1::2, 3] = 1 + y_cur * y_cur
    L_cur[1::2, 4] = -x_cur * y_cur
    L_cur[1::2, 5] = -x_cur

    x_tar = fp_tar[:, 0]
    y_tar = fp_tar[:, 1]
    # build interaction matrix at target camera frame
    L_tar = np.zeros((num_fp * 2, 6))
    L_tar[0::2, 0] = -1. / Z_tar
    L_tar[0::2, 2] = x_tar / Z_tar
    L_tar[0::2, 3] = x_tar * y_tar
    L_tar[0::2, 4] = -(1 + x_tar * x_tar)
    L_tar[0::2, 5] = y_tar
    L_tar[1::2, 1] = -1. / Z_tar
    L_tar[1::2, 2] = y_tar / Z_tar
    L_tar[1::2, 3] = 1 + y_tar * y_tar
    L_tar[1::2, 4] = -x_tar * y_tar
    L_tar[1::2, 5] = -x_tar

    error = np.zeros(num_fp * 2)
    error[0::2] = x_tar - x_cur
    error[1::2] = y_tar - y_cur

    invL = np.linalg.pinv((L_cur + L_tar) / 2.)
    vel = np.dot(invL, error)

    return vel


def pbvs_center(cur_wcT, tar_wcT, wP):
    """PBVS: ensure the projection of scene center is always at 
    center of camera's FoV
    """
    wPo = np.mean(wP, axis=0)  # w: world frame, o: center, P: points

    tar_cwT = np.linalg.inv(tar_wcT)
    tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
    # t: target camera frame, c: current camera frame, w: world frame

    cur_cwT = np.linalg.inv(cur_wcT)
    cPo = wPo @ cur_cwT[:3, :3].T + cur_cwT[:3, 3]

    tcT = tar_cwT @ cur_wcT
    u = R.from_matrix(tcT[:3, :3]).as_rotvec()

    v = -(tPo - cPo + np.cross(cPo, u))
    w = -u
    vel = np.concatenate([v, w])

    return vel


def pbvs_straight(cur_wcT, tar_wcT):
    """PBVS2: goes straight and shortest path"""
    tcT = np.linalg.inv(tar_wcT) @ cur_wcT
    u = R.from_matrix(tcT[:3, :3]).as_rotvec()

    v = -tcT[:3, :3].T @ tcT[:3, 3]
    w = -u
    vel = np.concatenate([v, w])

    return vel


##########################################################
# below is the implementation of hybrid pbvs policy which
# goes straight and always looks at scene center


def _move_yx(ax, ay, az_m):
    ay = np.cross(az_m, ax)
    ax = np.cross(ay, az_m)

    ax = ax / np.linalg.norm(ax)
    ay = ay / np.linalg.norm(ay)
    az = az_m / np.linalg.norm(az_m)

    rot = np.stack([ax, ay, az], axis=1)
    return rot


def _move_xy(ax, ay, az_m):
    ax = np.cross(ay, az_m)
    ay = np.cross(az_m, ax)

    ax = ax / np.linalg.norm(ax)
    ay = ay / np.linalg.norm(ay)
    az = az_m / np.linalg.norm(az_m)

    rot = np.stack([ax, ay, az], axis=1)
    return rot


def _move_avg(ax, ay, az_m):
    """
    Original R: (ax, ay, az) -> Modified R: (ax_m, ay_m, az_m); 
    Imagine this process as you push the tip of original z-axis to pointing to 
    another direction (az_m), and how the original x-axis and y-axis will change

    Arguments:
    - ax: original x-axis
    - ay: original y-axis
    - az_m: modified z-axis

    Returns:
    - rot: 3x3 mat, modified rotation
    """
    rot_xy = _move_xy(ax, ay, az_m)
    rot_yx = _move_yx(ax, ay, az_m)

    rot = (rot_xy + rot_yx) / 2.0
    rot = rot / np.linalg.norm(rot, axis=0, keepdims=True)

    return rot


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2

    Arguments:
    - vec1: A 3d "source" vector
    - vec2: A 3d "destination" vector

    Returns:
    - mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) / (1 + c)
    return rotation_matrix


def _cur_ideal_pose(cur_wct, tar_wcT, wPo):
    """The ideal current rotation is calculated by applying left-hand transformation 
    to target pose (whose z-axis is confirmed to point towards wPo) to position that 
    locates on vec_o2cur and rotation of which z-axis still point towards wPo
    
              rot_tar2cur
            <------------
        cur_wct      tar_wcT
            \           /
             \         /
    vec_o2cur \       /  vec_o2tar
               \     /
                \   /
                 \ /
                  .
                 wPo

    Arguments:
    - cur_wct: (3,), current position in world coordinate
    - tar_wcT: (4, 4), target pose in world coordinate
    - wPo: (3,), position of points center in world coordinate

    Returns:
    - cur_wcT: (4, 4), ideal current pose
    """

    vec_o2tar = tar_wcT[:3, 3] - wPo
    vec_o2cur = cur_wct - wPo

    rot_tar2cur = rotation_matrix_from_vectors(vec_o2tar, vec_o2cur)
    cur_wcT = np.zeros_like(tar_wcT)
    cur_wcT[:3, :3] = rot_tar2cur @ tar_wcT[:3, :3]
    cur_wcT[:3, 3] = cur_wct
    cur_wcT[3, 3] = 1

    return cur_wcT


def _look_at_center(cur_wcT, wPo):
    vec_c2Po = wPo - cur_wcT[:3, 3]
    moved_cur_wcT = cur_wcT.copy()
    moved_cur_wcT[:3, :3] = _move_avg(cur_wcT[:3, 0], cur_wcT[:3, 1], vec_c2Po)

    ideal_cam_cam_R = np.linalg.inv(moved_cur_wcT[:3, :3]) @ cur_wcT[:3, :3]
    u = R.from_matrix(ideal_cam_cam_R).as_rotvec()
    return u


def _pbvs_mine(cur_wcT, tar_wcT, wPo):
    """确保场景中心大致在视野中央, 且大致走直线"""
    # wPo = np.mean(wP, axis=0)  # w: world frame, o: center, P: points

    # the z-axis of desired pose may not points to the scene center, 
    # here we first slightly change the rotation of desired pose as moved_tar_wcT
    # of which the z-axis is pointing towrads scene center
    vec_t2Po = wPo - tar_wcT[:3, 3]
    moved_tar_wcT = tar_wcT.copy()
    moved_tar_wcT[:3, :3] = _move_avg(tar_wcT[:3, 0], tar_wcT[:3, 1], vec_t2Po)

    ideal_wcT = _cur_ideal_pose(cur_wcT[:3, 3], moved_tar_wcT, wPo)
    ideal_cam_cam_R = np.linalg.inv(ideal_wcT[:3, :3]) @ cur_wcT[:3, :3]
    u_to_cur_ideal = R.from_matrix(ideal_cam_cam_R).as_rotvec()
    u_to_cur_center = _look_at_center(cur_wcT, wPo)

    u = u_to_cur_center * 7 + u_to_cur_ideal
    w = -u

    word_v_cam = moved_tar_wcT[:3, 3] - cur_wcT[:3, 3]
    cam_v_cam = np.linalg.inv(cur_wcT[:3, :3]) @ word_v_cam
    return np.concatenate([cam_v_cam, w])


def _logistic(x, a=1):
    return 1.0 / (1 + np.exp(-x*a))


def pbvs_hybrid(cur_wcT, tar_wcT, wP):
    wPo = np.mean(wP, axis=0)
    u = np.linalg.inv(cur_wcT[:3, :3]) @ tar_wcT[:3, :3]
    u = R.from_matrix(u).as_rotvec()
    uz_deg = np.abs(u[-1]) / np.pi * 180
    alpha = _logistic(uz_deg - 30, 0.4)

    control = alpha * _pbvs_mine(cur_wcT, tar_wcT, wPo) + \
        (1 - alpha) * pbvs_straight(cur_wcT, tar_wcT)
    return control


class Policy(Enum):
    IBVS = auto()
    PBVS_Center = auto()
    PBVS_Straight = auto()
    PBVS_Center_Straight = auto()


def supervisor_vel(
    policy: Policy,
    cur_fp, cur_Z, tar_fp, tar_Z, intrinsic: CameraIntrinsic,
    cur_wcT, tar_wcT, wP, *args, **kwargs
):
    if policy == Policy.IBVS:
        vel = ibvs(cur_fp, cur_Z, tar_fp, tar_Z, intrinsic)
    elif policy == Policy.PBVS_Center:
        vel = pbvs_center(cur_wcT, tar_wcT, wP)
    elif policy == Policy.PBVS_Straight:
        vel = pbvs_straight(cur_wcT, tar_wcT)
    elif policy == Policy.PBVS_Center_Straight:
        vel = pbvs_hybrid(cur_wcT, tar_wcT, wP)
    else:
        raise ValueError("Unknown Policy")

    # calculate distance decoupled velocity
    wPo = np.mean(wP, axis=0)
    tar_cwT = np.linalg.inv(tar_wcT)
    tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
    tPo_norm = np.linalg.norm(tPo)

    vel_si = vel.copy()
    vel_si[:3] /= tPo_norm + 1e-7

    return vel, (tPo_norm, vel_si)

