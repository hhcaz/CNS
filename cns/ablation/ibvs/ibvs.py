import json
import numpy as np
from cns.midend.graph_gen import GraphData


def ibvs_Ltar(
    fp_cur: np.ndarray,
    fp_tar: np.ndarray,
    Z_tar: np.ndarray,
) -> np.ndarray:
    """Image-based visual servo controller.

    Arguments:
    - fp_cur: (N, 2), 2 represents (x, y), 
            feature points in current normalized camera frame
    - fp_tar: (N, 2), feature points in target normalized camera frame
    - Z_tar: (N,), depth of feature points in target camera frame

    Returns:
    - vel: (6,), [tx, ty, tz, wx, wy, wz], camera velocity in current camera frame
    """

    assert fp_cur.shape == fp_tar.shape, "number of feature points not match"
    num_fp = fp_cur.shape[0]

    x_cur = fp_cur[:, 0]
    y_cur = fp_cur[:, 1]
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

    invL = np.linalg.pinv(L_tar)
    vel = np.dot(invL, error)

    return vel


def ibvs_Lmean(
    fp_cur: np.ndarray,
    Z_cur: np.ndarray,
    fp_tar: np.ndarray,
    Z_tar: np.ndarray
) -> np.ndarray:
    """Image-based visual servo controller.

    Arguments:
    - fp_cur: (N, 2), 2 represents (x, y), 
            feature points in current normalized camera frame
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


class IBVS(object):
    # def __init__(self, config_path):
    #     with open(config_path, "r") as fp:
    #         config = json.load(fp)
    #     self.use_mean = config["use_mean"]

    def __init__(self, use_mean=True):
        self.use_mean = use_mean
    
    def __call__(self, data: GraphData):
        node_mask = getattr(data, "node_mask").cpu().numpy()
        pos_cur = getattr(data, "pos_cur").cpu().numpy()[node_mask]
        pos_tar = getattr(data, "pos_tar").cpu().numpy()[node_mask]
        dist = getattr(data, "tPo_norm").cpu().numpy()

        if self.use_mean:
            return ibvs_Lmean(pos_cur, dist, pos_tar, dist)
        else:
            return ibvs_Ltar(pos_cur, pos_tar, dist)

