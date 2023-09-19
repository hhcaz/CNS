import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from torch_geometric.data import Batch

from ..midend.graph_gen import GraphData
from .perception import CameraIntrinsic


def get_cycle_colors():
    def hex2rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [hex2rgb(c)[::-1] for c in colors]  # hex to bgr
    return colors


CYCLE_COLORS = get_cycle_colors()


def draw_keypoints(data: Union[GraphData, Batch], batch=-1):
    """Draw keypoints observed from current pose (white) and desired pose (colored).

    Arguments:
    - data: GraphData directly from cns.sim.dataset.Dataset or frontend, 
            or batch of GraphData from cns.sim.dataset.DataLoader
    - batch: if `data` is Batch, specify which batch to visualize,
            default is -1, meaning the last batch
    
    Returns:
    - image: numpy color image, dtype=uint8, shape=(H, W, 3),
            H and W are the same as those in camera config
    """
    if isinstance(data, Batch):
        if batch < 0:
            batch_idx = getattr(data, "batch", None)
            num_batches = (batch_idx.max() + 1) if batch_idx is not None else 1
            batch = num_batches + batch
        data: GraphData = data.get_example(batch)

    intrinsic: CameraIntrinsic = getattr(data, "intrinsic")
    fx, fy = intrinsic.fx, intrinsic.fy
    cx, cy = intrinsic.cx, intrinsic.cy
    W, H = intrinsic.width, intrinsic.height
    image = np.zeros((H, W, 3), dtype=np.uint8)

    belong_cluster_index = getattr(data, "belong_cluster_index")
    node_mask = getattr(data, "node_mask").cpu().numpy()
    pos_cur = getattr(data, "pos_cur").cpu().numpy()
    pos_tar = getattr(data, "pos_tar").cpu().numpy()

    walker_centers = getattr(data, "walker_centers", None)
    if walker_centers is not None:
        for u, v in walker_centers:
            j = np.clip(int(round(u * fx + cx)), 0, W-1)
            i = np.clip(int(round(v * fy + cy)), 0, H-1)
            cv2.circle(image, (j, i), radius=10, color=(200, 200, 200), thickness=3)

    colors = CYCLE_COLORS
    Ncolor = len(colors)

    j_tar, i_tar = intrinsic.norm_camera_plane_to_pixel(pos_tar, clip=True, round=True).T
    j_cur, i_cur = intrinsic.norm_camera_plane_to_pixel(pos_cur, clip=True, round=True).T

    for i in range(len(j_tar)):
        c = j_tar[i]; r = i_tar[i]
        color = colors[belong_cluster_index[i] % Ncolor]
        cv2.circle(image, (c, r), radius=3, color=color, thickness=cv2.FILLED)
    
    for c, r in zip(j_cur[node_mask], i_cur[node_mask]):
        cv2.circle(image, (c, r), radius=2, 
            color=(255, 255, 255), thickness=cv2.FILLED)
    
    proj_wPo = getattr(data, "proj_wPo", None)
    if proj_wPo is not None:
        u, v = proj_wPo.cpu().numpy().flatten()
        u, v = int(round(u)), int(round(v))
        cv2.circle(image, (u, v), radius=8, color=(0, 0, 255), thickness=cv2.FILLED)

    return image


def show_keypoints(data: Union[GraphData, Batch], batch=-1):
    image = draw_keypoints(data, batch)
    cv2.imshow("keypoints", image)
    return cv2.waitKey(1)


# def draw_graph(data: Union[GraphData, Batch], batch=-1):
#     """Draw graph structure (only draw edges from keypoints to cluster centers).

#     Arguments:
#     - data: GraphData directly from cns.sim.dataset.Dataset or frontend, 
#             or batch of GraphData from cns.sim.dataset.DataLoader
#     - batch: if `data` is Batch, specify which batch to visualize,
#             default is -1, meaning the last batch
    
#     Returns:
#     - image: numpy color image, dtype=uint8, shape=(H, W, 3),
#             H and W are the same as those in camera config
#     """
#     if isinstance(data, Batch):
#         if batch < 0:
#             batch_idx = getattr(data, "batch", None)
#             num_batches = (batch_idx.max() + 1) if batch_idx is not None else 1
#             batch = num_batches + batch
#         data: GraphData = data.get_example(batch)

#     intrinsic: CameraIntrinsic = getattr(data, "intrinsic")
#     W, H = intrinsic.width, intrinsic.height
#     image = np.zeros((H, W, 3), dtype=np.uint8)
    
#     pos_cur = getattr(data, "pos_cur").cpu().numpy()
#     pos_tar = getattr(data, "pos_tar").cpu().numpy()
#     node_mask = getattr(data, "node_mask").cpu().numpy()

#     l0_to_l1_edge_index_j_cur = getattr(data, "l0_to_l1_edge_index_j_cur").cpu().numpy()
#     l0_to_l1_edge_index_i_cur = getattr(data, "l0_to_l1_edge_index_i_cur").cpu().numpy()
#     cluster_centers_index = getattr(data, "cluster_centers_index").cpu().numpy()

#     pos_tar = intrinsic.norm_camera_plane_to_pixel(pos_tar, clip=True, round=True)
#     pos_cur = intrinsic.norm_camera_plane_to_pixel(pos_cur, clip=True, round=True)

#     jj, ii = pos_cur[node_mask].T
#     for j, i in zip(jj, ii):
#         cv2.circle(image, (j, i), radius=4, color=(255, 255, 255), thickness=cv2.FILLED)

#     p_j = pos_cur[l0_to_l1_edge_index_j_cur]
#     p_i = pos_tar[cluster_centers_index][l0_to_l1_edge_index_i_cur]

#     for src, dst in zip(p_j, p_i):
#         src = (src[0], src[1])
#         dst = (dst[0], dst[1])
#         cv2.line(image, src, dst, color=(0, 255, 0), thickness=2)
    
#     proj_wPo = getattr(data, "proj_wPo", None)
#     if proj_wPo is not None:
#         u, v = proj_wPo.cpu().numpy().flatten()
#         u, v = int(round(u)), int(round(v))
#         cv2.circle(image, (u, v), radius=8, color=(0, 0, 255), thickness=cv2.FILLED)
    
#     return image


def draw_graph(
    data: Union[GraphData, Batch], 
    image=None,
    plot_E0=True,
    color_point=True,
    current_frame=True,
    batch=-1
):
    """Draw graph structure (only draw edges from keypoints to cluster centers).

    Arguments:
    - data: GraphData directly from cns.sim.dataset.Dataset or frontend, 
            or batch of GraphData from cns.sim.dataset.DataLoader
    - image: background image, if is None, generate black background
    - plot_E0: True to plot Edges Type 0 else Edges Type 1
    - color_point: assign different colors for points belonging to different cluster
    - current_frame: plot graph structure of current frame or desired frame
    - batch: if `data` is Batch, specify which batch to visualize,
            default is -1, meaning the last batch
    
    Returns:
    - image: numpy color image, dtype=uint8, shape=(H, W, 3),
            H and W are the same as those in camera config
    """
    if isinstance(data, Batch):
        if batch < 0:
            batch_idx = getattr(data, "batch", None)
            num_batches = (batch_idx.max() + 1) if batch_idx is not None else 1
            batch = num_batches + batch
        data: GraphData = data.get_example(batch)

    intrinsic: CameraIntrinsic = getattr(data, "intrinsic")
    W, H = intrinsic.width, intrinsic.height
    if image is None:
        # image = np.zeros((H, W, 3), dtype=np.uint8)
        image = np.ones((H, W, 3), dtype=np.uint8) * 64
    else:
        assert isinstance(image, np.ndarray)
        iH, iW = image.shape[:2]
        # print("shapes", (iW, iH), (W, H))
        assert W == iW and H == iH, "Input image shape do not match camera intrinsic"
        image = image.copy()
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255.).clip(0, 255).astype(np.uint8)

    pos_tar = getattr(data, "pos_tar").cpu().numpy()
    pos_cur = getattr(data, "pos_cur").cpu().numpy() if current_frame else pos_tar
    node_mask = getattr(data, "node_mask").cpu().numpy() if current_frame else np.ones(len(pos_tar), bool)
    belong_cluster_index = getattr(data, "belong_cluster_index").cpu().numpy()
    cluster_centers_index = getattr(data, "cluster_centers_index").cpu().numpy()

    pos_tar = intrinsic.norm_camera_plane_to_pixel(pos_tar, clip=True, round=True)
    pos_cur = intrinsic.norm_camera_plane_to_pixel(pos_cur, clip=True, round=True)
    center_pos = pos_tar[cluster_centers_index]

    colors = CYCLE_COLORS
    Ncolor = len(colors)

    suffix = "_cur" if current_frame else "_tar"
    if plot_E0:
        if current_frame:
            l0_to_l1_edge_index_j_cur = getattr(data, "l0_to_l1_edge_index_j" + "_cur").cpu().numpy()
            l0_to_l1_edge_index_i_cur = getattr(data, "l0_to_l1_edge_index_i" + "_cur").cpu().numpy()
            belong_cluster_index = belong_cluster_index[node_mask]
        else:
            l0_to_l1_edge_index_j_cur = np.arange(len(node_mask))
            l0_to_l1_edge_index_i_cur = belong_cluster_index

        # plot edges
        p_j = pos_cur[l0_to_l1_edge_index_j_cur]
        p_i = center_pos[l0_to_l1_edge_index_i_cur]
        for src, dst, c_idx in zip(p_j, p_i, l0_to_l1_edge_index_i_cur):
            src = (src[0], src[1])
            dst = (dst[0], dst[1])
            color = colors[c_idx % Ncolor]
            cv2.line(image, src, dst,
                # color=(0, 255, 0),
                color=color,
                thickness=2, lineType=cv2.LINE_AA)

        # plot source nodes
        for i, pos in enumerate(pos_cur[node_mask]):
            color = colors[belong_cluster_index[i] % Ncolor] if color_point else (255,)*3
            cv2.circle(image, (pos[0], pos[1]), 
                radius=4, color=color, thickness=cv2.FILLED)
            cv2.circle(image, (pos[0], pos[1]),
                radius=5, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    else:
        # plot edges
        l1_dense_edge_index_cur = getattr(data, "l1_dense_edge_index" + suffix).cpu().numpy()
        ej, ei = l1_dense_edge_index_cur
        for src, dst in zip(center_pos[ej], center_pos[ei]):
            src = (src[0], src[1])
            dst = (dst[0], dst[1])
            cv2.line(image, src, dst, color=(0, 255, 0),
                thickness=2, lineType=cv2.LINE_AA)

    # plot target nodes
    for i, pos in enumerate(center_pos):
        color = colors[i % Ncolor] if color_point else (255,)*3
        cv2.circle(image, (pos[0], pos[1]),
            radius=6, color=color, thickness=cv2.FILLED)
        cv2.circle(image, (pos[0], pos[1]),
            radius=7, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return image


def show_graph(data: Union[GraphData, Batch], batch=-1):
    image = draw_graph(data, batch=batch)
    cv2.imshow("graph", image)
    return cv2.waitKey(1)
