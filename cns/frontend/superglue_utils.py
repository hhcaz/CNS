import cv2
import torch
import numpy as np
import matplotlib.cm as cm
from ..thirdparty.SuperGluePretrainedNetwork.models.matching import Matching
from ..thirdparty.SuperGluePretrainedNetwork.models.utils import make_matching_plot_fast


def get_matching(device="cuda:0") -> Matching:
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': "indoor",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)
    return matching


def detect_feat(frame_tensor, super_point):
    feat = super_point({"image": frame_tensor})
    feat["scores"] = list(feat["scores"])
    feat["image"] = frame_tensor
    return feat


def add_suffix(feat, suffix="0"):
    return {(k+suffix):v for k, v in feat.items()}


def sort_and_trim_feat(feat):
    batch_size = len(feat["keypoints"])
    if batch_size <= 1:
        return feat
    
    kp_nums = [feat["scores"][i].size(0) for i in range(batch_size)]
    min_kp_num = min(kp_nums)

    for i in range(batch_size):
        _, sort_indices = torch.sort(feat["scores"][i], descending=True)
        selected_indices = sort_indices[:min_kp_num]

        for key in feat.keys():
            if isinstance(feat[key], torch.Tensor):
                continue

            sel_dim = 1 if "descriptors" in key else 0
            feat[key][i] = torch.index_select(feat[key][i], sel_dim, selected_indices)
    return feat


def make_matching_plot_fast(image0, image1, kpts0, kpts1, matches, color, text,
                            show_keypoints=False, margin=10, small_text=[]):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out_size = (H, W) if len(image0.shape) == 2 else (H, W, image0.shape[-1])
    out = 255*np.ones(out_size, np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1

    if len(out_size) == 2:
        out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    return out


def draw_matches(image0, image1, kpts0, kpts1, matches, confidence, 
                 matching: Matching, show_keypoints=False, margin=10):
    color = cm.jet(confidence)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(matches))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]

    out = make_matching_plot_fast(
        image0, image1, kpts0, kpts1, matches, 
        color, text, show_keypoints, margin, small_text
    )

    return out

