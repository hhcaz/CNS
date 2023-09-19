import cv2
import numpy as np
from typing import List
from dataclasses import dataclass
from ..utils.perception import CameraIntrinsic


@dataclass
class Correspondence(object):
    """ 
    - cur_pos[match[:, 1]] matches tar_pos[match[:, 0]]; 
    - cur_pos_aligned[i] matches tar_pos[i] (when valid_mask[i] is True)
    """
    intrinsic: CameraIntrinsic

    tar_img: np.ndarray  # (H, W, 3), float32
    tar_pos: np.ndarray  # (N_tar, 2), float32, 
                         # not normalized! range 0 ~ H-1 (or W-1)
    cur_img: np.ndarray  # (H, W, 3), float32
    cur_pos: np.ndarray  # (N_cur, 2), float32

    match: np.ndarray  # (M, 2), int64, where 
    valid_mask: np.ndarray  # (N_tar,), bool8
    cur_pos_aligned: np.ndarray  # (N_tar, 2), float32

    detector_name: str = "Unknown"

    # indicate whether the graph should be reconstructed
    tar_img_changed: bool = True

    @classmethod
    def concat(cls, corr: List["Correspondence"]):
        valid_corr = [c for c in corr if c is not None]
        if len(valid_corr) == 0:
            return None
        else:
            tar_offset = 0
            cur_offset = 0
            matches_shift = []

            for c in valid_corr:
                match = c.match.copy()
                match[:, 0] += tar_offset
                match[:, 1] += cur_offset

                tar_offset += c.tar_pos.shape[0]
                cur_offset += c.cur_pos.shape[1]
                matches_shift.append(match)
            matches_shift = np.concatenate(matches_shift, axis=0)

            return Correspondence(
                # suppose intrinsics, target and current images are
                # the same for each detector
                intrinsic=valid_corr[0].intrinsic, 

                tar_img=valid_corr[0].tar_img,
                cur_img=valid_corr[0].cur_img,

                tar_pos=np.concatenate([c.tar_pos for c in valid_corr], axis=0),
                cur_pos=np.concatenate([c.cur_pos for c in valid_corr], axis=0),

                match=matches_shift, 
                valid_mask=np.concatenate([c.valid_mask for c in valid_corr], axis=0), 
                cur_pos_aligned=np.concatenate([c.cur_pos_aligned for c in valid_corr], axis=0),

                detector_name=" + ".join([c.detector_name for c in valid_corr]),
                tar_img_changed=any(c.tar_img_changed for c in valid_corr)
            )


class FrontendBase(object):
    def update_target_frame(self, image, mask=None) -> bool:
        raise NotImplementedError
    
    def process_current_frame(self, image) -> Correspondence:
        raise NotImplementedError
    
    def reset_intrinsic(self, intrinsic: CameraIntrinsic):
        self.intrinsic = intrinsic


class FrontendConcatWrapper(object):
    def __init__(self, frontends: List[FrontendBase]):
        self.frontends = frontends
    
    @property
    def intrinsic(self):
        """Suppose all frontends share the same intrinsic"""
        return self.frontends[0].intrinsic
    
    def update_target_frame(self, image, mask=None):
        return all([f.update_target_frame(image, mask) for f in self.frontends])
    
    def process_current_frame(self, image):
        corr = [f.process_current_frame(image) for f in self.frontends]
        corr = Correspondence.concat(corr)
        return corr
    
    def reset_intrinsic(self, intrinsic: CameraIntrinsic):
        for frontend in self.frontends:
            frontend.reset_intrinsic(intrinsic)


def to_f32(image: np.ndarray):
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.
    else:
        image = image.astype(np.float32)
    return image


def to_u8(image: np.ndarray):
    if np.issubdtype(image.dtype, np.floating):
        image = image * 255.
    image = image.astype(np.uint8)
    return image


def put_text(image: np.ndarray, text=[], small_text=[]):
    H = image.shape[0]
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(image, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(image, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    return image


def plot_corr(corr: Correspondence, color=None, show_keypoints=False, margin=10):
    H0, W0 = corr.tar_img.shape[:2]
    H1, W1 = corr.cur_img.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out_size = (H, W) if len(corr.tar_img.shape) == 2 else \
               (H, W, corr.tar_img.shape[-1])
    out = 255 * np.ones(out_size, np.uint8)
    out[:H0, :W0] = to_u8(corr.tar_img)
    out[:H1, W0+margin:] = to_u8(corr.cur_img)

    if len(out_size) == 2:
        out = np.stack([out]*3, -1)

    if show_keypoints:
        black = (0, 0, 0)
        white = (255, 255, 255)

        kpts0 = np.round(corr.tar_pos).astype(int)
        kpts1 = np.round(corr.cur_pos).astype(int)

        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)


    mkpts0 = corr.tar_pos[corr.valid_mask]
    mkpts1 = corr.cur_pos_aligned[corr.valid_mask]

    # # this results the same visual result but points are in different order
    # mkpts0 = corr.tar_pos[corr.match[:, 0]]
    # mkpts1 = corr.cur_pos[corr.match[:, 1]]

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1] if color is not None \
        else np.random.randint(low=0, high=256, size=(len(mkpts0), 3))

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
    
    out = put_text(out, text=[
        corr.detector_name.upper(),
        'Keypoints: {}:{}'.format(corr.tar_pos.shape[0], corr.cur_pos.shape[0]),
        'Matches: {}'.format(np.sum(corr.valid_mask))
    ])

    return out


def show_corr(corr: Correspondence, color=None, show_keypoints=False, margin=10):
    out = plot_corr(corr, color, show_keypoints, margin)
    cv2.imshow("Target | Current", out)
    return cv2.waitKey(1)
