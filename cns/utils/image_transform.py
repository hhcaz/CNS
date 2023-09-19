import numpy as np
import skimage.transform as transform


def scale_to_fit(image, box_wh, order=1, minimize=True):
    box_w, box_h = box_wh
    input_h, input_w = image.shape[:2]

    scale_w = box_w / input_w
    scale_h = box_h / input_h

    if minimize:
        scale_factor = min(scale_w, scale_h)
    else:
        scale_factor = max(scale_w, scale_h)

    output_w = round(scale_factor * input_w)
    output_h = round(scale_factor * input_h)

    image = transform.resize(image, (output_h, output_w), order=order)
    
    def tform_points(points_xy):
        points_xy[..., 0] *= scale_factor
        points_xy[..., 1] *= scale_factor
        return points_xy
    
    def inv_tform_points(points_xy):
        points_xy[..., 0] /= scale_factor
        points_xy[..., 1] /= scale_factor
        return points_xy
    
    return image, tform_points, inv_tform_points


def pad_to_ratio(image, wh_ratio, mode='constant', fill=0):
    input_h, input_w = image.shape[:2]
    input_wh_ratio = input_w / input_h

    if wh_ratio > input_wh_ratio:
        output_w = round(input_h * wh_ratio)
        output_h = input_h
        pad_width = output_w - input_w
        pad_height = 0
    else:
        output_h = round(input_w / wh_ratio)
        output_w = input_w
        pad_height = output_h - input_h
        pad_width = 0

    pad_left   = pad_width // 2
    pad_top    = pad_height // 2
    pad_right  = pad_width - pad_left
    pad_bottom = pad_height - pad_top

    if image.ndim == 3:
        p2d = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    elif image.ndim == 2:
        p2d = ((pad_top, pad_bottom), (pad_left, pad_right))

    if mode == 'constant':
        image = np.pad(image, p2d, 'constant', constant_values=fill)
    else:
        image = np.pad(image, p2d, mode)
    
    def tform_points(points_xy):
        points_xy[..., 0] += pad_left
        points_xy[..., 1] += pad_top
        return points_xy
    
    def inv_tform_points(points_xy):
        points_xy[..., 0] -= pad_left
        points_xy[..., 1] -= pad_top
        return points_xy
    
    return image, tform_points, inv_tform_points


def rot90(image, k=1):
    h, w = image.shape[:2]
    image = np.rot90(image, k, axes=(0, 1))
    
    def tform_points(points_xy):
        px, py = points_xy[..., 0], points_xy[..., 1]
        if k%4 == 1:
            px, py = py, w - px
        elif k%4 == 2:
            px, py = w - px, h - py
        elif k%4 == 3:
            px, py = h - py, px
        return np.stack([px, py], axis=-1)
    
    def inv_tform_points(points_xy):
        px, py = points_xy[..., 0], points_xy[..., 1]
        if k%4 == 1:
            px, py = w - py, px
        elif k%4 == 2:
            px, py = w - px, h - py
        elif k%4 == 3:
            px, py = py, h - px
        return np.stack([px, py], axis=-1)
    
    return image, tform_points, inv_tform_points
