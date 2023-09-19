import numpy as np
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity


class ErrorHoldingStopPolicy(object):
    def __init__(self, waiting_time, conduct_thresh):
        self.waiting_time = waiting_time
        self.conduct_thresh = conduct_thresh
        self.min_err = None
        self.cur_err = float("inf")
        self.prev_t = None
        self.prev_err = None
        self.accum_time = 0
    
    def reset(self):
        self.min_err = None
        self.prev_t = None
        self.prev_err = None
        self.accum_time = 0
    
    def calculate_error(self, data):
        raise NotImplementedError
    
    def __call__(self, data, timestamp: float) -> bool:
        if data is None:
            self.reset()
            self.cur_err = float("inf")
            return False
        
        cur_err = self.calculate_error(data)
        self.cur_err = cur_err

        if cur_err > self.conduct_thresh:
            self.reset()
            return False
        
        if self.min_err is None or cur_err < 0.9 * self.min_err:
            self.min_err = cur_err
            self.prev_t = timestamp
            self.prev_err = cur_err
            self.accum_time = 0
            return False
        
        thresh = 1.1 * self.min_err + 5e-4
        # thresh = 1.1 * self.min_err + 2e-4
        a, b = self.prev_err, cur_err
        t0, t1 = self.prev_t, timestamp
        self.prev_t = timestamp
        self.prev_err = cur_err

        if a < thresh and thresh <= b:
            dt = (thresh - a) / (b - a) * (t1 - t0)
        elif a >= thresh and thresh > b:
            dt = (thresh - b) / (a - b) * (t1 - t0)
        elif a <= thresh and b <= thresh:
            dt = t1 - t0
        else:
            dt = 0
            # dt = (t1 - t0) * 0.01  # avoid waiting for too much time
        self.accum_time += dt

        print("[INFO] min_err = {:.3e}, cur_err = {:.3e}, dt = {:.3f}, accum = {:.3f}"
              .format(self.min_err, cur_err, dt, self.accum_time))

        return self.accum_time > self.waiting_time


# class PixelStopPolicy(ErrorHoldingStopPolicy):
#     def __init__(self, waiting_time, conduct_thresh=0.01):
#         super().__init__(waiting_time, conduct_thresh)
    
#     def calculate_error(self, data):
#         delta_pos = getattr(data, "pos_cur") - getattr(data, "pos_tar")
#         delta_pos = torch.abs(delta_pos[getattr(data, "node_mask")])
#         delta_pos = delta_pos.squeeze(0).cpu().numpy()
#         delta_pos = np.linalg.norm(delta_pos, axis=-1)
#         # mask = stats.zscore(delta_pos) < 2
#         # mask = delta_pos < np.percentile(delta_pos, 90)
#         mask = delta_pos < np.percentile(delta_pos, 95)
#         delta_pos = delta_pos[mask]
#         mean_pix_err = delta_pos.mean()
#         return mean_pix_err


class PixelStopPolicy(ErrorHoldingStopPolicy):
    def __init__(self, waiting_time, conduct_thresh=0.01):
        super().__init__(waiting_time, conduct_thresh)
        self.weight = None
        self.witness_count = None
        self.decay = 0.95
        self.maximum = 1.0 / (1 - self.decay)  # solve from: x = decay * x + 1
    
    def reset(self):
        if isinstance(self.weight, np.ndarray):
            self.weight.fill(0)
            self.witness_count.fill(0)
        return super().reset()
    
    def calculate_error(self, data):
        delta_pos = (getattr(data, "pos_cur") - getattr(data, "pos_tar")).cpu().numpy()
        node_mask = getattr(data, "node_mask").cpu().numpy()

        if self.weight is None or len(self.weight) != len(node_mask):
            # initialize weight and witness count
            self.weight = np.zeros(len(node_mask), dtype=np.float32)
            self.witness_count = np.zeros(len(node_mask), dtype=np.int64)
        
        pre_is_0 = self.witness_count == 0
        self.witness_count[node_mask] += 1
        now_is_1 = self.witness_count == 1

        mask = pre_is_0 & now_is_1  # first witness mask
        self.weight[mask] = self.maximum  # set weight of first witness point to maximum 
        self.weight[~mask] = self.decay * self.weight[~mask] + node_mask[~mask]

        weight = self.weight[node_mask]
        delta_pos = np.linalg.norm(delta_pos[node_mask], axis=-1)

        error = weight * delta_pos
        mask = error < np.percentile(error, 90)
        weight = weight[mask]
        weight = weight / (weight.sum() + 1e-7)
        mean_err = np.sum(weight * delta_pos[mask])

        return mean_err


class SSIMStopPolicy(ErrorHoldingStopPolicy):
    def __init__(self, waiting_time, conduct_thresh=0.1):
        super().__init__(waiting_time, conduct_thresh)
    
    def calculate_error(self, data):
        if isinstance(data, (tuple, list)):
            cur_img, tar_img = data
        else:
            cur_img = data["cur_img"]
            tar_img = data["tar_img"]
        
        if cur_img.ndim == 3: cur_img = rgb2gray(cur_img)
        if tar_img.ndim == 3: tar_img = rgb2gray(tar_img)

        ssim = structural_similarity(tar_img, cur_img)
        error = 1 - ssim
        return error
