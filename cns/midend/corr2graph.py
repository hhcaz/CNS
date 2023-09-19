import numpy as np
from .graph_gen import GraphGenerator
from ..frontend.utils import Correspondence


class Midend(object):
    def __init__(self):
        self.graph_generator_class = GraphGenerator
        self.graph_generator = None

    def get_graph_data(self, corr: Correspondence):
        if corr.tar_img_changed or self.graph_generator is None:
            xy = corr.intrinsic.pixel_to_norm_camera_plane(corr.tar_pos)
            self.graph_generator = self.graph_generator_class(xy)
        
        missing_kp_indices = np.nonzero(~corr.valid_mask)[0]
        xy = corr.intrinsic.pixel_to_norm_camera_plane(corr.cur_pos_aligned)
        graph_data = self.graph_generator.get_data(
            current_points=xy,
            missing_node_indices=missing_kp_indices
        )
        return graph_data

