import os
import time
import pickle
import numpy as np
from typing import Union, Optional, Dict
from .environment import BenchmarkEnvRender, BenchmarkEnvAffine
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline
from .stop_policy import ErrorHoldingStopPolicy


class BaseRecord(object):
    def __init__(self, folder):
        self.folder = folder
        self.trajs = []
        self.rates = []
        self.timings = dict()
        self.errors = []

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
    
    def clear(self):
        self.trajs.clear()
        self.rates.clear()
        self.errors.clear()
        self.timings.clear()

    def append_traj(
        self, 
        cur_bcT: np.ndarray, 
        vel: np.ndarray, 
        timing: Optional[Dict] = None, 
        error: float = "Unknown",
        **kwargs
    ):
        self.trajs.append(cur_bcT)
        self.rates.append(vel)
        self.errors.append(error)

        if timing is None:
            timing = {"timestamp": time.time()}

        for k in timing:
            if k not in self.timings:
                self.timings[k] = []
            self.timings[k].append(timing[k])
    
    def finalize(self, fname: str, tar_bcT: np.ndarray, ini_bcT: np.ndarray):
        trajs = np.stack(self.trajs, axis=0)
        rates = np.stack(self.rates, axis=0)
        errors = np.array(self.errors)
        timings = {k: np.array(v) for k, v in self.timings.items()}
        fname = os.path.join(self.folder, fname)

        save_dict = dict(
            tar_bcT=tar_bcT,  # (4, 4)
            ini_bcT=ini_bcT,  # (4, 4)
            trajs=trajs,  # (N, 4, 4)
            rates=rates,  # (N, 6),
            errors=errors,  # (N,)
            **timings
        )

        return fname, save_dict


class NpzRecord(BaseRecord):
    def finalize(self, fname: str, tar_bcT: np.ndarray, ini_bcT: np.ndarray):
        fname, save_dict = super().finalize(fname, tar_bcT, ini_bcT)
        np.savez(fname, **save_dict)
        print("[INFO] Result saved to {}".format(fname))
        self.clear()


class PickleRecord(BaseRecord):
    def __init__(self, folder):
        super().__init__(folder)
        self.extra_data_traj = []
    
    def clear(self):
        self.extra_data_traj.clear()
        return super().clear()
    
    def append_traj(
        self, 
        cur_bcT: np.ndarray, 
        vel: np.ndarray, 
        timing: Optional[Dict] = None, 
        error: float = "Unknown",
        extra_data = None
    ):  
        if len(self.extra_data_traj) and extra_data is not None:
            if isinstance(extra_data, dict):
                extra_data["tar_img"] = None
            else:
                setattr(extra_data, "tar_img", None)
            # avoid saving duplicate target image
        self.extra_data_traj.append(extra_data)
        return super().append_traj(cur_bcT, vel, timing, error)
    
    def finalize(self, fname: str, tar_bcT: np.ndarray, ini_bcT: np.ndarray):
        fname, save_dict = super().finalize(fname, tar_bcT, ini_bcT)
        save_dict["extra_data_traj"] = self.extra_data_traj
        with open(fname, "wb") as fp:
            pickle.dump(save_dict, fp)
        print("[INFO] Result saved to {}".format(fname))
        self.clear()


def run_benchmark(
    env: Union[BenchmarkEnvRender, BenchmarkEnvAffine],
    pipeline: Union[CorrespondenceBasedPipeline, ImageBasedPipeline],
    stop_policy: ErrorHoldingStopPolicy,
    result_folder: str,
    record: bool = False,
    skip_saved: bool = True,
):
    npz_recorder = NpzRecord(result_folder) if record else None
    if skip_saved and result_folder and record:
        env.skip_indices(env.get_global_indices_of_saved(result_folder))
    
    requires_intrinsic = isinstance(pipeline, CorrespondenceBasedPipeline)
    need_rgb_channel_shuffle = pipeline.REQUIRE_IMAGE_FORMAT != env.RETURN_IMAGE_FORMAT
    print("Need to reorder RGB channel: {}".format(need_rgb_channel_shuffle))
    
    for i in range(len(env)):
        env.clear_debug_items()
        
        tar_img = env.init(i)
        if need_rgb_channel_shuffle:
            tar_img = np.ascontiguousarray(tar_img[:, :, ::-1])
        tPo_norm = np.linalg.norm(env.target_wcT[:3, 3])

        if requires_intrinsic:
            # pipeline.frontend.intrinsic = env.camera.intrinsic
            pipeline.frontend.reset_intrinsic(env.camera.intrinsic)
        pipeline.set_target(tar_img, dist_scale=tPo_norm, intrinsic=env.camera.intrinsic)
        stop_policy.reset()

        while True:
            cur_img = env.observation()
            if need_rgb_channel_shuffle:
                cur_img = np.ascontiguousarray(cur_img[:, :, ::-1])
            
            vel, data, timing = pipeline.get_control_rate(cur_img)
            need_stop = (
                stop_policy(data, env.steps*env.dt) or 
                env.exceeds_maximum_steps() or
                data is None
                or env.camera_under_ground()
                # or env.abnormal_pose()
            )

            if record:
                npz_recorder.append_traj(
                    env.current_wcT, vel, timing, stop_policy.cur_err)

            if need_stop:
                break
            
            env.action(vel * 2)
            # env.action(vel)

        print("[INFO] Round: {}/{}".format(i+1, len(env)))
        print("[INFO] Steps: {}/{}".format(env.steps, env.max_steps))
        env.print_pose_err()
        print("---------------------------------------------------")

        if record:
            npz_recorder.finalize(
                env.prefer_result_fname(i),
                env.target_wcT, 
                env.initial_wcT
            )


if __name__ == "__main__":
    from .pipeline import VisOpt
    from .stop_policy import PixelStopPolicy

    env = BenchmarkEnvRender(
        scale=1.0,
        section="A"
    )

    # original checkpoint
    ckpt_path = "checkpoints/05_18_14_17_58_graph_128/checkpoint_best.pth"

    # # scale ablation
    # ckpt_path = "checkpoints/05_16_06_41_51_graph_128/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_16_06_42_46_graph_128_scale_before_net/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_16_06_42_50_graph_128_wo_scale_prior/checkpoint_best.pth"

    # # data generation ablation
    # ckpt_path = "checkpoints/05_17_04_28_24_graph_128_data_ideal/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_17_04_29_18_graph_128_data_simple_noise/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_17_04_30_07_graph_128_data_uniform/checkpoint_best.pth"

    # # network structure ablation
    # ckpt_path = "checkpoints/05_17_22_53_43_graph_128_edge_conv/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_17_22_54_55_graph_128_simple_gru/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_17_22_55_25_graph_128_wo_gru/checkpoint_best.pth"
    # ckpt_path = "checkpoints/05_19_16_00_49_graph_128_no_cluster/checkpoint_best.pth"

    # # compare with IBVS
    # ckpt_path = "checkpoints/ibvs_Ltar/ibvs_config.json"
    # ckpt_path = "checkpoints/ibvs_Lmean/ibvs_config.json"

    pipeline = CorrespondenceBasedPipeline(
        # detector="ORB",
        # detector="AKAZE",
        # detector="BRISK",
        # detector="SIFT",
        detector="SuperGlue",
        ckpt_path=ckpt_path,
        intrinsic=None,
        # vis=VisOpt.MATCH
    )

    stop_policy = PixelStopPolicy(waiting_time=0.8, conduct_thresh=0.01)
    result_folder = env.prefer_result_folder(
        prefix=os.path.join(
            os.path.dirname(ckpt_path),
            "benchmark"
        )
    )

