import os
import numpy as np
from typing import Union
from .environment import BenchmarkEnvRender, BenchmarkEnvAffine
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline
from .stop_policy import ErrorHoldingStopPolicy, PixelStopPolicy
from .benchmark import NpzRecord


def run_benchmark(
    env: Union[BenchmarkEnvRender, BenchmarkEnvAffine],
    pipeline: Union[CorrespondenceBasedPipeline, ImageBasedPipeline],
    stop_policy: ErrorHoldingStopPolicy,
    result_folder: str,
    record: bool = False,
    skip_saved: bool = True,
    est_dist=0.7,
):
    npz_recorder = NpzRecord(result_folder) if record else None
    if skip_saved:
        env.skip_indices(env.get_global_indices_of_saved(result_folder))
    
    requires_intrinsic = isinstance(pipeline, CorrespondenceBasedPipeline)
    need_rgb_channel_shuffle = pipeline.REQUIRE_IMAGE_FORMAT != env.RETURN_IMAGE_FORMAT
    print("Need to reorder RGB channel: {}".format(need_rgb_channel_shuffle))
    
    for i in range(len(env)):
        env.clear_debug_items()
        
        tar_img = env.init(i)
        if need_rgb_channel_shuffle:
            tar_img = np.ascontiguousarray(tar_img[:, :, ::-1])
        tPo_norm = est_dist

        if requires_intrinsic:
            pipeline.frontend.intrinsic = env.camera.intrinsic
        pipeline.set_target(tar_img, tPo_norm)
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
            )

            if record:
                npz_recorder.append_traj(
                    env.current_wcT, vel, timing, stop_policy.cur_err)

            if need_stop:
                break
            
            env.action(vel * 2)

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
    import pybullet as p

    ckpt_path = "checkpoints/06_03_00_22_21_graph_128/checkpoint_best.pth"
    pipeline = CorrespondenceBasedPipeline(
        detector="AKAZE",
        ckpt_path=ckpt_path,
        intrinsic=None,
        # vis=VisOpt.MATCH,
        # vis=VisOpt.KP,
    )
    stop_policy = PixelStopPolicy(
        waiting_time=0.5, 
        # conduct_thresh=0.01
        conduct_thresh=4e-3
    )

    for est_dist in [0.25, 0.5, 1, 2]:
        np.random.seed(0)

        env = BenchmarkEnvRender(
            scale=1.0,
            section="A"
        )

        here = os.path.dirname(__file__)
        result_folder = os.path.join(
            here, 
            "results", 
            "inaccurate_dist", 
            "est_dist={:.2f}m".format(est_dist),
            env.prefer_result_folder()
        )

        run_benchmark(
            env=env,
            pipeline=pipeline, 
            stop_policy=stop_policy,
            result_folder=result_folder, 
            record=True,
            skip_saved=True,
            est_dist=est_dist
        )

        p.disconnect()
