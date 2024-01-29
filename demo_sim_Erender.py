import torch
import numpy as np
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.stop_policy import PixelStopPolicy
from cns.benchmark.environment import BenchmarkEnvRender
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt


def set_seed(seed=2023):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_demo():
    # pipeline = CorrespondenceBasedPipeline.from_file("pipeline.json")
    pipeline = CorrespondenceBasedPipeline(
        detector="AKAZE",
        # detector="SuperGlue:0123",
        # ckpt_path="checkpoints/cns.pth",
        ckpt_path="checkpoints/cns_state_dict.pth",
        intrinsic=CameraIntrinsic.default(),
        device="cuda:0",
        ransac=True,
        vis=VisOpt.MATCH|VisOpt.GRAPH
    )
    stop_policy = PixelStopPolicy(waiting_time=0.5, conduct_thresh=5e-3)
    env = BenchmarkEnvRender(scale=1, section="A")

    results = {
        "gid": [], 
        "initial_pose": [], 
        "final_pose": [], 
        "desired_pose": [], 
        "steps": []
    }
    for i in range(len(env)):
        env.clear_debug_items()
        
        tar_img = env.init(i)  # uint8, bgr image
        tPo_norm = np.linalg.norm(env.target_wcT[:3, 3])

        set_seed()
        pipeline.frontend.reset_intrinsic(env.camera.intrinsic)
        pipeline.set_target(tar_img, dist_scale=tPo_norm)
        stop_policy.reset()

        while True:
            cur_img = env.observation()  # uint8, bgr image
            vel, data, timing = pipeline.get_control_rate(cur_img)
            need_stop = (
                stop_policy(data, env.steps*env.dt) or 
                env.exceeds_maximum_steps() or
                env.camera_under_ground() or
                data is None
            )

            if need_stop:
                break
            
            env.action(vel * 2)  # gain=2 to speed up

        print("[INFO] Round: {}/{}".format(i+1, len(env)))
        print("[INFO] Steps: {}/{}".format(env.steps, env.max_steps))
        env.print_pose_err()
        print("-"*80)

        results["gid"].append(env.global_indices[i])
        results["initial_pose"].append(env.initial_wcT)
        results["final_pose"].append(env.current_wcT)
        results["desired_pose"].append(env.target_wcT)
        results["steps"].append(env.steps)

    for k in results.keys(): results[k] = np.asarray(results[k])
    np.savez("sim_Erender_result.npz", **results)


if __name__ == "__main__":
    run_demo()

