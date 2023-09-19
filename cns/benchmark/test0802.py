import os
import numpy as np
import pybullet as p
from pathlib import Path
from .pipeline import VisOpt
from .environment import BenchmarkEnvRender, BenchmarkEnvAffine
from .stop_policy import PixelStopPolicy, SSIMStopPolicy
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline
from .benchmark import run_benchmark


def get_benchmark_results_root(ckpt_path: str, category: str = None):
    if category is None or len(category) == 0:
        category = "default"
    
    here = os.path.dirname(__file__)
    if ckpt_path is None:
        root = os.path.join(here, "results", category)
    else:
        p = Path(os.path.abspath(ckpt_path))
        root = os.path.join(here, "results", category, p.parent.name)

    return root


def test0802():
    ckpt_paths = [
        # "checkpoints/06_03_00_22_21_graph_128/checkpoint_best.pth",
        # "checkpoints/08_06_01_10_59_graph5_128_hy/checkpoint_last.pth",
        # "checkpoints/hybrid_from_st/checkpoint_best.pth"
        # "checkpoints/hybrid_from_scratch_large/checkpoint_best.pth"
        "checkpoints/hybrid_from_scratch_large2/checkpoint_best.pth"

        # "checkpoints/ibvs_Lmean/ibvs_config.json"
    ]

    for ckpt_path in ckpt_paths:
        np.random.seed(0)

        pipeline = CorrespondenceBasedPipeline(
            # detector="AKAZE",
            detector="AKAZE:0",
            ckpt_path=ckpt_path,
            intrinsic=None,
            vis=VisOpt.MATCH,
            # ransac=False
        )

        stop_policy = PixelStopPolicy(
            waiting_time=0.5, 
            conduct_thresh=0.01
        )

        env = BenchmarkEnvRender(
            scale=1.0,
            section="A"
        )

        result_folder = os.path.join(
            get_benchmark_results_root(ckpt_path, "hybrid_pbvs"),
            # get_benchmark_results_root(ckpt_path, "hybrid_pbvs_wo_ransac"),
            env.prefer_result_folder()
        )

        # run_benchmark(
        #     env, pipeline, stop_policy, None, 
        #     record=False,
        #     skip_saved=True
        # )

        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            # skip_saved=True
            skip_saved=False
        )

        p.disconnect()



def benchmark_raft_ibvs():
    ckpt_path = "checkpoints/raft_ibvs/checkpoint.pth"
    category = "hybrid_pbvs"

    pipeline = ImageBasedPipeline(
        ckpt_path=ckpt_path,
        vis=VisOpt.ALL
    )
    env = BenchmarkEnvRender(
        scale=1.0,
        section="A"
    )
    stop_policy = SSIMStopPolicy(
        waiting_time=0.5, 
        conduct_thresh=0.01
    )

    result_folder = os.path.join(
        get_benchmark_results_root(ckpt_path, category),
        env.prefer_result_folder()
    )

    print("[INFO] Result folder: {}".format(result_folder))
    run_benchmark(
        env, pipeline, stop_policy, result_folder, 
        record=True,
        skip_saved=False
    )

    p.disconnect()


def benchmark_raft_ibvs2():
    ckpt_path = "checkpoints/raft_ibvs/checkpoint.pth"
    category = "raft_ibvs_on_image"

    pipeline = ImageBasedPipeline(
        ckpt_path=ckpt_path,
        vis=VisOpt.ALL
    )
    env = BenchmarkEnvAffine(
        scale=1.0,
        one_image=BenchmarkEnvAffine.stars1953
    )
    stop_policy = SSIMStopPolicy(
        waiting_time=0.5, 
        conduct_thresh=0.01
    )

    result_folder = os.path.join(
        get_benchmark_results_root(ckpt_path, category),
        env.prefer_result_folder()
    )

    print("[INFO] Result folder: {}".format(result_folder))
    run_benchmark(
        env, pipeline, stop_policy, result_folder, 
        record=True,
        skip_saved=False
    )

    p.disconnect()



if __name__ == "__main__":
    # test0802()
    # benchmark_raft_ibvs()
    benchmark_raft_ibvs2()
