import os
import numpy as np
import pybullet as p
from pathlib import Path
from .pipeline import VisOpt
from .stop_policy import PixelStopPolicy, SSIMStopPolicy
from .environment import BenchmarkEnvRender, BenchmarkEnvAffine
from .pipeline import CorrespondenceBasedPipeline, ImageBasedPipeline
from .benchmark import run_benchmark

# import models
from ..ablation import *
from ..reimpl import *


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


def benchmark_scale_info():
    ckpt_paths = [
        # "checkpoints/05_18_14_17_58_graph_128/checkpoint_best.pth",
        # "checkpoints/05_19_14_17_58_graph_128/checkpoint_last.pth",
        # "checkpoints/06_02_07_08_59_graph_128/checkpoint_last.pth",
        "checkpoints/06_03_00_22_21_graph_128/checkpoint_best.pth",
        "checkpoints/05_16_06_42_46_graph_128_scale_before_net/checkpoint_best.pth",
        "checkpoints/05_16_06_42_50_graph_128_wo_scale_prior/checkpoint_best.pth",
    ]
    category = "ablation_on_scale_info"

    for ckpt_path in ckpt_paths:
        np.random.seed(0)

        pipeline = CorrespondenceBasedPipeline(
            detector="AKAZE",
            ckpt_path=ckpt_path,
            intrinsic=None,
            # vis=VisOpt.MATCH,
            # vis=VisOpt.KP,
        )

        stop_policy = PixelStopPolicy(
            waiting_time=0.5, 
            conduct_thresh=0.01
        )

        for scale in [1.0, 0.2, 5.0]:
            env = BenchmarkEnvRender(
                scale=scale,
                section="A"
            )

            result_folder = os.path.join(
                get_benchmark_results_root(ckpt_path, category),
                env.prefer_result_folder()
            )

            print("[INFO] Result folder: {}".format(result_folder))
            run_benchmark(
                env, pipeline, stop_policy, result_folder, 
                record=True,
                skip_saved=True
            )

            p.disconnect()


def benchmark_gvs_on_render_env(ckpt_paths, category):
    for ckpt_path in ckpt_paths:
        pipeline = CorrespondenceBasedPipeline(
            detector="AKAZE",
            ckpt_path=ckpt_path,
            intrinsic=None,
            # vis=VisOpt.MATCH
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
            get_benchmark_results_root(ckpt_path, category),
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )
        
        p.disconnect()


def benchmark_data_generation():
    ckpt_paths = [
        "checkpoints/05_17_04_28_24_graph_128_data_ideal/checkpoint_best.pth",
        "checkpoints/05_17_04_29_18_graph_128_data_simple_noise/checkpoint_best.pth",
        "checkpoints/05_17_04_30_07_graph_128_data_uniform/checkpoint_best.pth",
        "checkpoints/06_06_21_08_10_graph_128_data_all_uniform/checkpoint_best.pth"
    ]
    category = "ablation_on_data_generation"
    benchmark_gvs_on_render_env(ckpt_paths, category)


def benchmark_network_structure():
    ckpt_paths = [
        "checkpoints/05_17_22_53_43_graph_128_edge_conv/checkpoint_best.pth",
        "checkpoints/05_17_22_54_55_graph_128_simple_gru/checkpoint_best.pth",
        "checkpoints/05_17_22_55_25_graph_128_wo_gru/checkpoint_best.pth",
        "checkpoints/05_19_16_00_49_graph_128_no_cluster/checkpoint_best.pth",
    ]
    category = "ablation_on_network_structure"
    benchmark_gvs_on_render_env(ckpt_paths, category)


def benchmark_ibvs():
    ckpt_paths = [
        "checkpoints/ibvs_Ltar/ibvs_config.json",
        "checkpoints/ibvs_Lmean/ibvs_config.json"
    ]
    category = "compare_with_ibvs"
    benchmark_gvs_on_render_env(ckpt_paths, category)


def benchmark_raft_ibvs():
    ckpt_path = "checkpoints/raft_ibvs/checkpoint.pth"
    category = "compare_with_raft_ibvs"

    pipeline = ImageBasedPipeline(
        ckpt_path=ckpt_path,
        vis=VisOpt.ALL
    )
    env = BenchmarkEnvRender(
        scale=1.0,
        section="S"
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
        skip_saved=True
    )

    p.disconnect()


def benchmark_detector():
    # ckpt_path = "checkpoints/05_18_14_17_58_graph_128/checkpoint_best.pth"
    ckpt_path = "checkpoints/06_03_00_22_21_graph_128/checkpoint_best.pth"
    category = "test_different_detector"

    threshs = {
        "ORB": 0.005,
        "SIFT": 0.002,
        "BRISK": 0.002,
        "SuperGlue": 0.01
    }

    for detector in ["ORB", "SIFT", "BRISK", "SuperGlue"]:
        pipeline = CorrespondenceBasedPipeline(
            detector=detector,
            ckpt_path=ckpt_path,
            intrinsic=None,
            # vis=VisOpt.MATCH
        )

        stop_policy = PixelStopPolicy(
            waiting_time=0.5, 
            conduct_thresh=threshs[detector]
        )

        env = BenchmarkEnvRender(
            scale=1.0,
            section="A"
        )

        result_folder = os.path.join(
            get_benchmark_results_root(None, category),
            detector,
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )

        p.disconnect()


def benchmark_seen_image():
    ckpt_paths = [
        # "checkpoints/05_18_14_17_58_graph_128/checkpoint_best.pth",
        # "checkpoints/06_03_00_22_21_graph_128/checkpoint_best.pth",
        # "checkpoints/05_06_02_59_50_ICRA2018_improved/checkpoint_best.pth",
        # "checkpoints/05_06_02_57_39_ICRA2021_improved/checkpoint_best.pth",
        "checkpoints/06_08_00_36_22_ICRA2018_improved_mse/checkpoint_best.pth",
        "checkpoints/06_08_00_37_55_ICRA2021_improved_mse/checkpoint_best.pth"
    ]
    category = "compare_with_image_vs_on_star1953"

    for i, ckpt_path in enumerate(ckpt_paths):
        # if i == 0:
        if "graph" in ckpt_path:
            pipeline = CorrespondenceBasedPipeline(
                detector="AKAZE",
                ckpt_path=ckpt_path,
                intrinsic=None,
                vis=VisOpt.MATCH
            )
            stop_policy = SSIMStopPolicy(
                # waiting_time=0.8, 
                waiting_time=999999, 
                conduct_thresh=0.1
            )
        else:
            pipeline = ImageBasedPipeline(
                ckpt_path=ckpt_path,
                vis=VisOpt.ALL
            )
            stop_policy = SSIMStopPolicy(
                # waiting_time=1.0, 
                waiting_time=999999, 
                conduct_thresh=0.5
            )
        
        env = BenchmarkEnvAffine(
            scale=1.0,
            aug=False,
            one_image=BenchmarkEnvAffine.stars1953
        )
        env.max_steps = 400  # avoid too much waiting time

        result_folder = os.path.join(
            get_benchmark_results_root(ckpt_path, category),
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )

        p.disconnect()


def benchmark_unseen_image():
    ckpt_paths = [
        # "checkpoints/05_18_14_17_58_graph_128/checkpoint_best.pth",
        "checkpoints/06_03_00_22_21_graph_128/checkpoint_best.pth",
        # "checkpoints/05_06_02_59_50_ICRA2018_improved/checkpoint_best.pth",
        # "checkpoints/05_06_02_57_39_ICRA2021_improved/checkpoint_best.pth",
        # "checkpoints/06_01_04_44_56_ICRA2018_improved_multi_scene/checkpoint_best.pth",
        # "checkpoints/06_01_04_45_50_ICRA2021_improved_multi_scene/checkpoint_best.pth",
    ]
    category = "compare_with_image_vs_on_mscoco14_150"

    for i, ckpt_path in enumerate(ckpt_paths):
        if "graph" in ckpt_path:
            pipeline = CorrespondenceBasedPipeline(
                detector="AKAZE",
                ckpt_path=ckpt_path,
                intrinsic=None,
                vis=VisOpt.MATCH
            )
            stop_policy = SSIMStopPolicy(
                # waiting_time=0.8, 
                waiting_time=999999, 
                conduct_thresh=0.1
            )
        else:
            pipeline = ImageBasedPipeline(
                ckpt_path=ckpt_path,
                vis=VisOpt.ALL
            )
            stop_policy = SSIMStopPolicy(
                # waiting_time=1.0, 
                waiting_time=999999, 
                conduct_thresh=0.5
            )
        
        env = BenchmarkEnvAffine(
            scale=1.0,
            aug=False,
            one_image=False
        )
        env.max_steps = 400  # avoid waiting too much time

        result_folder = os.path.join(
            get_benchmark_results_root(ckpt_path, category),
            env.prefer_result_folder()
        )

        print("[INFO] Result folder: {}".format(result_folder))
        run_benchmark(
            env, pipeline, stop_policy, result_folder, 
            record=True,
            skip_saved=True
        )

        p.disconnect()


if __name__ == "__main__":
    # benchmark_scale_info()  # tested

    # benchmark_data_generation()  # tested
    # benchmark_network_structure()  # tested
    # benchmark_ibvs()  # tested

    # benchmark_detector()  # tested

    benchmark_seen_image()
    # benchmark_unseen_image()

    # benchmark_raft_ibvs()
