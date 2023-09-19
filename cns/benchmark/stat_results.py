import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.spatial.transform import Rotation as R
from statsmodels.stats.proportion import proportion_confint
from .stat_scenes import stat_scenes


def process_npz(data, result_buffer: dict):
    tar_bcT = data["tar_bcT"]  # (4, 4)
    ini_bcT = data["ini_bcT"]  # (4, 4)
    trajs = data["trajs"]  # (N, 4, 4)

    final_dT = np.linalg.inv(trajs[-1]) @ tar_bcT
    final_du = R.from_matrix(final_dT[:3, :3]).as_rotvec()
    final_dt = final_dT[:3, 3]

    ini_dT = np.linalg.inv(ini_bcT) @ tar_bcT
    ini_du = R.from_matrix(ini_dT[:3, :3]).as_rotvec()
    ini_dt = ini_dT[:3, 3]

    local_vars = locals().copy()
    for attr in ["ini_du", "ini_dt", "final_du", "final_dt"]:
        for i, xyz in enumerate(["x", "y", "z"]):
            attr_xyz = attr + xyz  # ini_dux, ini_duy, ...
            if attr_xyz not in result_buffer:
                result_buffer[attr_xyz] = []
            result_buffer[attr_xyz].append(local_vars[attr][i])
        
        attr_norm = attr + "_norm"
        if attr_norm not in result_buffer:
            result_buffer[attr_norm] = []
        result_buffer[attr_norm].append(np.linalg.norm(local_vars[attr]))
    
    if "steps" not in result_buffer:
        result_buffer["steps"] = []
    result_buffer["steps"].append(len(trajs) - 1)         

    if "convergence" not in result_buffer:
        result_buffer["convergence"] = []
    result_buffer["convergence"].append(
        np.linalg.norm(final_dt) < 0.1 * np.linalg.norm(tar_bcT[:3, 3]) and \
        # np.linalg.norm(final_dt) < 0.5 and \
        np.linalg.norm(final_du) / np.pi * 180 < 10  # degree
    )

    if "total_time" in data.keys():
        if "total_time" not in result_buffer:
            result_buffer["total_time"] = []
        result_buffer["total_time"].append(np.mean(data["total_time"]))
    if "frontend_time" in data.keys():
        if "frontend_time" not in result_buffer:
            result_buffer["frontend_time"] = []
        result_buffer["frontend_time"].append(np.mean(data["frontend_time"]))
    if "backend_time" in data.keys():
        if "backend_time" not in result_buffer:
            result_buffer["backend_time"] = []
        result_buffer["backend_time"].append(np.mean(data["backend_time"]))   


def process_folder(folder: str):
    results = {
        "gid": []
    }
    files = glob.glob(os.path.join(folder, "*.npz"))
    files.sort()

    for i, f in enumerate(files):
        # print("[INFO] {}/{} | Processing file: {}".format(i+1, len(files), f))
        fname = os.path.split(f)[-1]
        fname_wo_ext = os.path.splitext(fname)[0]
        gid = int(fname_wo_ext.replace("scene_gid=", ""))
        results["gid"].append(gid)

        data = np.load(f)
        process_npz(data, results)
    
    return results


def gather_results(ckpt_folder):
    folders = glob.glob(os.path.join(ckpt_folder, "scale=*"))
    folders = [f for f in folders if not f.endswith(".xlsx")]

    for folder in folders:
        results = process_folder(folder)
        results = pd.DataFrame(results)
        results.to_excel(folder+".xlsx", index=False)


def st_metric(a: np.ndarray):
    if len(a) >= 2:
        std = np.std(a)
        mean = np.mean(a)

        if a.dtype in [bool, np.bool8]:
            success = np.sum(a)
            tries = len(a)
            interval = proportion_confint(
                count=success, nobs=tries, alpha=0.05,
                method="normal"
            )
        else:
            interval = st.t.interval(
                0.95, len(a)-1, loc=mean, scale=st.sem(a))
        
        return {
            "mean": mean,
            "std": std,
            "ci95": interval
        }

    else:
        mean = 0.0 if len(a) == 0 else float(a[0])
        return {
            "mean": mean,
            "std": 0.0,
            "ci95": (0.0, 0.0)
        }


def stat_results(ckpt_folder, skip_scales=[]):
    files = glob.glob(os.path.join(ckpt_folder, "*.xlsx"))
    for f in files:
        fname = os.path.split(f)[-1]
        fname_wo_ext = os.path.splitext(fname)[0]
        scale = float(fname_wo_ext.replace("scale=", ""))

        if scale in skip_scales:
            continue

        df = pd.read_excel(f)
        mask = df["convergence"].to_numpy()
        du_norm = df["final_du_norm"].to_numpy()
        dt_norm = df["final_dt_norm"].to_numpy()
        steps = df["steps"].to_numpy()

        metrics = {
            "SR": mask,
            "TS": steps[mask],
            "|du| (°)": du_norm[mask] / np.pi * 180,
            "|dt| (mm)": dt_norm[mask] * 1000
        }

        if "total_time" in df.columns:
            metrics["mTT (ms)"] = df["total_time"].to_numpy()[mask] * 1000
        if "frontend_time" in df.columns:
            metrics["mFT (ms)"] = df["frontend_time"].to_numpy()[mask] * 1000
        if "backend_time" in df.columns:
            metrics["mBT (ms)"] = df["backend_time"].to_numpy()[mask] * 1000
        
        formatting = {
            "SR": "{:.4f}",
            "TS": "{:.1f}",
            "|du| (°)": "{:.3f}",
            "|dt| (mm)": "{:.3f}",
            "mTT (ms)": "{:.2f}",
            "mFT (ms)": "{:.2f}",
            "mBT (ms)": "{:.2f}",
        }
        
        message = []
        for k in metrics:
            metric = st_metric(metrics[k])
            fmt = formatting[k]
            fmt = "{} = " + "{}±{}, ci95: ({} ~ {})".format(fmt, fmt, fmt, fmt)
            msg = fmt.format(k, 
                metric["mean"], metric["std"], metric["ci95"][0], metric["ci95"][1])
            # msg = "{} = {:.2f} ({:.2f} ~ {:.2f})".format(
            #     k, metric["mean"], metric["ci95"][0], metric["ci95"][1])
            message.append(msg)
        message = "\n\t".join(message)

        print("scale = {:.2f}: \n\t{}".format(scale, message))


def stat_results2(ckpt_folder, skip_scales=[], secs=[]):
    files = glob.glob(os.path.join(ckpt_folder, "*.xlsx"))
    sections, _ = stat_scenes()

    if not secs:
        secs = ["A"]
    
    stat_df = dict()
    method_mapping = {}

    for f in files:
        fname = os.path.split(f)[-1]
        fname_wo_ext = os.path.splitext(fname)[0]
        scale = float(fname_wo_ext.replace("scale=", ""))

        if scale in skip_scales:
            continue

        for sec in secs:
            print("---- section = {} ----".format(sec))
            gidx = sections[sec]

            df = pd.read_excel(f)
            df: pd.DataFrame = df.iloc[gidx].copy()

            # add section header
            if "section" not in stat_df:
                stat_df["section"] = []
            stat_df["section"].append(sec)

            if "scale" not in stat_df:
                stat_df["scale"] = []
            stat_df["scale"].append(scale)

            mask = df["convergence"].to_numpy()
            du_norm = df["final_du_norm"].to_numpy()
            dt_norm = df["final_dt_norm"].to_numpy()
            steps = df["steps"].to_numpy()

            metrics = {
                "SR": mask,
                "TS": steps[mask],
                "|du| (°)": du_norm[mask] / np.pi * 180,
                "|dt| (mm)": dt_norm[mask] * 1000
            }

            if "total_time" in df.columns:
                metrics["mTT (ms)"] = df["total_time"].to_numpy()[mask] * 1000
            if "frontend_time" in df.columns:
                metrics["mFT (ms)"] = df["frontend_time"].to_numpy()[mask] * 1000
            if "backend_time" in df.columns:
                metrics["mBT (ms)"] = df["backend_time"].to_numpy()[mask] * 1000
            
            formatting = {
                "SR": "{:.4f}",
                "TS": "{:.1f}",
                "|du| (°)": "{:.3f}",
                "|dt| (mm)": "{:.3f}",
                "mTT (ms)": "{:.2f}",
                "mFT (ms)": "{:.2f}",
                "mBT (ms)": "{:.2f}",
            }
            
            message = []
            for k in metrics:
                metric = st_metric(metrics[k])
                fmt = formatting[k]
                fmt = "{} = " + "{}±{}, ci95: ({} ~ {})".format(fmt, fmt, fmt, fmt)
                msg = fmt.format(k, 
                    metric["mean"], metric["std"], metric["ci95"][0], metric["ci95"][1])
                # msg = "{} = {:.2f} ({:.2f} ~ {:.2f})".format(
                #     k, metric["mean"], metric["ci95"][0], metric["ci95"][1])
                message.append(msg)

                if k not in stat_df:
                    stat_df[k] = []
                stat_df[k + " mean"].append(metric["mean"])
                stat_df[k + " std"].append(metric["std"])
                stat_df[k + "cib"].append(metric["ci95"][0])
                stat_df[k + "cit"].append(metric["ci95"][1])

            message = "\n\t".join(message)

            print("scale = {:.2f}: \n\t{}".format(scale, message))


if __name__ == "__main__":
    # category = "ablation_on_scale_info"
    # category = "ablation_on_data_generation"
    # category = "ablation_on_network_structure"
    category = "hybrid_pbvs"
    # category = "hybrid_pbvs_wo_ransac"
    # category = "inaccurate_dist"

    # category = "test_different_detector"
    # category = "ablation_on_scale_info"
    # category = "compare_with_ibvs"
    # category = "compare_with_raft_ibvs"

    # category = "compare_with_image_vs_on_star1953"
    # category = "compare_with_image_vs_on_mscoco14_150"

    here = os.path.dirname(__file__)
    ckpt_folders = glob.glob(os.path.join(here, "results", category, "*"))
    ckpt_folders = [f for f in ckpt_folders if os.path.isdir(f)]

    for ckpt_folder in ckpt_folders:
        gather_results(ckpt_folder)
    
    for ckpt_folder in ckpt_folders:
        print("######## ckpt = {}".format(os.path.split(ckpt_folder)[-1]))
        stat_results(ckpt_folder)
        # stat_results2(ckpt_folder, secs=["S", "M", "L", "A"])
