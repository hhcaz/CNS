import os
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def stat_scenes(scenes=None):
    if scenes is None:
        print("[INFO] Loading scenes...")
        here = os.path.dirname(__file__)
        scene_files = glob.glob(os.path.join(here, "scenes", "*.json"))

        scenes = []
        for f in scene_files:
            with open(f, "r") as fp:
                scenes.extend(json.load(fp))
    
    print("[INFO] Total {} scenes.".format(len(scenes)))

    du_norms = []
    dt_norms = []
    dists = []
    for scene in scenes:
        target_wcT = np.array(scene["target_wcT"])
        initial_wcT = np.array(scene["initial_wcT"])

        dT = np.linalg.inv(target_wcT) @ initial_wcT
        du = R.from_matrix(dT[:3, :3]).as_rotvec()
        dt = dT[:3, 3]

        du_norms.append(np.linalg.norm(du))
        dt_norms.append(np.linalg.norm(dt))
        dists.append(np.linalg.norm(target_wcT[:3, 3]))
    
    du_norms = np.array(du_norms) / np.pi * 180  # degree
    dt_norms = np.array(dt_norms) * 100  # cm
    dists = np.array(dists)  # m

    sort_index = np.argsort(du_norms)
    div3 = len(sort_index) // 3
    sections = {
        "A": sort_index,
        "S": sort_index[:div3],
        "M": sort_index[div3:2*div3],
        "L": sort_index[2*div3:],
    }

    print("-------------------------------------------------------------------------")
    for section in ["A", "S", "M", "L"]:
        idx = sections[section]
        dus = du_norms[idx]
        dts = dt_norms[idx]
        tPo = dists[idx]

        print("[INFO] Statistics of section {}:".format(section))
        print("[INFO] |du|: min = {:.2f}° , max = {:.2f}° , mean = {:.2f}° , std = {:.2f}°"
            .format(dus.min(), dus.max(), dus.mean(), dus.std()))
        print("[INFO] |dt|: min = {:.2f}cm, max = {:.2f}cm, mean = {:.2f}cm, std = {:.2f}cm"
            .format(dts.min(), dts.max(), dts.mean(), dts.std()))
        print("[INFO] tPo: min = {:.3f}m, max = {:.3f}m, mean = {:.3f}m, std = {:.3f}m"
            .format(tPo.min(), tPo.max(), tPo.mean(), tPo.std()))
        print("-------------------------------------------------------------------------")

    return sections, scenes


if __name__ == "__main__":
    stat_scenes()

