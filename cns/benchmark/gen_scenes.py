import os
import cv2
import glob
import json
import numpy as np
import pybullet as p
from ..sim.environment import ImageEnvGUI


class SimEnv(ImageEnvGUI):
    def __init__(self, camera_config, resample=True, auto_reinit=True):
        super().__init__(camera_config, resample, auto_reinit)

    def init(self):
        if self.resample or (self.points is None):
            num_objs = np.random.randint(8, 12)
            self.gen_scenes(num_objs, self.obs_r, self.obs_h)

            while True:
                self.target_wcT = self.sample_target_pose()
                self.initial_wcT = self.sample_initial_pose()
                if not self.close_enough(self.target_wcT, self.initial_wcT):
                    break
            
            target_cwT = np.linalg.inv(self.target_wcT)
            p.setGravity(0, 0, -9.8, physicsClientId=self.client)
            for _ in range(200):
                p.stepSimulation(physicsClientId=self.client)
            frame = self.camera.render(target_cwT, self.client)

            rgb = frame.color_image()
            self.tar_img = np.ascontiguousarray(rgb[:, :, ::-1])  # rgb to bgr

            self.points = np.zeros((1, 3))  # (N, 3)
            self.target_xy = np.zeros((1, 2))
            self.target_Z = np.ones(1)  # (N,)

        self.steps = 0
        self.few_match_occur = False
        self.current_wcT = self.initial_wcT.copy()

        if self.debug_items_displayed:
            self.axes_cam_tar.update(self.target_wcT)
            self.axes_cam_cur.update(self.current_wcT)
            self.pc.update(self.points)

        return self.tar_img.copy()

    def observation(self):
        current_cwT = np.linalg.inv(self.current_wcT)
        frame = self.camera.render(current_cwT, self.client)

        rgb = frame.color_image()
        cur_img = np.ascontiguousarray(rgb[:, :, ::-1])  # rgb to bgr

        if not self.debug_items_displayed:
            self.init_debug_items()

        return cur_img
    
    def export_scenes(self):
        objects = []
        for obj_id, obj in zip(self.obj_ids, self.objs):
            pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
            obj_info = obj._asdict()
            obj_info["pos"] = pos
            obj_info["orn"] = orn

            sub_path: str = obj_info["path"]
            idx = sub_path.find("ycb_objects")
            if idx >= 0:
                sub_path = sub_path[idx:]
                sub_path = sub_path.split(os.sep, 1)[-1].replace("\\", "/")
            obj_info["path"] = sub_path
            objects.append(obj_info)

        return {
            "target_wcT": self.target_wcT.tolist(),
            "initial_wcT": self.initial_wcT.tolist(),
            "camera_config": self.camera.intrinsic.to_dict(),
            "camera_nf": [self.camera.near, self.camera.far], 
            "objects": objects
        }


def save_scenes_to_json(scenes):
    here = os.path.dirname(__file__)
    folder = os.path.join(here, "scenes")
    if not os.path.exists(folder):
        os.makedirs(folder)

    saved_files = glob.glob(os.path.join(folder, "*.json"))
    if len(saved_files) == 0:
        number = 0
    else:
        saved_files.sort()
        last_fname = os.path.split(saved_files[-1])[-1]
        last_number = int(last_fname.replace(".json", ""))
        number = last_number + 1
    
    save_path = os.path.join(folder, "{:0>3d}.json".format(number))
    with open(save_path, "w") as fp:
        json.dump(scenes, fp, indent=4)
    return save_path


if __name__ == "__main__":
    import time
    
    t = time.time()
    seed = int(("{:.6f}".format(t - int(t)))[2:])
    np.random.seed(seed)

    p.connect(p.GUI_SERVER)
    env = SimEnv(None)
    tar_image = env.init()

    scenes = []
    while True:
        cur_image = env.observation()
        tc_image = np.concatenate([tar_image, cur_image], axis=1)
        cv2.imshow("target | current", tc_image)
        cv2.waitKey(1)

        print("---------------------------------------")
        print("[INFO] Scene No. {}".format(len(scenes)))
        k = input("[INFO] Input a to add, s to save: ")
        if k.startswith('a'):
            scene = env.export_scenes()
            scenes.append(scene)
        elif k.startswith('s'):
            save_scenes_to_json(scenes)
            break
        tar_image = env.init()


