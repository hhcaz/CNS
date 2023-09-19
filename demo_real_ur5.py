import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from cns.real.environment import RealEnv
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.benchmark import PickleRecord
from cns.benchmark.stop_policy import PixelStopPolicy
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt

import matplotlib
matplotlib.use("TkAgg")


def get_stop_hint_image():
    H, W = 100, 400
    image = np.zeros((H, W, 3), dtype=np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Press b to break this eposide"

    # get boundary of this text
    text_size = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = (W - text_size[0]) / 2
    textY = (H + text_size[1]) / 2

    # add text centered on image
    cv2.putText(image, text, (textX, textY ), font, 1, (255, 255, 255), 2)
    return image


def detect_external_break(image=None):
    if image is None:
        image = get_stop_hint_image()
    cv2.imshow("Keyboard Breaker", image)
    key = cv2.waitKey(1)
    need_break = key & 0xFF == ord('b')

    if need_break:
        print("[INFO] Detect external break!")

    return need_break


manual_mode = True
total_rounds = 50

env = RealEnv(resample=True, auto_reinit=False)
target_poses_seq, initial_poses_seq = env.generate_poses(total_rounds*2, seed=42)

with open("pipeline.json", "r") as fp:
    config = json.load(fp)
    intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
pipeline = CorrespondenceBasedPipeline(
    "",
    detector="SIFT",
    intrinsic=intrinsic,
    vis=VisOpt.KP
)
result_folder = os.path.join(
    "experiment_results",
    "gvs_scene60",
    "3objs"
)
pkl_record = PickleRecord(result_folder)
stopper = PixelStopPolicy(waiting_time=2, conduct_thresh=0.005)

stop_hint_image = get_stop_hint_image()

env.go_home()
current_round = 0
while True:
    current_round += 1
    if current_round > total_rounds:
        print("[INFO] All test rounds complete, exit")
        break

    print("[INFO] ----------------------------------")
    print("[INFO] Start test round {}/{}".format(current_round, total_rounds))

    # move to another place
    env.recover_end_joint_pose()
    # tar_bcT = env.sample_target_pose()
    tar_bcT = target_poses_seq[current_round-1]
    env.move_to(tar_bcT)
    time.sleep(1.0)
    tar_img, dist_scale = env.observation()
    tar_img = np.ascontiguousarray(tar_img[:, :, [2, 1, 0]])
    pipeline.set_target(tar_img, dist_scale)

    if manual_mode:
        print("[INFO] Image at target: ")
        plt.figure()
        plt.imshow(tar_img)
        plt.show()

    # move to another place
    env.recover_end_joint_pose()
    # ini_bcT = env.sample_initial_pose()
    ini_bcT = initial_poses_seq[current_round-1]
    env.move_to(ini_bcT)

    if manual_mode:
        input("[INFO] Press Enter to continue: ")

    actual_conduct_vel = np.zeros(6)
    stopper.reset()
    start_time = time.time()
    while True:
        cur_bcT = env.get_current_base_cam_T()
        cur_img, _ = env.observation()
        cur_img = np.ascontiguousarray(cur_img[:, :, [2, 1, 0]])

        vel, data, timing = pipeline.get_control_rate(cur_img)
        need_stop = (
            stopper(data, time.time()) or 
            not env.is_safe_pose() or
            data is None or
            (time.time() - start_time > 40) or
            detect_external_break(stop_hint_image)
        )

        pkl_record.append_traj(cur_bcT, vel, timing, stopper.cur_err, data)
        if need_stop:
            env.stop_action()
            # calculate servo error and print
            dT = np.linalg.inv(cur_bcT) @ tar_bcT
            du = R.from_matrix(dT[:3, :3]).as_rotvec()
            dt = dT[:3, 3]
            print("[INFO] Servo error: |du| = {:.3f} degree, |dt| = {:.3f} mm"
                  .format(np.linalg.norm(du)/np.pi*180, np.linalg.norm(dt)*1000))
            
            pkl_record.finalize(
                "round{:0>3d}.pkl".format(current_round),
                tar_bcT, ini_bcT
            )
            
            if manual_mode:
                input("[INFO] Press Enter to start round {}: ".format(current_round + 1))
            break

        print("[INFO] dist_scale at target = {:.3f}, pred_vel = {}"
              .format(dist_scale, np.round(vel, 2)))
        actual_conduct_vel = env.action(vel)

env.go_home()

