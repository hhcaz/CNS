import cv2
import json
import numpy as np
import pyrealsense2 as rs
from ..utils.perception import CameraIntrinsic


class Stream(object):
    def __init__(self):
        self.config = rs.config()
        self.pipeline = rs.pipeline()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("[INFO] RealSense product line: {}".format(device_product_line))

        if device_product_line == "D400":
            self.color_wh = (640, 480)
            self.depth_wh = (640, 480)
        elif device_product_line == "L500":
            self.color_wh = (1280, 720)
            self.depth_wh = (1024, 768)

        self.config.enable_stream(
            rs.stream.depth, self.depth_wh[0], self.depth_wh[1], rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, self.color_wh[0], self.color_wh[1], rs.format.bgr8, 30)

        # Align depth image to color image
        self.alignedFs = rs.align(rs.stream.color)

        # Start streaming
        self.profile = self.pipeline.start(self.config)
        color_sensor = self.profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, 500)
        # time.sleep(2)
        print("[INFO] Start streaming")

    def get(self):
        """
        Returns:
        - color_image: np.ndarray, shape=(H, W, 3), BGR format
        - depth_image: np.ndarray, shape=(H, W, 1), unit: m
        """
        fs = self.pipeline.wait_for_frames()
        frames = self.alignedFs.process(fs)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            return None, None

        color_image = np.asarray(color_frame.get_data())
        depth_image = np.asarray(depth_frame.get_data()) / 1000  # mm -> m

        return color_image, depth_image

    def get_color_intr(self):
        fs = self.pipeline.wait_for_frames()
        frames = self.alignedFs.process(fs)
        color_frame = frames.get_color_frame()
        color_profile = color_frame.get_profile()
        cvsprofile = rs.video_stream_profile(color_profile)
        c_intr = cvsprofile.get_intrinsics()
        color_intr = np.array(
            [[c_intr.fx, 0, c_intr.ppx],
             [0, c_intr.fy, c_intr.ppy],
             [0, 0, 1]]
        )
        return color_intr


if __name__ == "__main__":
    stream = Stream()

    while True:
        bgr, depth = stream.get()
        cv2.imshow("default", bgr)
        print(bgr.shape, depth.shape, depth.dtype, depth.mean())
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("./image.jpg", bgr)
            print("[INFO] saving done")
        elif key == ord('i'):
            intrinsic = stream.get_color_intr()
            cam_intr = CameraIntrinsic(
                width=bgr.shape[1],
                height=bgr.shape[0],
                fx=intrinsic[0, 0],
                fy=intrinsic[1, 1],
                cx=intrinsic[0, 2],
                cy=intrinsic[1, 2]
            )
            with open("camera_intrinsic.json", "w") as fp:
                json.dump({"intrinsic": cam_intr.to_dict()}, fp, indent=4)
            print("[INFO] intrinsic is: \n{}".format(intrinsic))
        elif key == ord('q'):
            break
