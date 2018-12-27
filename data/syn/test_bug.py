import os
import sys
import numpy as np
import cv2
sys.path.append("../../")
import time
import math

from utils.visualize_utils import display_utils
from utils.visualize_utils import visualize_tools as vtools
from utils.common_utils import my_utils
from utils.common_utils import h36m_camera
from utils.common_utils import math_utils
from utils.defs import pose_defs
from utils.postprocess_utils import volume_utils
from utils.preprocess_utils import get_bone_relations
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.preprocess_utils import pose_preprocess

preprocessor = pose_preprocess.PoseProcessor(skeleton, 256, with_fb=True, with_br=True, bone_width=6, joint_ratio=6, overlap_threshold=6, pad_scale=0.4)

############## function to handle the keyboard event
class m_btn_callback(object):
    next_flag = 1
    keep_going = 0

    @classmethod
    def call(cls, keys):
        if keys == vtools.glfw.KEY_J:
            # next
            cls.next_flag = 1
        elif keys == vtools.glfw.KEY_K:
            # prev
            cls.next_flag = -1
        elif keys == vtools.glfw.KEY_SPACE:
            cls.keep_going = not cls.keep_going

    @classmethod
    def reset(cls):
        cls.next_flag = 0

    @classmethod
    def get_next(cls):
        return cls.next_flag

    @classmethod
    def get_going(cls):
        return cls.keep_going


if __name__ == "__main__":
    wnd_width = 1000
    wnd_height = 1000

    # proj_mat = vtools.OpenGLUtils.perspective(np.radians(45), float(wnd_width) / wnd_height, 0.1, 10000.0)
    # view_mat = vtools.OpenGLUtils.lookAt((0, 0, 6), (0, 0, 0), (0, 1, 0))
    view_mat = np.identity(4)
    proj_mat, _ = h36m_camera.get_cam_mat(1, 54138969)

    visualBox = vtools.mVisualBox(wnd_width, wnd_height, title="ordinal show", btn_callback=m_btn_callback, proj_mat=proj_mat, view_mat=view_mat, limbs_n_root=[skeleton.bone_indices, 0], model_size=30.0)

    while not visualBox.checkStop():
        visualBox.begin()

        if m_btn_callback.get_next() or m_btn_callback.get_going():
            m_btn_callback.reset()

            data = np.load("bug_pose.npy").tolist()

            joints_3d = np.reshape(data["joints_3d"], [15, 3])
            joints_2d = np.reshape(data["joints_2d"], [15, 2])
            cur_center = data["center"]
            cur_scale = data["scale"]
            cur_cammat = np.eye(4)

            cur_cammat[0, 0], cur_cammat[1, 1], cur_cammat[0, 2], cur_cammat[1, 2] = data["cam_vec"]

            img = 0.2 * np.ones([256, 256, 3])

            proj_mat = math_utils.cammat2projmat(cur_cammat, wnd_width, wnd_height)
            visualBox.setProjMat(proj_mat)

            # bone_relations = preprocessor.get_bone_relations(joints_2d, joints_3d, cur_scale, cur_center, cur_cammat)
            # print(bone_relations)
            ##########################################################

        cv2.imshow("syn_img", img.copy())
        cv2.imshow("skeleton_syn_img", display_utils.drawLines(img.copy(), joints_2d, indices=skeleton.bone_indices, color_table=skeleton.bone_colors))
        cv2.waitKey(2)

        visualBox.draw(None, [joints_3d], [[1, 1, 1]])
        visualBox.end()
    visualBox.terminate()
