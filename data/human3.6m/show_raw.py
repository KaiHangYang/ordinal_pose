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

preprocessor = pose_preprocess.PoseProcessor(skeleton, 256, with_fb=True, with_br=True, overlap_threshold=6, pure_color=False)

############## some Parameters
data_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/"

images_file_fn = lambda x: os.path.join(os.path.join(data_path, "images"), "{}.jpg".format(x))
annots_file_fn = lambda x: os.path.join(os.path.join(data_path, "labels_syn"), "{}.npy".format(x))

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
    total_sum = len(os.listdir(os.path.join(data_path, "images")))
    data_index = my_utils.mRangeVariable(0, total_sum, -1)
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
            if m_btn_callback.get_going():
                data_index.val += 1
            else:
                if m_btn_callback.get_next() == 1:
                    data_index.val += 1
                elif m_btn_callback.get_next() == -1:
                    data_index.val -= 1

            m_btn_callback.reset()

            cur_index = data_index.val
            # cur_index = 1228
            # print(cur_index)

            # cropped_img = cv2.imread(images_file_fn(cur_index))
            cropped_img = cv2.resize(cv2.imread(images_file_fn(cur_index)), (256, 256))
            cur_label = np.load(annots_file_fn(cur_index)).tolist()

            joints_3d = cur_label["joints_3d"][skeleton.h36m_selected_index]
            joints_2d = cur_label["joints_2d"][skeleton.h36m_selected_index]
            cam_mat = cur_label["cam_mat"][0:3, 0:3]

            cur_depth = joints_3d[:, 2] - joints_3d[0, 2]
            root_depth = joints_3d[0, 2]

            joints_2d, joints_3d, proj_mat = volume_utils.local_to_global(cur_depth, root_depth, joints_2d, cur_label["source"], cur_label["center"], cur_label["scale"])
            visualBox.setProjMat(proj_mat)

            cur_img = volume_utils.put_cropped_back(cropped_img, cur_label["center"], cur_label["scale"])

            cur_img = display_utils.drawLines(cur_img, joints_2d, indices=skeleton.bone_indices)
            cur_img = display_utils.drawPoints(cur_img, joints_2d)

            ############## Test to handler the joints_3d ##############

            test_joints_3d = joints_3d.copy()
            bone_lengths = skeleton.get_bonelengths(test_joints_3d)

            root_joint = test_joints_3d[0].copy()
            test_joints_3d = test_joints_3d - root_joint

            GET_ANGLE_TIME = time.clock()
            angles = skeleton.get_angles(test_joints_3d)
            # angles = np.zeros([skeleton.n_joints, 3])
            GET_ANGLE_TIME = time.clock() - GET_ANGLE_TIME

            # angles = skeleton.jitter_angles(angles, jitter_size = math.pi / 20)
            # bone_lengths = skeleton.jitter_bonelengths(bone_lengths, jitter_size=30)

            RECOVER_JOINTS_TIME = time.clock()
            test_joints_3d = skeleton.get_joints(angles, bone_lengths) + root_joint
            RECOVER_JOINTS_TIME = time.clock() - RECOVER_JOINTS_TIME

            sys.stdout.write("\rget angle time {:0.4f}s, recover time {:0.4f}s".format(GET_ANGLE_TIME, RECOVER_JOINTS_TIME))
            sys.stdout.flush()

            syn_img, _, _ = preprocessor.preprocess(angles, bone_lengths, root_joint, cam_mat, is_training=False)

            ##########################################################

        cv2.imshow("img_2d", cropped_img)
        cv2.imshow("syn_img", syn_img)
        cv2.waitKey(4)

        visualBox.draw(cur_img, [joints_3d, test_joints_3d], [[1, 1, 1], [0, 1, 1]])
        # visualBox.draw(cur_img, [joints_3d], [[1, 1, 1]])
        visualBox.end()
    visualBox.terminate()
