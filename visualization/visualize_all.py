import numpy as np
import sys
import cv2
import time

sys.path.append("../")
from utils.visualize_utils import visualize_tools as vtools
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.defs.skeleton import mSkeleton15 as skeleton

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

################### Define the path storing the results ####################
raw_img_path_fn = lambda x: "/home/kaihang/Projects/pose_project/new_pose/evaluation/eval_result/overall/results/raw-{}.jpg".format(x)
gt_syn_img_path_fn = lambda x: "/home/kaihang/Projects/pose_project/new_pose/evaluation/eval_result/overall/results/gt-{}.jpg".format(x)
pd_syn_img_path_fn = lambda x: "/home/kaihang/Projects/pose_project/new_pose/evaluation/eval_result/overall/results/pd-{}.jpg".format(x)
lbl_path_fn = lambda x: "/home/kaihang/Projects/pose_project/new_pose/evaluation/eval_result/overall/results/{}.npy".format(x)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        initial_val = 0
    else:
        initial_val = int(sys.argv[1])

    wnd_width = 512
    wnd_height = 512

    visualBox = vtools.mVisualBox(wnd_width=wnd_width, wnd_height=wnd_height, title="overall", btn_callback=m_btn_callback, limbs_n_root=[skeleton.bone_indices, 0])
    data_index = my_utils.mRangeVariable(min_val=0, max_val=109867, initial_val=initial_val)

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

            cur_raw_img = cv2.imread(raw_img_path_fn(cur_index))
            cur_syn_gt_img = cv2.imread(gt_syn_img_path_fn(cur_index))
            cur_syn_pd_img = cv2.imread(pd_syn_img_path_fn(cur_index))
            cur_lbl = np.load(lbl_path_fn(cur_index)).tolist()

            cur_img_display = np.concatenate([cur_raw_img, cur_syn_gt_img, cur_syn_pd_img], axis=1)

            cur_joints_3d_gt = cur_lbl["gt_3d"] * 0.002
            cur_joints_3d_pd = cur_lbl["pd_3d"] * 0.002

            cur_joints_3d_gt[:, 1:3] *= -1
            cur_joints_3d_pd[:, 1:3] *= -1

            error_per = np.sqrt(np.sum((cur_lbl["gt_3d"]-cur_lbl["pd_3d"]) ** 2, axis=-1))
            log = ("Frame:{:08d} MPJE(mm): {:0.6f} \n"+"{:<15}: {:0.6f}\n" * skeleton.n_joints).format(cur_index, error_per.mean(), *reduce(lambda x,y: list(x) + list(y), zip(skeleton.joint_names, error_per)))

            print(log)

        cv2.imshow("raw_gt_pd", cur_img_display)
        cv2.waitKey(3)

        visualBox.draw((128 * np.ones([wnd_height, wnd_width, 3])).astype(np.uint8), [cur_joints_3d_gt, cur_joints_3d_pd], [[0.3, 1.0, 0.3], [1.0, 0.3, 0.3]])
        visualBox.end()
    visualBox.terminate()
