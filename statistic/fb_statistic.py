import numpy as np
import os
import sys

sys.path.append("../")

if __name__ == "__main__":
    result_dir = "/home/kaihang/Projects/pose_project/new_pose/evaluation/eval_result/syn_net_mixed-11000/valid_datas"
    datas = [os.path.join(result_dir, i) for i in os.listdir(result_dir)]
    gt_arr = []
    pd_arr = []

    for cur_data_path in datas:
        cur_data = np.load(cur_data_path).tolist()

        gt_fb = cur_data["gt_fb"]
        pd_fb = cur_data["pd_fb"]

        select_index = np.logical_not(gt_fb == pd_fb)
        gt_arr = gt_arr + gt_fb[select_index].tolist()
        pd_arr = pd_arr + pd_fb[select_index].tolist()

    np.save("fb_result.npy", {"gt_arr": gt_arr, "pd_arr": pd_arr})
