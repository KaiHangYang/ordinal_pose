import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# lbl_path_fn = lambda x: "/home/kaihang/Projects/pose_project/new_pose/evaluation/eval_result/overall/results/{}.npy".format(x)

if __name__ == "__main__":
    # total_datas = 109867

    # total_errors = []

    # for i in range(0, total_datas):
        # cur_lbl = np.load(lbl_path_fn(i)).tolist()
        # gt_3d = cur_lbl["gt_3d"]
        # pd_3d = cur_lbl["pd_3d"]

        # error = np.sqrt(((gt_3d - pd_3d) ** 2).sum(axis=-1)).mean()
        # total_errors.append(error)

    # np.save("errors.npy", total_errors)

    errors = np.load("errors.npy")

    plt.hist(x=errors, bins=np.arange(0, 500, 1))
    plt.title("Error distribution")
    plt.show()
