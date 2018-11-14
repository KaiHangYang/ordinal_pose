import numpy as np
import os
import sys
sys.path.append("../../")
from utils.common_utils import my_utils

label_dir = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/labels_syn"

bones_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16]
])

if __name__ == "__main__":
    datas = os.listdir(label_dir)
    datas = [os.path.join(label_dir, i) for i in datas]
    mean_skeleton = my_utils.mAverageCounter(bones_indices.shape[0])
    mean_cammat = my_utils.mAverageCounter(16)

    count = 0
    for cur_data_path in datas:
        sys.stderr.write("\rCurrently process {}".format(count))
        sys.stderr.flush()
        cur_data = np.load(cur_data_path).tolist()
        cur_joints_3d = cur_data["joints_3d"].copy()
        cur_cammat = cur_data["cam_mat"].copy().flatten()

        cur_bone_len = []

        for cur_bone in bones_indices:
            cur_bone_len.append(np.linalg.norm(cur_joints_3d[cur_bone[0]] - cur_joints_3d[cur_bone[1]]))

        mean_skeleton.add(np.array(cur_bone_len))
        mean_cammat.add(cur_cammat)
        count += 1

    np.save("./mean_skeleton.npy", mean_skeleton.cur_average)
    np.save("./mean_cammat", mean_cammat.cur_average)
