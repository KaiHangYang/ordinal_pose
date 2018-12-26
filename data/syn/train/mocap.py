import numpy as np
import os
import sys

sys.path.append("../../../")

from utils.defs.skeleton import mSkeleton15 as skeleton

rerank_index = np.array([14, 11, 12, 13, 8, 9, 10, 1, 0, 5, 6, 7, 2, 3, 4])

prefix = "mpi"

source_label_dir = "/home/kaihang/DataSet_2/mocap/" + prefix
target_label_dir = "/home/kaihang/DataSet_2/Ordinal/syn/train"

if __name__ == "__main__":

    angles_array = []

    for cur_subdir in os.listdir(source_label_dir):
        cur_subdir = os.path.join(source_label_dir, cur_subdir)

        for cur_pose_file in os.listdir(cur_subdir):
            sys.stdout.write("\rCurrent process {}".format(len(angles_array)))
            sys.stdout.flush()

            cur_pose_file = os.path.join(cur_subdir, cur_pose_file)

            with open(cur_pose_file, "r") as f:
                pose_data = f.readlines()
                pose_data = np.reshape([float(i) for i in pose_data[-1].strip().split(" ")], [skeleton.n_joints, 3])

                pose_data = pose_data[rerank_index]

            angles = skeleton.get_angles(pose_data)
            angles_array.append(angles)

    np.save(os.path.join(target_label_dir, prefix+"_angles.npy"), angles_array)
