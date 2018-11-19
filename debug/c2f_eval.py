import h5py
import numpy as np
import os
import sys

sys.path.append("../")

from utils.postprocess_utils.skeleton17 import skeleton_opt
from utils.postprocess_utils import volume_utils
from utils.evaluate_utils import evaluators

c2f_result_fn = lambda x: "/home/kaihang/c2f-vol-demo/exp/h36m/valid_{}.h5".format(x + 1)
my_label_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/labels_syn/{}.npy".format(x)

if __name__ == "__main__":

    gtr_coords_eval = evaluators.mEvaluatorPose3D(nJoints=17)
    coords_eval = evaluators.mEvaluatorPose3D(nJoints=17)
    for i in range(109867):
        c2f_data = h5py.File(c2f_result_fn(i), "r")
        lbl_data = np.load(my_label_fn(i)).tolist()

        gt_scale = lbl_data["scale"]
        gt_center = lbl_data["center"]
        gt_cam_mat = lbl_data["cam_mat"]
        gt_joints_3d = lbl_data["joints_3d"]
        gt_source = lbl_data["source"]

        vol_joints = c2f_data["preds3D"][0].copy()

        vol_joints = vol_joints.astype(np.int32)
        c2f_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], vol_joints[:, 2].tolist()))
        c2f_coords_2d = vol_joints[:, 0:2] * 4.0 * (200 / 256.0) + (256 - 200) / 2.0

        _, gtr_c2f_joints_3d, _ = volume_utils.local_to_global(c2f_depth, gt_joints_3d[0, 2], c2f_coords_2d, gt_source, gt_center, gt_scale)
        c2f_joints_3d = np.reshape(skeleton_opt.opt(volume_utils.recover_2d(c2f_coords_2d, scale=gt_scale, center=gt_center).flatten().tolist(), c2f_depth.flatten().tolist(), gt_cam_mat.flatten().tolist()), [-1, 3])

        coords_eval.add(gt_joints_3d - gt_joints_3d[0], c2f_joints_3d - c2f_joints_3d[0])
        gtr_coords_eval.add(gt_joints_3d - gt_joints_3d[0], gtr_c2f_joints_3d - gtr_c2f_joints_3d[0])

        sys.stdout.write("Current Process: {}\n opt:\n".format(i))
        coords_eval.printMean()
        sys.stdout.write("gtr:\n")
        gtr_coords_eval.printMean()

    coords_eval.save("/home/kaihang/Desktop/test_dir/test_eval.npy")
    gtr_coords_eval.save("/home/kaihang/Desktop/test_dir/gtr_test_eval.npy")
