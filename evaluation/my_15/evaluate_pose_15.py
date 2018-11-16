import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import pose_net
from utils.preprocess_utils import pose_preprocess as preprocessor
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators

from utils.postprocess_utils import volume_utils
from utils.postprocess_utils import skeleton_opt

##################### Evaluation Configs ######################
import configs

# t means gt(0) or ord(1)
# ver means experiment version
# d means validset(0) or trainset(1)
configs.parse_configs(t=0, ver=4, d=0)
configs.print_configs()

# evaluation_models = [440000, 480000, 500000, 540000]
evaluation_models = [680000]
###############################################################
def recalculate_bone_status(joints_z, bones_indices):
    bone_status = []
    for cur_bone_idx in bones_indices:
        if np.abs(joints_z[cur_bone_idx[0]] - joints_z[cur_bone_idx[1]]) < 100:
            bone_status.append(0)
        elif joints_z[cur_bone_idx[1]] < joints_z[cur_bone_idx[0]]:
            bone_status.append(1)
        else:
            bone_status.append(2)
    return np.array(bone_status)

if __name__ == "__main__":

    ################ Reconfigure #################
    network_batch_size = 2*configs.batch_size
    # configs.nJoints = 15 # setted in the config.py
    CUR_JOINTS_SELECTED = configs.H36M_JOINTS_SELECTED
    preprocessor.bones_indices = configs.NEW_BONE_INDICES
    preprocessor.bone_colors = configs.NEW_BONE_COLORS
    preprocessor.flip_array = configs.NEW_FLIP_ARRAY

    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[network_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)

    pose_model = pose_net.mPoseNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=network_batch_size, is_training=False, loss_weight_heatmap=5.0, loss_weight_volume=1.0, is_use_bn=False)
    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            pose_model.build_model(input_images)
            pose_model.build_evaluation(eval_batch_size=configs.batch_size, flip_array=preprocessor.flip_array)

        print("Network built!")
        # log_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model

        for cur_model_iterations in evaluation_models:
            mean_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
            opt_mean_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
            raw_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
            opt_raw_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

            ################# Restore the model ################
            if os.path.exists(configs.restore_model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.restore_model_path_fn(cur_model_iterations))
            else:
                print(configs.restore_model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():
                global_steps = sess.run(pose_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

                label_path_for_show = []

                source_txt_arr = []
                center_arr = []
                scale_arr = []
                depth_root_arr = []
                gt_joints_3d_arr = []
                crop_joints_2d_arr = []
                cam_matrix_arr = []

                for b in range(configs.batch_size):
                    label_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    ########## Save the data for evaluation ###########
                    source_txt_arr.append(cur_label["source"])
                    center_arr.append(cur_label["center"])
                    scale_arr.append(cur_label["scale"])
                    depth_root_arr.append(cur_label["joints_3d"][0, 2].copy()[CUR_JOINTS_SELECTED])
                    gt_joints_3d_arr.append(cur_label["joints_3d"].copy()[CUR_JOINTS_SELECTED])
                    crop_joints_2d_arr.append(cur_label["joints_2d"].copy()[CUR_JOINTS_SELECTED])
                    cam_matrix_arr.append(cur_label["cam_mat"].copy())
                    ###################################################

                    cur_joints_3d = cur_label["joints_3d"].copy()[CUR_JOINTS_SELECTED]
                    cur_joints_2d = cur_label["joints_2d"].copy()[CUR_JOINTS_SELECTED]
                    cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy()[CUR_JOINTS_SELECTED] # cause lua is from 1 to n not 0 to n-1
                    # cur_bone_status = cur_label["bone_status"].copy()
                    # cur_bone_relations = cur_label["bone_relations"].copy()
                    cur_bone_status = recalculate_bone_status(cur_joints_3d[:, 2], preprocessor.bones_indices)
                    cur_bone_relations = None

                    ######### Generate the raw image ##########
                    cur_img, _, _ = preprocessor.preprocess(joints_2d=cur_joints_2d, joints_zidx=cur_joints_zidx, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=False, bone_width=6, joint_ratio=6, bg_color=0.2, num_of_joints=configs.nJoints)
                    # generate the heatmaps and volumes
                    batch_images_np[b] = cur_img.copy()

                    ######### Then Generate the flipped one
                    cur_joints_2d_flipped, cur_joints_zidx_flipped, cur_bone_status_flipped, cur_bone_relations_flipped = preprocessor.flip_annots(joints_2d=cur_joints_2d, joints_zidx=cur_joints_zidx, bone_status=cur_bone_status, bone_relations=cur_bone_relations, size=256)
                    cur_img_flipped, _, _ = preprocessor.preprocess(joints_2d=cur_joints_2d_flipped, joints_zidx=cur_joints_zidx_flipped, bone_status=cur_bone_status_flipped, bone_relations=cur_bone_relations_flipped, is_training=False, bone_width=6, joint_ratio=6, bg_color=0.2)
                    batch_images_flipped_np[b] = cur_img_flipped.copy()

                    # cv2.imshow("raw_img", batch_images_np[b])
                    # cv2.imshow("flipped_img", batch_images_flipped_np[b])
                    # cv2.waitKey()

                mean_vol_joints, \
                raw_vol_joints  = sess.run(
                        [
                         pose_model.mean_joints,
                         pose_model.raw_joints
                        ],
                        feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                print((len(label_path_for_show) * "{}\n").format(*label_path_for_show))

                mean_vol_joints = mean_vol_joints.astype(np.int32)
                mean_pd_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], mean_vol_joints[:, :, 2].tolist()))
                mean_pd_coords_2d = mean_vol_joints[:, :, 0:2] * configs.joints_2d_scale

                raw_vol_joints = raw_vol_joints.astype(np.int32)
                raw_pd_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], raw_vol_joints[:, :, 2].tolist()))
                raw_pd_coords_2d = raw_vol_joints[:, :, 0:2] * configs.joints_2d_scale

                # ############# evaluate the coords recovered from the gt 2d and gt root depth
                for b in range(configs.batch_size):
                    mean_c_j_2d_pd, mean_c_j_3d_pd, _ = volume_utils.local_to_global(mean_pd_depth[b], depth_root_arr[b], mean_pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    raw_c_j_2d_pd, raw_c_j_3d_pd, _ = volume_utils.local_to_global(raw_pd_depth[b], depth_root_arr[b], raw_pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])

                    #### Use the mean skeleton to evaluate
                    opt_mean_c_j_3d_pd = np.reshape(skeleton_opt.opt(volume_utils.recover_2d(mean_pd_coords_2d[b], scale=).flatten().tolist(), mean_pd_depth[b].flatten().tolist(), cam_matrix_arr[b].flatten().tolist()), [-1, 3])
                    opt_raw_c_j_3d_pd = np.reshape(skeleton_opt.opt(volume_utils.recover_2d(raw_pd_coords_2d[b]).flatten().tolist(), raw_pd_depth[b].flatten().tolist(), cam_matrix_arr[b].flatten().tolist()), [-1, 3])

                    # Here I used the root aligned pose to evaluate the error
                    # according to https://github.com/geopavlakos/c2f-vol-demo/blob/master/matlab/utils/errorH36M.m
                    mean_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], mean_c_j_3d_pd - mean_c_j_3d_pd[0])
                    raw_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], raw_c_j_3d_pd - raw_c_j_3d_pd[0])

                    opt_mean_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], opt_mean_c_j_3d_pd - opt_mean_c_j_3d_pd[0])
                    opt_raw_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], opt_raw_c_j_3d_pd - opt_raw_c_j_3d_pd[0])

                sys.stdout.write("Mean: ")
                mean_coords_eval.printMean()

                sys.stdout.write("Raw: ")
                raw_coords_eval.printMean()

                sys.stdout.write("Opt Mean: ")
                opt_mean_coords_eval.printMean()

                sys.stdout.write("Opt Raw: ")
                opt_raw_coords_eval.printMean()

                print("\n\n")

            mean_coords_eval.save("../eval_result/pose_15/coord_eval_{}w_mean.npy".format(cur_model_iterations / 10000))
            raw_coords_eval.save("../eval_result/pose_15/coord_eval_{}w_raw.npy".format(cur_model_iterations / 10000))

            opt_mean_coords_eval.save("../eval_result/pose_15/coord_eval_{}w_mean_opt.npy".format(cur_model_iterations / 10000))
            opt_raw_coords_eval.save("../eval_result/pose_15/coord_eval_{}w_raw_opt.npy".format(cur_model_iterations / 10000))
