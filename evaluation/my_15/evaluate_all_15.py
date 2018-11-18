import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import fb_net
# from net import syn_net
from net import pose_net

# from utils.preprocess_utils import syn_preprocess
from utils.preprocess_utils import fb_preprocess
from utils.preprocess_utils import pose_preprocess

from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.postprocess_utils import volume_utils

from utils.postprocess_utils.skeleton15 import skeleton_opt

##################### Evaluation Configs ######################
import configs

# t means gt(0) or ord(1)
# ver means experiment version
# d means validset(0) or trainset(1)
configs.parse_configs(t=0, ver=0, d=0, all_type=1)
configs.print_configs()

# in 15 point fb is syn_3
# pose is syn_4

pretrained_fb_model = 680000
pretrained_pose_model = 760000

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

    ################### Some reconfiguration ####################
    h36m_selected_index = configs.H36M_JOINTS_SELECTED
    h36m_bone_selected_index = h36m_selected_index[1:] - 1

    configs.nJoints = 15
    NEW_BONE_INDICES = configs.NEW_BONE_INDICES

    fb_preprocess.flip_array = configs.NEW_FLIP_ARRAY
    pose_preprocess.flip_array = configs.NEW_FLIP_ARRAY
    pose_preprocess.bones_indices = configs.NEW_BONE_INDICES
    pose_preprocess.bone_colors = configs.NEW_BONE_COLORS

    fbnet_batch_size = configs.batch_size
    posenet_batch_size = 2*configs.batch_size
    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_raw_images = tf.placeholder(shape=[fbnet_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_syn_images = tf.placeholder(shape=[posenet_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)

    fb_model = fb_net.mFBNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=fbnet_batch_size, is_training=False, loss_weight_heatmaps=1.0, loss_weight_fb=1.0, is_use_bn=False)
    pose_model = pose_net.mPoseNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=posenet_batch_size, is_training=False, loss_weight_heatmap=5.0, loss_weight_volume=1.0, is_use_bn=False)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            fb_model.build_model(input_raw_images)
            fb_model.build_evaluation(eval_batch_size=configs.batch_size)

            pose_model.build_model(input_syn_images)
            pose_model.build_evaluation(eval_batch_size=configs.batch_size, flip_array=pose_preprocess.flip_array)

        print("Network built!")

        mean_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
        opt_mean_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
        raw_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
        opt_raw_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)

        data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

        sess.run(tf.global_variables_initializer())
        ################# Restore the model ################
        if os.path.exists(configs.pose_restore_model_path_fn(pretrained_pose_model)+".index") and os.path.exists(configs.syn_restore_model_path_fn(pretrained_fb_model)+".index"):
            print("#######################Restored all weights ###########################")
            fb_vars = []
            pose_vars = []
            for variable in tf.trainable_variables():
                if variable.name.split("/")[0] == "PoseNet":
                    pose_vars.append(variable)
                elif variable.name.split("/")[0] == "FBNet":
                    fb_vars.append(variable)

            print("PoseNet Trainable Variables: {}".format(len(pose_vars)))
            print("FBNet Trainable Variables: {}".format(len(fb_vars)))

            pose_model_saver = tf.train.Saver(var_list=pose_vars)
            fb_model_saver = tf.train.Saver(var_list=fb_vars)

            pose_model_saver.restore(sess, configs.pose_restore_model_path_fn(pretrained_pose_model))
            fb_model_saver.restore(sess, configs.syn_restore_model_path_fn(pretrained_fb_model))
        else:
            print(configs.pose_restore_model_path_fn(pretrained_pose_model), configs.syn_restore_model_path_fn(pretrained_fb_model))
            print("The prev model is not existing!")
            quit()
        ####################################################
        print("Network restored")

        while not data_index.isEnd():
            global_steps = sess.run(pose_model.global_steps)

            batch_raw_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_joints_2d_np = np.zeros([configs.batch_size, configs.nJoints, 2], dtype=np.float32)
            batch_joints_zidx_np = np.zeros([configs.batch_size, configs.nJoints], dtype=np.float32)

            batch_bone_relations_np = np.zeros([configs.batch_size, configs.nJoints-1, configs.nJoints-1])
            batch_bone_status_np = np.zeros([configs.batch_size, configs.nJoints-1])

            batch_syn_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_syn_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

            label_path_for_show = []
            img_path_for_show = []

            source_txt_arr = []
            center_arr = []
            scale_arr = []
            depth_root_arr = []
            gt_joints_3d_arr = []
            crop_joints_2d_arr = []
            cam_matrix_arr = []

            for b in range(configs.batch_size):
                label_path_for_show.append(os.path.basename(lbl_list[data_index.val]))
                img_path_for_show.append(os.path.basename(img_list[data_index.val]))

                cur_img = cv2.imread(img_list[data_index.val])
                cur_label = np.load(lbl_list[data_index.val]).tolist()
                data_index.val += 1

                ########## Save the data for evaluation ###########
                source_txt_arr.append(cur_label["source"])
                center_arr.append(cur_label["center"])
                scale_arr.append(cur_label["scale"])
                depth_root_arr.append(cur_label["joints_3d"][0, 2])

                gt_joints_3d_arr.append(cur_label["joints_3d"].copy()[h36m_selected_index])
                crop_joints_2d_arr.append(cur_label["joints_2d"].copy()[h36m_selected_index])
                cam_matrix_arr.append(cur_label["cam_mat"].copy())
                ###################################################

                cur_joints_3d = gt_joints_3d_arr[b].copy()
                cur_joints_2d = crop_joints_2d_arr[b].copy()
                cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy()[h36m_selected_index] # cause lua is from 1 to n not 0 to n-1
                cur_bone_status = recalculate_bone_status(cur_joints_3d[:, 2], bones_indices=NEW_BONE_INDICES)

                ########## Preprocess for the fb_net ##########
                cur_img, cur_joints_2d, cur_bone_status = fb_preprocess.preprocess(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, is_training=False, mask=None)
                # generate the heatmaps and volumes

                batch_raw_images_np[b] = cur_img.copy()
                batch_joints_2d_np[b] = cur_joints_2d.copy()
                batch_joints_zidx_np[b] = cur_joints_zidx.copy()

                batch_bone_status_np[b] = cur_bone_status.copy()

            pd_joints_2d,\
            pd_fb_results = sess.run(
                    [
                     fb_model.pd_joints_2d,
                     fb_model.pd_fb_result,
                     ],
                    feed_dict={input_raw_images: batch_raw_images_np})

            pd_joints_2d = pd_joints_2d * configs.joints_2d_scale
            # pd_joints_2d = batch_joints_2d_np

            ##### TODO Here I test the gt_fb_results
            # pd_fb_results = batch_bone_status_np

            for b in range(configs.batch_size):
                cur_synmap, _, _ = pose_preprocess.preprocess(joints_2d=pd_joints_2d[b], joints_zidx=batch_joints_zidx_np[b], bone_status=pd_fb_results[b], bone_relations=None, is_training=False, bone_width=6, joint_ratio=6, bg_color=0.2, num_of_joints=configs.nJoints)
                batch_syn_images_np[b] = cur_synmap

                flipped_joints_2d, flipped_joints_zidx, flipped_bone_status, _ = pose_preprocess.flip_annots(joints_2d=pd_joints_2d[b], joints_zidx=batch_joints_zidx_np[b], bone_status=pd_fb_results[b], bone_relations=None, size=256)
                flipped_synmap, _, _ = pose_preprocess.preprocess(joints_2d=flipped_joints_2d, joints_zidx=flipped_joints_zidx, bone_status=flipped_bone_status, bone_relations=None, is_training=False, bone_width=6, joint_ratio=6, bg_color=0.2, num_of_joints=configs.nJoints)
                batch_syn_images_flipped_np[b] = flipped_synmap

                # cv2.imshow("raw", batch_syn_images_np[b])
                # cv2.imshow("flipped", batch_syn_images_flipped_np[b])
                # cv2.waitKey()

            mean_vol_joints, \
            raw_vol_joints  = sess.run(
                    [
                     pose_model.mean_joints,
                     pose_model.raw_joints
                    ],
                    feed_dict={input_syn_images: np.concatenate([batch_syn_images_np, batch_syn_images_flipped_np], axis=0)})

            print((len(label_path_for_show) * "{}\n").format(*label_path_for_show))

            mean_vol_joints = mean_vol_joints.astype(np.int32)
            mean_pd_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], mean_vol_joints[:, :, 2].tolist()))
            mean_pd_coords_2d = mean_vol_joints[:, :, 0:2] * configs.joints_2d_scale

            raw_vol_joints = raw_vol_joints.astype(np.int32)
            raw_pd_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], raw_vol_joints[:, :, 2].tolist()))
            raw_pd_coords_2d = raw_vol_joints[:, :, 0:2] * configs.joints_2d_scale

            ############## evaluate the coords recovered from the pd 2d and gt root depth and camera matrix
            for b in range(configs.batch_size):
                mean_c_j_2d_pd, mean_c_j_3d_pd, _ = volume_utils.local_to_global(mean_pd_depth[b], depth_root_arr[b], mean_pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                raw_c_j_2d_pd, raw_c_j_3d_pd, _ = volume_utils.local_to_global(raw_pd_depth[b], depth_root_arr[b], raw_pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])

                #### Use the mean skeleton to evaluate
                opt_mean_c_j_3d_pd = np.reshape(skeleton_opt.opt(volume_utils.recover_2d(mean_pd_coords_2d[b], scale=scale_arr[b], center=center_arr[b]).flatten().tolist(), mean_pd_depth[b].flatten().tolist(), cam_matrix_arr[b].flatten().tolist()), [-1, 3])
                opt_raw_c_j_3d_pd = np.reshape(skeleton_opt.opt(volume_utils.recover_2d(raw_pd_coords_2d[b], scale=scale_arr[b], center=center_arr[b]).flatten().tolist(), raw_pd_depth[b].flatten().tolist(), cam_matrix_arr[b].flatten().tolist()), [-1, 3])

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

        mean_coords_eval.save("../eval_result/all_15/coord_eval_syn{}w_fb{}w_mean.npy".format(pretrained_fb_model/10000, pretrained_pose_model / 10000))
        raw_coords_eval.save("../eval_result/all_15/coord_eval_syn{}w_fb{}w_raw.npy".format(pretrained_fb_model/10000, pretrained_pose_model / 10000))

        opt_mean_coords_eval.save("../eval_result/all_15/coord_eval_syn{}w_fb{}w_mean_opt.npy".format(pretrained_fb_model/10000, pretrained_pose_model / 10000))
        opt_raw_coords_eval.save("../eval_result/all_15/coord_eval_syn{}w_fb{}w_raw_opt.npy".format(pretrained_fb_model/10000, pretrained_pose_model / 10000))