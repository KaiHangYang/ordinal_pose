import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import syn_net
from net import pose_net
from utils.preprocess_utils import syn_preprocess
from utils.preprocess_utils import pose_preprocess

from utils.visualize_utils import visualize_tools as vtools
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.postprocess_utils import volume_utils
from utils.defs import pose_defs

##################### Evaluation Configs ######################
import configs

# t means gt(0) or ord(1)
# ver means experiment version
# d means validset(0) or trainset(1)
configs.parse_configs(t=0, d=0)
configs.print_configs()

pretrained_syn_model = 860000
pretrained_pose_model = 440000

###############################################################

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

    wnd_width = 512
    wnd_height = 512

    ################### Initialize the visualization tools ##################
    proj_mat = vtools.OpenGLUtils.perspective(np.radians(45), float(wnd_width) / wnd_height, 0.1, 10000.0)
    view_mat = vtools.OpenGLUtils.lookAt((0, 0, 6), (0, 0, 0), (0, 1, 0))
    visualBox = vtools.mVisualBox(wnd_width, wnd_height, title="all show", btn_callback=m_btn_callback, proj_mat=proj_mat, view_mat=view_mat, limbs_n_root=[pose_defs.h36m_pose, pose_defs.h36m_root])
    #########################################################################
    synnet_batch_size = 1
    posenet_batch_size = 2
    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_raw_images = tf.placeholder(shape=[synnet_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_syn_images = tf.placeholder(shape=[posenet_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)

    syn_model = syn_net.mSynNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=synnet_batch_size, is_training=False, loss_weight_heatmaps=1.0, loss_weight_fb=1.0, loss_weight_br=1.0, is_use_bn=False)
    pose_model = pose_net.mPoseNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=posenet_batch_size, is_training=False, loss_weight_heatmap=5.0, loss_weight_volume=1.0, is_use_bn=False)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_raw_images)
            syn_model.build_evaluation(eval_batch_size=1)

            pose_model.build_model(input_syn_images)
            pose_model.build_evaluation(eval_batch_size=1, flip_array=pose_preprocess.flip_array)

        print("Network built!")

        data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

        sess.run(tf.global_variables_initializer())
        ################# Restore the model ################
        if os.path.exists(configs.syn_restore_model_path_fn(pretrained_syn_model)+".index") and os.path.exists(configs.pose_restore_model_path_fn(pretrained_pose_model)+".index"):
            print("#######################Restored all weights ###########################")
            syn_vars = []
            pose_vars = []
            for variable in tf.trainable_variables():
                if variable.name.split("/")[0] == "SynNet":
                    syn_vars.append(variable)
                elif variable.name.split("/")[0] == "PoseNet":
                    pose_vars.append(variable)

            print("SynNet Trainable Variables: {}".format(len(syn_vars)))
            print("PoseNet Trainable Variables: {}".format(len(pose_vars)))

            syn_model_saver = tf.train.Saver(var_list=syn_vars)
            pose_model_saver = tf.train.Saver(var_list=pose_vars)

            syn_model_saver.restore(sess, configs.syn_restore_model_path_fn(pretrained_syn_model))
            pose_model_saver.restore(sess, configs.pose_restore_model_path_fn(pretrained_pose_model))
        else:
            print(configs.syn_restore_model_path_fn(pretrained_syn_model), configs.pose_restore_model_path_fn(pretrained_pose_model))
            print("The prev model is not existing!")
            quit()
        ####################################################
        print("Network restored")

        batch_raw_images_np = np.zeros([1, configs.img_size, configs.img_size, 3], dtype=np.float32)
        batch_joints_2d_np = np.zeros([1, configs.nJoints, 2], dtype=np.float32)
        batch_joints_zidx_np = np.zeros([1, configs.nJoints], dtype=np.float32)

        batch_syn_images_np = np.zeros([1, configs.img_size, configs.img_size, 3], dtype=np.float32)
        batch_syn_images_flipped_np = np.zeros([1, configs.img_size, configs.img_size, 3], dtype=np.float32)

        while not visualBox.checkStop():
            visualBox.begin()

            label_path_for_show = []
            img_path_for_show = []

            source_txt_arr = []
            center_arr = []
            scale_arr = []
            depth_root_arr = []
            gt_joints_3d_arr = []
            crop_joints_2d_arr = []

            if m_btn_callback.get_next() or m_btn_callback.get_going():

                if m_btn_callback.get_going():
                    data_index.val += 1
                else:
                    if m_btn_callback.get_next() == 1:
                        data_index.val += 1
                    elif m_btn_callback.get_next() == -1:
                        data_index.val -= 1
                m_btn_callback.reset()

                label_path_for_show.append(os.path.basename(lbl_list[data_index.val]))
                img_path_for_show.append(os.path.basename(img_list[data_index.val]))

                cur_img = cv2.imread(img_list[data_index.val])
                cur_label = np.load(lbl_list[data_index.val]).tolist()

                ########## Save the data for evaluation ###########
                source_txt_arr.append(cur_label["source"])
                center_arr.append(cur_label["center"])
                scale_arr.append(cur_label["scale"])
                depth_root_arr.append(cur_label["joints_3d"][0, 2])
                gt_joints_3d_arr.append(cur_label["joints_3d"].copy())
                crop_joints_2d_arr.append(cur_label["joints_2d"].copy())
                ###################################################

                cur_joints_2d = cur_label["joints_2d"].copy()
                cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy() # cause lua is from 1 to n not 0 to n-1
                cur_bone_status = cur_label["bone_status"].copy()
                cur_bone_relations = cur_label["bone_relations"].copy()

                ########## Preprocess for the syn_net ##########
                cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = syn_preprocess.preprocess(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=False, mask=None)
                # generate the heatmaps and volumes
                batch_raw_images_np[0] = cur_img.copy()
                batch_joints_2d_np[0] = cur_joints_2d.copy()
                batch_joints_zidx_np[0] = cur_joints_zidx.copy()

                pd_joints_2d,\
                pd_fb_result,\
                pd_fb_belief,\
                pd_br_result,\
                pd_br_belief = sess.run(
                        [
                         syn_model.pd_joints_2d,
                         syn_model.pd_fb_result,
                         syn_model.pd_fb_belief,
                         syn_model.pd_br_result,
                         syn_model.pd_br_belief],
                        feed_dict={input_raw_images: batch_raw_images_np})

                cur_bone_order = syn_preprocess.bone_order_from_bone_relations(pd_br_result[0], pd_br_belief[0], nBones=configs.nJoints-1)
                cur_synmap, _ = syn_preprocess.draw_syn_img(batch_joints_2d_np[0], pd_fb_result[0], cur_bone_order, size=256, sep_size=64, bone_width=6, joint_ratio=6, bg_color=0.2)
                batch_syn_images_np[0] = cur_synmap / 255.0

                flipped_joints_2d, flipped_bone_status, flipped_bone_order = syn_preprocess.flip_annots(joints_2d=batch_joints_2d_np[0], bone_status=pd_fb_result[0], bone_order=cur_bone_order)
                flipped_synmap, _ = syn_preprocess.draw_syn_img(flipped_joints_2d, flipped_bone_status, flipped_bone_order, size=256, sep_size=64, bone_width=6, joint_ratio=6, bg_color=0.2)
                batch_syn_images_flipped_np[0] = flipped_synmap / 255.0

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
                mean_c_j_2d_pd, mean_c_j_3d_pd, _ = volume_utils.local_to_global(mean_pd_depth[0], depth_root_arr[0], mean_pd_coords_2d[0], source_txt_arr[0], center_arr[0], scale_arr[0])
                raw_c_j_2d_pd, raw_c_j_3d_pd, _ = volume_utils.local_to_global(raw_pd_depth[0], depth_root_arr[0], raw_pd_coords_2d[0], source_txt_arr[0], center_arr[0], scale_arr[0])

                cur_joints_3d_gt = gt_joints_3d_arr[0] - gt_joints_3d_arr[0][0]
                # gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], raw_c_j_3d_pd - raw_c_j_3d_pd[0]
                cur_joints_3d_pd = mean_c_j_3d_pd - mean_c_j_3d_pd[0]

                eval_joints_3d_gt = cur_joints_3d_gt.copy()
                eval_joints_3d_pd = cur_joints_3d_pd.copy()

                cur_joints_3d_gt[:, 1:3] *= -1 # flip the yz
                cur_joints_3d_pd[:, 1:3] *= -1 # flip the yz
                cur_joints_3d_gt /= 450
                cur_joints_3d_pd /= 450

                cur_mpjpe = np.sqrt(np.sum((eval_joints_3d_gt - eval_joints_3d_pd) ** 2, axis=1))
                cur_mean_mpjpe = np.mean(cur_mpjpe)

                print("Mean Error(mm): {}".format(cur_mean_mpjpe))
                print(("Joints Errors(mm):\n" + ("{}: {}\n") * configs.nJoints).format(*np.concatenate([pose_defs.h36m_joint_names[:, np.newaxis], cur_mpjpe[:, np.newaxis]], axis=1).flatten()))

            cv2.imshow("raw_image", batch_raw_images_np[0])
            cv2.imshow("syn_image", batch_syn_images_np[0])
            cv2.waitKey(2)

            visualBox.draw((255 * batch_raw_images_np[0]).astype(np.uint8), [cur_joints_3d_gt, cur_joints_3d_pd], [[0.3, 1.0, 0.3], [1.0, 0.3, 0.3]])
            visualBox.end()
        visualBox.terminate()
