import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import syn_net
from net import pose_net

from utils.preprocess_utils import syn_preprocess
from utils.preprocess_utils import pose_preprocess

from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton

##################### Evaluation Configs ######################
configs = mConfigs("../eval.conf", "combined")
syn_configs = mConfigs("../eval.conf", "syn_net_mixed")
pose_configs = mConfigs("../eval.conf", "pose_net_br")

syn_preprocessor = syn_preprocess.SynProcessor(skeleton=skeleton, img_size=configs.img_size, bone_width=6, joint_ratio=6, bg_color=0.2)
pose_preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, with_br=True, bone_width=6, joint_ratio=6, bg_color=0.2)

evaluation_models = [(720000, 900000)]
###############################################################

if __name__ == "__main__":

    ################### Resetting ####################
    configs.loss_weight_heatmap = 1
    configs.loss_weight_pose = 100
    configs.pose_2d_scale = 4.0
    configs.pose_3d_scale = 1000.0
    configs.is_use_bn = False

    configs.batch_size = configs.valid_batch_size

    ### train or valid
    configs.range_file =  configs.h36m_valid_range_file
    configs.img_path_fn = configs.h36m_valid_img_path_fn
    configs.lbl_path_fn = configs.h36m_valid_lbl_path_fn
    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)
    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_syn_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)

    syn_model = syn_net.mSynNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn)
    pose_model = pose_net.mPoseNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, pose_scale=configs.pose_3d_scale, is_use_bn=configs.is_use_bn)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)
            syn_model.build_evaluation()
            pose_model.build_model(input_images)
            pose_model.build_evaluation()

        print("Network built!")

        net_init = tf.global_variables_initializer()
        sess.run([net_init])

        for cur_model_iterations in evaluation_models:
            pose3d_evaluator = evaluators.mEvaluatorPose3D(nJoints=skeleton.n_joints)

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)
            ################# Restore the model ################

            if os.path.exists(syn_configs.model_path_fn(cur_model_iterations[0])+".index") and os.path.exists(pose_configs.model_path_fn(cur_model_iterations[1])+".index"):
                print("#######################Restored all weights ###########################")
                syn_vars = []
                pose_vars = []
                for variable in tf.trainable_variables():
                    if variable.name.split("/")[0] == syn_model.model_name:
                        syn_vars.append(variable)
                    elif variable.name.split("/")[0] == pose_model.model_name:
                        pose_vars.append(variable)

                print("{} Trainable variables: {}".format(syn_model.model_name, len(syn_vars)))
                print("{} Trainable variables: {}".format(pose_model.model_name, len(pose_vars)))

                syn_saver = tf.train.Saver(var_list=syn_vars)
                pose_saver = tf.train.Saver(var_list=pose_vars)

                syn_saver.restore(sess, syn_configs.model_path_fn(cur_model_iterations[0]))
                pose_saver.restore(sess, pose_configs.model_path_fn(cur_model_iterations[1]))

            else:
                print(syn_configs.model_path_fn(cur_model_iterations[0]), pose_configs.model_path_fn(cur_model_iterations[1]))
                print("The Trained models are not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():
                global_steps = sess.run(pose_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_syn_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

                batch_joints_3d_np = np.zeros([configs.batch_size, skeleton.n_joints, 3], dtype=np.float32)

                img_path_for_show = []
                lbl_path_for_show = []

                for b in range(configs.batch_size):
                    img_path_for_show.append(os.path.basename(img_list[data_index.val]))
                    lbl_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_img = cv2.imread(img_list[data_index.val])
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    cur_joints_2d = cur_label["joints_2d"][skeleton.h36m_selected_index].copy()
                    cur_joints_3d = cur_label["joints_3d"][skeleton.h36m_selected_index].copy()
                    cur_scale = cur_label["scale"]
                    cur_center = cur_label["center"]
                    cur_cam_mat = cur_label["cam_mat"]

                    cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = syn_preprocessor.preprocess_h36m(img=cur_img, joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, scale=cur_scale, center=cur_center, cam_mat=cur_cam_mat, is_training=False)

                    batch_images_np[b] = cur_img.copy()

                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                    # cv2.imshow("raw_img", batch_images_np[b])
                    # cv2.imshow("flipped_img", batch_images_flipped_np[b])
                    # cv2.waitKey()

                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, lbl_path_for_show)))

                pd_fb, \
                pd_fb_belief, \
                pd_br, \
                pd_br_belief, \
                pd_2d = sess.run([
                    syn_model.pd_fb_result,
                    syn_model.pd_fb_belief,
                    syn_model.pd_br_result,
                    syn_model.pd_br_belief,
                    syn_model.pd_2d
                    ],
                   feed_dict={input_images: batch_images_np})

                # The pd_2d is already in 256x256

                ############## evaluate the coords recovered from the gt 2d and gt root depth
                for b in range(configs.batch_size):
                    cur_bone_order = syn_preprocessor.bone_order_from_bone_relations(pd_br[b], pd_br_belief[b])
                    cur_syn_image = syn_preprocessor.draw_syn_img(pd_2d[b], pd_fb[b], cur_bone_order)
                    cur_syn_image = cur_syn_image / 255.0
                    batch_syn_images_np[b] = cur_syn_image.copy()

                raw_img_display = np.concatenate(batch_images_np, axis=0)
                syn_img_display = np.concatenate(batch_syn_images_np, axis=0)

                all_img_display = np.concatenate([raw_img_display, syn_img_display], axis=1)

                # cv2.imshow("all_img_display", all_img_display)
                # cv2.waitKey(3)

                pd_3d = sess.run(
                        pose_model.pd_3d,
                        feed_dict={input_images: batch_syn_images_np}
                        )

                # Here the pd_3d is root related
                pose3d_evaluator.add(batch_joints_3d_np - batch_joints_3d_np[:, 0][:, np.newaxis], pd_3d)

                sys.stdout.write("Pose Error: ")
                pose3d_evaluator.printMean()
                print("\n\n")

            pose3d_evaluator.save("../eval_result/{}/mpje_{}w.npy".format(configs.prefix, cur_model_iterations / 10000))
