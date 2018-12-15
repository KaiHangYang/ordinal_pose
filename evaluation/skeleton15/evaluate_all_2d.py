import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import dlcm_net
from net import pose_net

from utils.preprocess_utils import dlcm_preprocess
from utils.preprocess_utils import pose_preprocess

from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton

##################### Evaluation Configs ######################
configs = mConfigs("../eval.conf", "combined")
dlcm_configs = mConfigs("../eval.conf", "dlcm_net")
pose_configs = mConfigs("../eval.conf", "pose_net")

################### Resetting ####################
configs.loss_weights = [10.0, 1.0, 1.0]
configs.loss_weight_heatmap = 1
configs.loss_weight_pose = 100

configs.pose_2d_scale = 4.0
configs.pose_3d_scale = 1000.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)

dlcm_configs.is_use_bn = True
dlcm_configs.nModules = 1
dlcm_configs.nFeats = 256

pose_configs.is_use_bn = False
pose_configs.nFeats = 256

configs.batch_size = configs.valid_batch_size
### train or valid
configs.range_file =  configs.h36m_valid_range_file
configs.img_path_fn = configs.h36m_valid_img_path_fn
configs.lbl_path_fn = configs.h36m_valid_lbl_path_fn
#####################################################

dlcm_preprocessor = dlcm_preprocess.DLCMProcessor(skeleton=skeleton, img_size=configs.img_size, hm_size=configs.hm_size, sigma=1.0)
pose_preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, with_br=False, with_fb=False, bone_width=6, joint_ratio=6, bg_color=0.2)

evaluation_models = [(480000, 900000)]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[2*configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_syn_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)

    dlcm_model = dlcm_net.mDLCMNet(skeleton=skeleton, img_size=configs.img_size, batch_size=2*configs.batch_size, is_training=False, loss_weights=configs.loss_weights, pose_2d_scale=configs.pose_2d_scale, is_use_bn=dlcm_configs.is_use_bn, nFeats=dlcm_configs.nFeats, nModules=dlcm_configs.nModules)
    pose_model = pose_net.mPoseNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, pose_scale=configs.pose_3d_scale, is_use_bn=pose_configs.is_use_bn)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            dlcm_model.build_model(input_images)
            dlcm_model.build_evaluation(skeleton.flip_array)
            pose_model.build_model(input_syn_images)
            pose_model.build_evaluation()

        print("Network built!")

        net_init = tf.global_variables_initializer()
        sess.run([net_init])

        for cur_model_iterations in evaluation_models:
            pose3d_evaluator = evaluators.mEvaluatorPose3D(nJoints=skeleton.n_joints)

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)
            ################# Restore the model ################

            if os.path.exists(dlcm_configs.model_path_fn(cur_model_iterations[0])+".index") and os.path.exists(pose_configs.model_path_fn(cur_model_iterations[1])+".index"):
                print("#######################Restored all weights ###########################")
                dlcm_vars = []
                pose_vars = []
                for variable in tf.trainable_variables():
                    if variable.name.split("/")[0] == dlcm_model.model_name:
                        dlcm_vars.append(variable)
                    elif variable.name.split("/")[0] == pose_model.model_name:
                        pose_vars.append(variable)

                print("{} Trainable variables: {}".format(dlcm_model.model_name, len(dlcm_vars)))
                print("{} Trainable variables: {}".format(pose_model.model_name, len(pose_vars)))

                dlcm_saver = tf.train.Saver(var_list=dlcm_vars)
                pose_saver = tf.train.Saver(var_list=pose_vars)

                dlcm_saver.restore(sess, dlcm_configs.model_path_fn(cur_model_iterations[0]))
                pose_saver.restore(sess, pose_configs.model_path_fn(cur_model_iterations[1]))

            else:
                print(dlcm_configs.model_path_fn(cur_model_iterations[0]), pose_configs.model_path_fn(cur_model_iterations[1]))
                print("The Trained models are not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():
                global_steps = sess.run(pose_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

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

                    batch_images_np[b] = cur_img.copy() / 255.0
                    batch_images_flipped_np[b] = cv2.flip(cur_img, 1) / 255.0

                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, lbl_path_for_show)))

                pd_2d = sess.run([
                    dlcm_model.raw_pd_2d
                    ],
                   feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                pd_2d = pd_2d[0]

                # The pd_2d is already in 256x256
                ############## evaluate the coords recovered from the gt 2d and gt root depth
                for b in range(configs.batch_size):
                    cur_img, _, _ = pose_preprocessor.preprocess(joints_2d=pd_2d[b], joints_3d=np.zeros([skeleton.n_joints, 3]), is_training=False, scale=None, center=None, cam_mat=None)
                    batch_syn_images_np[b] = cur_img

                pd_3d = sess.run(
                        pose_model.pd_3d,
                        feed_dict={input_syn_images: batch_syn_images_np}
                        )

                # Here the pd_3d is root related
                pose3d_evaluator.add(batch_joints_3d_np - batch_joints_3d_np[:, 0][:, np.newaxis], pd_3d)

                sys.stdout.write("Pose Error: ")
                pose3d_evaluator.printMean()

            pose3d_evaluator.save("../eval_result/{}/mpje_syn{}w_pose{}w.npy".format(configs.prefix, cur_model_iterations[0] / 10000, cur_model_iterations[1] / 10000))
