import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import math

sys.path.append("../../")
from net import pose_net
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import pose_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorPose3D

##################### Setting for training ######################
configs = mConfigs("../eval.conf", "overall")
pose_configs = mConfigs("../eval.conf", "pose_net_aug_global")

################ Reseting  #################
configs.loss_weight_heatmap = 1
configs.loss_weight_pose = 100
configs.pose_2d_scale = 4.0
configs.pose_3d_scale = 1000.0

configs.is_use_bn = False

configs.batch_size = 1

configs.n_epoches = 50
configs.learning_rate = 2.5e-5
configs.gamma = 0.1
configs.schedule = [5, 25]
configs.valid_steps = 0 # every training epoch valid the network

configs.nFeats = 256
configs.nModules = 3
configs.nRegModules = 2

configs.valid_type = "valid"

configs.extra_log_dir = "../eval_result/" + configs.prefix

configs.h36m_valid_range_file = os.path.join(configs.range_file_dir, "valid_range.npy")

################### Initialize the data reader ####################

configs.printConfig()
preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, with_br=True, with_fb=True, bone_width=6, joint_ratio=6, overlap_threshold=6, bone_status_threshold=80, bg_color=0.2, pad_scale=0.4, pure_color=True)

restore_model_epoch = 21
#################################################################

if __name__ == "__main__":
    #################################Set the train and valid datas##################################

    if configs.valid_type == "train":
        configs.lbl_path_fn = configs.h36m_train_lbl_path_fn
        configs.img_path_fn = configs.h36m_train_img_path_fn
        valid_range = np.load(configs.h36m_train_range_file)
        configs.syn_data_path_fn = lambda x: "../eval_result/syn_net_mixed-11000/train_datas/{}.npy".format(x)
        configs.dlcm_data_path_fn = lambda x: "../eval_result/dlcm_mixed-15000/train_datas/{}.npy".format(x)
    elif configs.valid_type == "valid":
        configs.lbl_path_fn = configs.h36m_valid_lbl_path_fn
        configs.img_path_fn = configs.h36m_valid_img_path_fn
        valid_range = np.load(configs.h36m_valid_range_file)
        configs.syn_data_path_fn = lambda x: "../eval_result/syn_net_mixed-11000/valid_datas/{}.npy".format(x)
        configs.dlcm_data_path_fn = lambda x: "../eval_result/dlcm_mixed-15000/valid_datas/{}.npy".format(x)

    print("Valid DataSet number: {}".format(valid_range.shape[0]))

    input_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    pose_model = pose_net.mPoseNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_pose=configs.loss_weight_pose, pose_2d_scale=configs.pose_2d_scale, pose_3d_scale=configs.pose_3d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nRegModules=configs.nRegModules, nModules=configs.nModules)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            pose_model.build_model(input_images, global_average_pooling=True)
            pose_model.build_evaluation()

        model_saver = tf.train.Saver(max_to_keep=configs.n_epoches)
        net_init = tf.global_variables_initializer()
        sess.run([net_init])

        # reload the model
        if os.path.exists(pose_configs.model_path_fn(restore_model_epoch)+".index"):
            print("#######################Restored all weights ###########################")
            model_saver.restore(sess, pose_configs.model_path_fn(restore_model_epoch))
        else:
            print("The prev model is not existing!")
            quit()

        cur_valid_global_steps = 0
        ############################ Next Evaluate #############################
        is_epoch_finished = False

        valid_pose3d_evaluator = mEvaluatorPose3D(nJoints=skeleton.n_joints)

        for idx, cur_data_idx in enumerate(valid_range):

            batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_joints_2d_np = np.zeros([configs.batch_size, skeleton.n_joints, 2], dtype=np.float32)
            batch_joints_3d_np = np.zeros([configs.batch_size, skeleton.n_joints, 3], dtype=np.float32)

            raw_img = cv2.imread(configs.img_path_fn(cur_data_idx))
            cur_label = np.load(configs.lbl_path_fn(cur_data_idx)).tolist()

            cur_dlcm_data = np.load(configs.dlcm_data_path_fn(idx)).tolist()
            cur_syn_data = np.load(configs.syn_data_path_fn(idx)).tolist()

            cur_joints_3d = cur_label["joints_3d"][skeleton.h36m_selected_index]

            cur_joints_2d = cur_dlcm_data["raw_pd_2d"]
            cur_fb = cur_syn_data["pd_fb"]
            cur_br = cur_syn_data["pd_br"]
            cur_br_belief = cur_syn_data["pd_br_belief"]

            # cur_joints_2d = cur_dlcm_data["gt_2d"]
            # cur_fb = cur_syn_data["gt_fb"]
            # cur_br = cur_syn_data["gt_br"]
            # cur_br_belief = np.ones_like(cur_br)

            cur_bone_order = preprocessor.bone_order_from_bone_relations(cur_br, cur_br_belief)
            cur_img = preprocessor.draw_syn_img(cur_joints_2d, cur_fb, cur_bone_order)

            ################## Draw the ground truth bone maps ######################
            gt_joints_2d = cur_dlcm_data["gt_2d"]
            gt_fb = cur_syn_data["gt_fb"]
            gt_br = cur_syn_data["gt_br"]
            gt_br_belief = np.ones_like(gt_br)

            gt_bone_order = preprocessor.bone_order_from_bone_relations(gt_br, gt_br_belief)
            gt_img = preprocessor.draw_syn_img(gt_joints_2d, gt_fb, gt_bone_order)
            #########################################################################
            # cv2.imshow("raw_img", raw_img)
            # cv2.imshow("syn_img", cur_img)
            # cv2.waitKey(3)

            batch_images_np[0] = cur_img / 255.0
            batch_joints_3d_np[0] = cur_joints_3d - cur_joints_3d[0]

            pd_poses = sess.run(
                    [pose_model.pd_poses],
                    feed_dict={
                               input_images: batch_images_np
                              })

            pd_poses = pd_poses[0]

            np.save(os.path.join(configs.extra_log_dir, "results", "{}.npy".format(idx)), {"gt_3d": batch_joints_3d_np[0], "pd_3d": pd_poses[0]})
            cv2.imwrite(os.path.join(configs.extra_log_dir, "results", "gt-{}.jpg".format(idx)), gt_img)
            cv2.imwrite(os.path.join(configs.extra_log_dir, "results", "pd-{}.jpg".format(idx)), cur_img)
            cv2.imwrite(os.path.join(configs.extra_log_dir, "results", "raw-{}.jpg".format(idx)), raw_img)

            valid_pose3d_evaluator.add(gt_coords=batch_joints_3d_np, pd_coords=pd_poses)
            valid_pose3d_evaluator.printMean()
            cur_valid_global_steps += 1

        valid_pose3d_evaluator.save(os.path.join(configs.extra_log_dir, "valid"), prefix="valid", epoch=0)
