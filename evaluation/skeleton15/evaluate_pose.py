import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import pose_net
from utils.preprocess_utils import pose_preprocess
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton

##################### Evaluation Configs ######################
configs = mConfigs("../eval.conf", "pose_net")
configs.printConfig()
preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, bone_width=6, joint_ratio=6, bg_color=0.2)

evaluation_models = [760000]
###############################################################

if __name__ == "__main__":

    ################### Resetting ####################
    configs.loss_weight_heatmap = 1
    configs.loss_weight_pose = 100
    configs.pose_2d_scale = 4.0
    configs.pose_3d_scale = 1000.0
    configs.is_use_bn = False

    configs.batch_size = configs.valid_batch_size

    # train or valid
    configs.range_file = configs.h36m_valid_range_file
    configs.lbl_path_fn = configs.h36m_valid_lbl_path_fn
    ################### Initialize the data reader ###################

    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    pose_model = pose_net.mPoseNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_pose=configs.loss_weight_pose, pose_scale=configs.pose_3d_scale, is_use_bn=configs.is_use_bn)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            pose_model.build_model(input_images)
            pose_model.build_evaluation()

        print("Network built!")

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])

        for cur_model_iterations in evaluation_models:
            pose3d_evaluator = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)
            ################# Restore the model ################

            if os.path.exists(configs.model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(cur_model_iterations))
            else:
                print(configs.model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():
                global_steps = sess.run(pose_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_joints_3d_np = np.zeros([configs.batch_size, skeleton.n_joints, 3], dtype=np.float32)

                label_path_for_show = []

                for b in range(configs.batch_size):
                    label_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    cur_joints_3d = cur_label["joints_3d"][skeleton.h36m_selected_index].copy()
                    cur_joints_3d = cur_joints_3d - cur_joints_3d[0]

                    cur_joints_2d = cur_label["joints_2d"][skeleton.h36m_selected_index].copy()

                    cur_bone_relations = None

                    cur_img, cur_joints_2d, cur_joints_3d = preprocessor.preprocess(joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, bone_relations=cur_bone_relations, is_training=False)
                    batch_images_np[b] = cur_img.copy()
                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                    # cv2.imshow("raw_img", batch_images_np[b])
                    # cv2.imshow("flipped_img", batch_images_flipped_np[b])
                    # cv2.waitKey()

                pd_3d = sess.run(
                        [
                         pose_model.pd_3d,
                        ],
                        feed_dict={input_images: batch_images_np})

                print((len(label_path_for_show) * "{}\n").format(*label_path_for_show))

                # ############# evaluate the coords recovered from the gt 2d and gt root depth
                for b in range(configs.batch_size):
                    pose3d_evaluator.add(batch_joints_3d_np[b], pd_3d[b])

                sys.stdout.write("Pose Error: ")
                pose3d_evaluator.printMean()
                print("\n\n")

            pose3d_evaluator.save("../eval_result/{}/mpje_{}w.npy".format(configs.prefix, cur_model_iterations / 10000))
