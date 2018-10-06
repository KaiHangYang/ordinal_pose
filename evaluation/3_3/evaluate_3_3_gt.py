import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_3_3
from utils.preprocess_utils import ordinal_3_3 as preprocessor
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.postprocess_utils import volume_utils

##################### Evaluation Configs ######################
import configs

# t means gt(0) or ord(1)
# d means validset(0) or trainset(1)
configs.parse_configs(0, 0)
configs.print_configs()

evaluation_models = [125000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_volumes = tf.placeholder(shape=[configs.batch_size, configs.feature_map_size, configs.feature_map_size, configs.nJoints * configs.feature_map_size], dtype=tf.float32)
    ordinal_model = ordinal_3_3.mOrdinal_3_3(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False)

    with tf.Session() as sess:

        ordinal_model.build_model(input_images)
        ordinal_model.build_loss_gt(input_volumes=input_volumes, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

        print("Network built!")
        # log_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model

        for cur_model_iterations in evaluation_models:

            coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

            if os.path.exists(configs.restore_model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.restore_model_path_fn(cur_model_iterations))
            else:
                print(configs.restore_model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()

            while not data_index.isEnd():
                global_steps = sess.run(ordinal_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_volumes_np = np.zeros([configs.batch_size, configs.feature_map_size, configs.feature_map_size, configs.nJoints*configs.feature_map_size], dtype=np.float32)

                img_path_for_show = []
                label_path_for_show = []

                source_txt_arr = []
                center_arr = []
                scale_arr = []
                depth_root_arr = []
                gt_joints_3d_arr = []
                crop_joints_2d_arr = []


                preprocess_time = time.clock()

                for b in range(configs.batch_size):
                    img_path_for_show.append(os.path.basename(img_list[data_index.val]))
                    label_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_img = cv2.imread(img_list[data_index.val])
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    ########## Save the data for evaluation ###########
                    source_txt_arr.append(cur_label["source"])
                    center_arr.append(cur_label["center"])
                    scale_arr.append(cur_label["scale"])
                    depth_root_arr.append(cur_label["joints_3d"][0, 2])
                    gt_joints_3d_arr.append(cur_label["joints_3d"].copy())
                    crop_joints_2d_arr.append(cur_label["joints_2d"].copy())
                    ###################################################

                    cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy() # cause lua is from 1 to n not 0 to n-1
                    cur_joints = np.concatenate([cur_label["joints_2d"], cur_joints_zidx[:, np.newaxis]], axis=1)

                    # Cause the dataset is to large, test no augment first
                    # cur_img, cur_joints, is_do_flip = preprocessor.preprocess(cur_img, cur_joints)
                    batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])

                    hm_joint_2d = cur_joints[:, 0:2] / configs.coords_2d_scale
                    hm_joint_3d = np.concatenate([hm_joint_2d, cur_joints[:, 2][:, np.newaxis]], axis=1)

                    # for j_idx in range(configs.nJoints):
                        # batch_volumes_np[b][:, :, configs.feature_map_size*j_idx:configs.feature_map_size*(j_idx+1)] = preprocessor.make_gaussian_3d(hm_joint_3d[j_idx], size=configs.feature_map_size, ratio=2)

                preprocess_time = time.clock() - preprocess_time

                forward_time = time.clock()
                gt_vol_joints, \
                pd_vol_joints = sess.run(
                        [
                         ordinal_model.gt_joints,
                         ordinal_model.pd_joints
                        ],
                        feed_dict={input_images: batch_images_np, input_volumes: batch_volumes_np})

                forward_time = time.clock() - forward_time

                # print("Iteration: {:07d} \nVolume Joints accuracy: {:07f}\n\n".format(global_steps, acc))
                # print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
                print("Forward time: {:07f}. Preprocess time: {:07f}".format(forward_time, preprocess_time))

                # pd_depth = (pd_vol_joints[:, :, 2] - pd_vol_joints[:, 0, 2]) * configs.depth_scale
                # pd_coords_2d = pd_vol_joints[:, :, 0:2] * configs.coords_2d_scale

                # gt_depth = (gt_vol_joints[:, :, 2] - gt_vol_joints[:, 0, 2]) * configs.depth_scale
                # gt_coords_2d = gt_vol_joints[:, :, 0:2] * configs.coords_2d_scale

                # ############# evaluate the coords recovered from the gt 2d and gt root depth

                # eval_time = time.clock()
                # for b in range(configs.batch_size):
                    # c_j_2d_pd, c_j_3d_pd, _ = volume_utils.local_to_global(pd_depth[b], depth_root_arr[b], pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    # c_j_2d_gt, c_j_3d_gt, _ = volume_utils.local_to_global(gt_depth[b], depth_root_arr[b], gt_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    # coords_eval.add(c_j_3d_gt, c_j_3d_pd)

                # eval_time = time.clock() - eval_time

                # print("eval_time: {:07f}".format(eval_time))
                # coords_eval.printMean()
                # print("\n\n")

            coords_eval.save("../eval_result/gt_3_3/coord_eval_{}w.npy".format(cur_model_iterations / 10000))
