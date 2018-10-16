import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_F
from utils.preprocess_utils import ordinal_3_3 as preprocessor
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.postprocess_utils import volume_utils

##################### Evaluation Configs ######################
import configs

# t means gt(0) or ord(1)
# d means validset(0) or trainset(1)
configs.parse_configs(t=1, ver=1, d=0)
configs.print_configs()

evaluation_models = [260000, 280000, 300000, 320000, 340000, 360000, 380000, 400000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    #### Used for valid
    network_batch_size = 2*configs.batch_size

    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    #### Used for pre-calculate the depth scale
    scale_range_arr = np.load(configs.scale_range_file)
    scale_data_from = 0
    scale_data_to = len(scale_range_arr)

    scale_img_list = [configs.scale_img_path_fn(i) for i in scale_range_arr]
    scale_lbl_list = [configs.scale_lbl_path_fn(i) for i in scale_range_arr]
    ##################################################################

    input_images = tf.placeholder(shape=[network_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")

    ordinal_model = ordinal_F.mOrdinal_F(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=network_batch_size, is_training=False)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
            ordinal_model.build_evaluation_nogt(configs.batch_size, preprocessor.flip_array)

        print("Network built!")
        # log_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model

        for cur_model_iterations in evaluation_models:

            if os.path.exists(configs.restore_model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.restore_model_path_fn(cur_model_iterations))
            else:
                print(configs.restore_model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()

            #################### First get the depth scale from the subset of the training set ######################
            cur_model_depth_scale = my_utils.mAverageCounter(shape=[1])
            scale_data_index = my_utils.mRangeVariable(min_val=scale_data_from, max_val=scale_data_to-1, initial_val=scale_data_from)
            while not scale_data_index.isEnd():
                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                ##### The depth is all related to the root
                gt_depth_arr = []

                for b in range(configs.batch_size):

                    cur_img = cv2.imread(scale_img_list[scale_data_index.val])
                    cur_label = np.load(scale_lbl_list[scale_data_index.val]).tolist()
                    scale_data_index.val += 1

                    ########## Save the data for evaluation ###########
                    gt_joints_3d = cur_label["joints_3d"].copy()
                    gt_depth_arr.append(gt_joints_3d[:, 2] - gt_joints_3d[0, 2])
                    ###################################################
                    cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

                    cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, is_training=False)
                    batch_images_np[b] = cur_img
                    batch_images_flipped_np[b] = preprocessor.flip_img(batch_images_np[b])

                mean_scale_depth,\
                raw_scale_depth = sess.run(
                        [
                         ordinal_model.mean_volumes_z,
                         ordinal_model.raw_volumes_z
                        ],
                        feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                scale_for_show = []

                scale_depth = mean_scale_depth

                for b in range(configs.batch_size):
                    scale_depth_related_to_root = scale_depth[b] - scale_depth[b][0]
                    cur_scale = (np.max(gt_depth_arr[b]) - np.min(gt_depth_arr[b])) / (np.max(scale_depth_related_to_root) - np.min(scale_depth_related_to_root) + 1e-7)
                    cur_model_depth_scale.add(cur_scale)

                    scale_for_show.append(cur_scale)

                print("Iter: {:07d} Scales: {}\n\n".format(scale_data_index.val, scale_for_show))
                print("Cur Scale: {:07f}\n\n".format(cur_model_depth_scale.cur_average[0]))
            ################################################################################################

            ##### Then evaluate it #####
            cur_depth_scale = cur_model_depth_scale.cur_average[0]
            print("Scale used to evaluate: {:07f}".format(cur_depth_scale))

            mean_depth_eval = evaluators.mEvaluatorDepth(nJoints=configs.nJoints)
            mean_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)

            raw_depth_eval = evaluators.mEvaluatorDepth(nJoints=configs.nJoints)
            raw_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

            while not data_index.isEnd():
                global_steps = sess.run(ordinal_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

                img_path_for_show = []
                label_path_for_show = []

                source_txt_arr = []
                center_arr = []
                scale_arr = []
                depth_root_arr = []
                gt_joints_3d_arr = []
                gt_depth_arr = []

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
                    gt_depth_arr.append(cur_label["joints_3d"].copy()[:, 2] - cur_label["joints_3d"][0, 2])
                    ###################################################

                    cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

                    cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, is_training=False)

                    batch_images_np[b] = cur_img
                    batch_images_flipped_np[b] = preprocessor.flip_img(batch_images_np[b])

                mean_depth,\
                mean_joints_2d,\
                raw_depth,\
                raw_joints_2d = sess.run(
                        [
                         ordinal_model.mean_volumes_z,
                         ordinal_model.mean_joints_2d,
                         ordinal_model.raw_volumes_z,
                         ordinal_model.raw_joints_2d],
                        feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))

                ##### the mean results
                mean_depth = cur_depth_scale * (mean_depth - mean_depth[:, 0][:, np.newaxis])
                mean_joints_2d = mean_joints_2d * configs.coords_2d_scale

                mean_depth_eval.add(np.array(gt_depth_arr), mean_depth)
                sys.stdout.write("Mean: ")
                mean_depth_eval.printMean()

                ##### The raw results
                raw_depth = cur_depth_scale * (raw_depth - raw_depth[:, 0][:, np.newaxis])
                raw_joints_2d = raw_joints_2d * configs.coords_2d_scale

                raw_depth_eval.add(np.array(gt_depth_arr), raw_depth)
                sys.stdout.write("raw: ")
                raw_depth_eval.printMean()

                ############# evaluate the coords recovered from the pd 2d and gt root depth
                for b in range(configs.batch_size):
                    mean_c_j_2d, mean_c_j_3d, _ = volume_utils.local_to_global(mean_depth[b], depth_root_arr[b], mean_joints_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    mean_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], mean_c_j_3d - mean_c_j_3d[0])

                    raw_c_j_2d, raw_c_j_3d, _ = volume_utils.local_to_global(raw_depth[b], depth_root_arr[b], raw_joints_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    raw_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], raw_c_j_3d - raw_c_j_3d[0])

                sys.stdout.write("Mean: ")
                mean_coords_eval.printMean()
                sys.stdout.write("Raw: ")
                raw_coords_eval.printMean()
                print("\n\n")

            mean_depth_eval.save("../eval_result/ord_f_1/depth_eval_{}w_mean.npy".format(cur_model_iterations / 10000))
            mean_coords_eval.save("../eval_result/ord_f_1/coord_eval_{}w_mean.npy".format(cur_model_iterations / 10000))

            raw_depth_eval.save("../eval_result/ord_f_1/depth_eval_{}w_raw.npy".format(cur_model_iterations / 10000))
            raw_coords_eval.save("../eval_result/ord_f_1/coord_eval_{}w_raw.npy".format(cur_model_iterations / 10000))
