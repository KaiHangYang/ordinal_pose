import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
# ver means experiment version
# d means validset(0) or trainset(1)
configs.parse_configs(t=0, ver=1, d=0)
configs.print_configs()

evaluation_models = [275000, 300000]
###############################################################

if __name__ == "__main__":

    network_batch_size = 2*configs.batch_size
    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[network_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)

    ordinal_model = ordinal_F.mOrdinal_F(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=network_batch_size, is_training=False, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_volume=configs.loss_weight_volume)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
            ordinal_model.build_evaluation(eval_batch_size=configs.batch_size, flip_array=preprocessor.flip_array)


        print("Network built!")
        # log_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model

        for cur_model_iterations in evaluation_models:

            mean_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
            raw_coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)

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
                crop_joints_2d_arr = []

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

                    cur_img, _ = preprocessor.preprocess(cur_img, None, is_training=False)

                    batch_images_np[b] = cur_img
                    batch_images_flipped_np[b] = preprocessor.flip_img(batch_images_np[b])

                mean_vol_joints, \
                raw_vol_joints  = sess.run(
                        [
                         ordinal_model.mean_joints,
                         ordinal_model.raw_joints
                        ],
                        feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))

                mean_vol_joints = mean_vol_joints.astype(np.int32)
                mean_pd_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], mean_vol_joints[:, :, 2].tolist()))
                mean_pd_coords_2d = mean_vol_joints[:, :, 0:2] * configs.coords_2d_scale

                raw_vol_joints = raw_vol_joints.astype(np.int32)
                raw_pd_depth = np.array(map(lambda x: volume_utils.voxel_z_centers[x], raw_vol_joints[:, :, 2].tolist()))
                raw_pd_coords_2d = raw_vol_joints[:, :, 0:2] * configs.coords_2d_scale

                # ############# evaluate the coords recovered from the gt 2d and gt root depth
                for b in range(configs.batch_size):
                    mean_c_j_2d_pd, mean_c_j_3d_pd, _ = volume_utils.local_to_global(mean_pd_depth[b], depth_root_arr[b], mean_pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    raw_c_j_2d_pd, raw_c_j_3d_pd, _ = volume_utils.local_to_global(raw_pd_depth[b], depth_root_arr[b], raw_pd_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])

                    # Here I used the root aligned pose to evaluate the error
                    # according to https://github.com/geopavlakos/c2f-vol-demo/blob/master/matlab/utils/errorH36M.m
                    mean_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], mean_c_j_3d_pd - mean_c_j_3d_pd[0])
                    raw_coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], raw_c_j_3d_pd - raw_c_j_3d_pd[0])

                sys.stdout.write("Mean: ")
                mean_coords_eval.printMean()

                sys.stdout.write("Raw: ")
                raw_coords_eval.printMean()

                print("\n\n")

            mean_coords_eval.save("../eval_result/f_4_1/coord_eval_{}w_mean.npy".format(cur_model_iterations / 10000))
            raw_coords_eval.save("../eval_result/f_4_1/coord_eval_{}w_raw.npy".format(cur_model_iterations / 10000))
