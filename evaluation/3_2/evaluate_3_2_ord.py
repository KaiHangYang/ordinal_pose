import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_3_2
from utils.preprocess_utils import ordinal_3_2 as preprocessor
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.postprocess_utils import volume_utils

##################### Evaluation Configs ######################
import configs

# t means gt(0) or ord(1)
# d means validset(0) or trainset(1)
configs.parse_configs(1, 0)
configs.print_configs()

evaluation_models = [175000, 200000, 225000, 300000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    #### Used for valid
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

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_coords_2d = tf.placeholder(shape=[None, configs.nJoints, 2], dtype=tf.float32, name="input_coords_2d")
    input_relation_table = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_relation_table")
    input_loss_table_log = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_log")
    input_loss_table_pow = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_pow")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    ordinal_model = ordinal_3_2.mOrdinal_3_2(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=False, coords_2d_scale=configs.coords_2d_scale, coords_2d_offset=configs.coords_2d_offset, rank_loss_weight=1.0, keyp_loss_weight=1000.0)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_no_gt(input_coords_2d=input_coords_2d, relation_table=input_relation_table, loss_table_log=input_loss_table_log, loss_table_pow=input_loss_table_pow, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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

            ##################### First get the depth scale from the subset of the training set ######################
            cur_model_depth_scale = my_utils.mAverageCounter(shape=[1])
            scale_data_index = my_utils.mRangeVariable(min_val=scale_data_from, max_val=scale_data_to-1, initial_val=scale_data_from)
            while not scale_data_index.isEnd():
                batch_images_np = np.zeros([configs.scale_batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_coords_2d_np = np.zeros([configs.scale_batch_size, configs.nJoints, 2], dtype=np.float32)
                batch_relation_table_np = np.zeros([configs.scale_batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
                batch_loss_table_log_np = np.zeros([configs.scale_batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
                batch_loss_table_pow_np = np.zeros([configs.scale_batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)

                ##### The depth is all related to the root
                gt_depth_arr = []

                for b in range(configs.scale_batch_size):

                    cur_img = cv2.imread(scale_img_list[scale_data_index.val])
                    cur_label = np.load(scale_lbl_list[scale_data_index.val]).tolist()
                    scale_data_index.val += 1

                    ########## Save the data for evaluation ###########
                    gt_joints_3d = cur_label["joints_3d"].copy()
                    gt_depth_arr.append(gt_joints_3d[:, 2] - gt_joints_3d[0, 2])
                    ###################################################
                    cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

                    cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, do_rotate=False, is_training=False)

                    batch_images_np[b] = cur_img
                    batch_coords_2d_np[b] = (cur_joints[:, 0:2] - configs.coords_2d_offset) / configs.coords_2d_scale
                    batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints[:, 2])

                scale_acc_2d, \
                scale_loss, \
                scale_result  = sess.run(
                        [ordinal_model.accuracy_2d,
                         ordinal_model.loss,
                         ordinal_model.result],
                        feed_dict={input_images: batch_images_np, input_coords_2d: batch_coords_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_batch_size: configs.scale_batch_size})

                scale_for_show = []

                scale_depth = scale_result[:, :, 2]

                for b in range(configs.scale_batch_size):
                    scale_depth_related_to_root = scale_depth[b] - scale_depth[b][0]
                    cur_scale = (np.max(gt_depth_arr[b]) - np.min(gt_depth_arr[b])) / (np.max(scale_depth_related_to_root) - np.min(scale_depth_related_to_root) + 1e-7)
                    cur_model_depth_scale.add(cur_scale)

                    scale_for_show.append(cur_scale)

                print("Iter: {:07d} Loss : {:07f} Scales: {}\n\n".format(scale_data_index.val, scale_loss, scale_for_show))
                print("Accuracy 2D: {:07f}".format(scale_acc_2d))
                print("Cur Scale: {:07f}\n\n".format(cur_model_depth_scale.cur_average[0]))
            ################################################################################################

            ##### Then evaluate it #####
            cur_depth_scale = cur_model_depth_scale.cur_average[0]
            print("Scale used to evaluate: {:07f}".format(cur_depth_scale))

            depth_eval = evaluators.mEvaluatorDepth(nJoints=configs.nJoints)
            coords_eval = evaluators.mEvaluatorPose3D(nJoints=configs.nJoints)
            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

            while not data_index.isEnd():
                global_steps = sess.run(ordinal_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_coords_2d_np = np.zeros([configs.scale_batch_size, configs.nJoints, 2], dtype=np.float32)
                batch_relation_table_np = np.zeros([configs.batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
                batch_loss_table_log_np = np.zeros([configs.batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
                batch_loss_table_pow_np = np.zeros([configs.batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)

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

                    cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, do_rotate=False, is_training=False)

                    batch_images_np[b] = cur_img
                    batch_coords_2d_np[b] = (cur_joints[:, 0:2] - configs.coords_2d_offset) / configs.coords_2d_scale
                    batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints[:, 2])

                acc_2d, \
                loss, \
                result  = sess.run(
                        [ordinal_model.accuracy_2d,
                         ordinal_model.loss,
                         ordinal_model.result],
                        feed_dict={input_images: batch_images_np, input_coords_2d: batch_coords_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_batch_size: configs.batch_size})

                print("Iter: {:07d}. Loss : {:07f}. Accuracy 2D : {:07f}\n\n".format(data_index.val, loss, acc_2d))
                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))

                # multiply the scale
                result_depth = result[:, :, 2].copy()
                result_coords_2d = result[:, :, 0:2] * configs.coords_2d_scale + configs.coords_2d_offset

                result_depth = result_depth - result_depth[:, 0]
                result_depth = cur_depth_scale * result_depth

                depth_eval.add(np.array(gt_depth_arr), result_depth)
                depth_eval.printMean()

                ############# evaluate the coords recovered from the pd 2d and gt root depth
                for b in range(configs.batch_size):
                    c_j_2d, c_j_3d, _ = volume_utils.local_to_global(result_depth[b], depth_root_arr[b], result_coords_2d[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    coords_eval.add(gt_joints_3d_arr[b] - gt_joints_3d_arr[b][0], c_j_3d - c_j_3d[0])

                coords_eval.printMean()
                print("\n\n")

            depth_eval.save("../eval_result/ord_3_2/depth_eval_{}w.npy".format(cur_model_iterations / 10000))
            coords_eval.save("../eval_result/ord_3_2/coord_eval_{}w.npy".format(cur_model_iterations / 10000))
