import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_3_1
from utils.dataread_utils import ordinal_3_1_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
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

evaluation_models = [300000, 350000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_depths = tf.placeholder(shape=[configs.batch_size, configs.nJoints], dtype=tf.float32)
    ordinal_model = ordinal_3_1.mOrdinal_3_1(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, depth_scale=configs.depth_scale)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_gt(input_depths, configs.learning_rate, lr_decay_rate=configs.lr_decay_rate, lr_decay_step=configs.lr_decay_step)

        print("Network built!")
        # log_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model

        for cur_model_iterations in evaluation_models:

            depth_eval = evaluators.mEvaluatorDepth(nJoints=configs.nJoints)
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
                batch_depth_np = np.zeros([configs.batch_size, configs.nJoints], dtype=np.float32)

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

                    cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

                    batch_depth_np[b] = (cur_joints[:, 2] - cur_joints[0, 2]) / configs.depth_scale # related to the root
                    batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])

                acc, depth, loss = sess.run([ordinal_model.accuracy, ordinal_model.result, ordinal_model.loss],
                        feed_dict={input_images: batch_images_np, input_depths: batch_depth_np})

                print("Iteration: {:07d} \nLoss : {:07f}\nDepth accuracy: {:07f}\n\n".format(global_steps, loss, acc))
                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))

                depth_eval.add(configs.depth_scale * batch_depth_np, configs.depth_scale * depth)
                depth_eval.printMean()

                ############# evaluate the coords recovered from the gt 2d and gt root depth
                for b in range(configs.batch_size):
                    c_j_2d, c_j_3d, _ = volume_utils.local_to_global(configs.depth_scale * depth[b], depth_root_arr[b], crop_joints_2d_arr[b], source_txt_arr[b], center_arr[b], scale_arr[b])
                    coords_eval.add(gt_joints_3d_arr[b], c_j_3d)

                coords_eval.printMean()
                print("\n\n")

            depth_eval.save("../eval_result/gt_3_1/depth_eval_{}w.npy".format(cur_model_iterations / 10000))
            coords_eval.save("../eval_result/gt_3_1/coord_eval_{}w.npy".format(cur_model_iterations / 10000))
