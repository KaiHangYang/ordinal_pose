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
configs.parse_configs(0, 0)
configs.print_configs()

evaluation_models = [50000, 100000, 150000, 200000, 250000, 300000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_coords = tf.placeholder(shape=[configs.batch_size, configs.nJoints, 3], dtype=tf.float32)
    ordinal_model = ordinal_3_2.mOrdinal_3_2(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, coords_scale=configs.coords_scale)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_gt(input_coords, configs.learning_rate, lr_decay_rate=configs.lr_decay_rate, lr_decay_step=configs.lr_decay_step)

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
                batch_coords_np = np.zeros([configs.batch_size, configs.nJoints, 3], dtype=np.float32)

                img_path_for_show = []
                label_path_for_show = []

                for b in range(configs.batch_size):
                    img_path_for_show.append(os.path.basename(img_list[data_index.val]))
                    label_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_img = cv2.imread(img_list[data_index.val])
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"]], axis=1)

                    cur_joints_3d = cur_joints[:, 2:5]
                    batch_coords_np[b] = (cur_joints_3d - cur_joints_3d[0]) / configs.coords_scale # related to the root
                    batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])

                acc, coords, loss = sess.run([ordinal_model.accuracy, ordinal_model.result, ordinal_model.loss],
                        feed_dict={input_images: batch_images_np, input_coords: batch_coords_np})

                print("Iteration: {:07d} \nLoss : {:07f}\nCoords accuracy: {:07f}\n\n".format(global_steps, loss, acc))
                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))

                coords_eval.add(configs.coords_scale * batch_coords_np, configs.coords_scale * coords)
                coords_eval.printMean()
                print("\n\n")

            coords_eval.save("../eval_result/gt_3_2/coord_eval_{}w.npy".format(cur_model_iterations / 10000))
