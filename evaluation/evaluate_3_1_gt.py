import os
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import ordinal_3_1
from utils.dataread_utils import ordinal_3_1_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils

##################### Setting for training ######################

nJoints = 17
batch_size = 1
img_size = 256

######################## To modify #############################
trash_log = "trash_"
# train_log_dir = "../logs/evaluate/3_1_gt/train"
valid_log_dir = "../"+trash_log+"logs/evaluate/3_1_gt/valid"
valid_data_source = "valid"
################################################################

restore_model_path = "../models/ordinal_3_1_1_gt-150000"
learning_rate = 2.5e-4
lr_decay_rate = 0.96
lr_decay_step = 2000


valid_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+valid_data_source+"/images/{}.jpg".format(x)
valid_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+valid_data_source+"/labels/{}.npy".format(x)

#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################

    valid_range = np.load("../train/train_range/sec_3/3_1_1/"+valid_data_source+"_range.npy")
    data_from = 0
    data_to = len(valid_range)
    valid_data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

    valid_img_list = [valid_img_path(i) for i in valid_range]
    valid_lbl_list = [valid_lbl_path(i) for i in valid_range]

    input_images = tf.placeholder(shape=[batch_size, img_size, img_size, 3], dtype=tf.float32)
    input_depths = tf.placeholder(shape=[batch_size, nJoints], dtype=tf.float32)
    ordinal_model = ordinal_3_1.mOrdinal_3_1(nJoints, img_size, batch_size, is_training=False)

    cur_average = 0

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_gt(input_depths, learning_rate, lr_decay_rate=lr_decay_rate, lr_decay_step=lr_decay_step)

        print("Network built!")
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model
        if os.path.exists(restore_model_path+".index"):
            print("#######################Restored all weights ###########################")
            model_saver.restore(sess, restore_model_path)
        else:
            print("The prev model is not existing!")
            quit()

        while not valid_data_index.isEnd():
            global_steps = sess.run(ordinal_model.global_steps)

            batch_images_np = np.zeros([batch_size, img_size, img_size, 3], dtype=np.float32)
            batch_depth_np = np.zeros([batch_size, nJoints], dtype=np.float32)

            cur_img = cv2.imread(valid_img_list[valid_data_index.val])
            cur_label = np.load(valid_lbl_list[valid_data_index.val]).tolist()
            valid_data_index.val += 1

            cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

            batch_depth_np[0] = cur_joints[:, 2] - cur_joints[0, 2] # related to the root
            batch_images_np[0] = preprocessor.img2train(cur_img, [-1, 1])

            depth, loss, summary  = sess.run([ordinal_model.result, ordinal_model.loss, ordinal_model.merged_summary],
                    feed_dict={input_images: batch_images_np, input_depths: batch_depth_np})
            valid_log_writer.add_summary(summary, global_steps)

            print("Iteration: {:07d} Loss : {:07f} ".format(global_steps, loss))

            cur_average = (cur_average * (valid_data_index.val - 1) + np.mean(np.abs(batch_depth_np[0] - depth[0])) ) / valid_data_index.val
            print(cur_average)
