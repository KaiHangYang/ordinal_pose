import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import ordinal_3_1
from utils.dataread_utils import ordinal_3_1_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
from utils.common_utils import my_utils

##################### Setting for training ######################

nJoints = 17
batch_size = 1
img_size = 256
learning_rate = 2.5e-7

valid_log_dir = "../logs/evaluate/valid_3_1_ordinal"

restore_model_path = "../models/ordinal_3_1_ordinal-300000"

valid_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images/{}.jpg".format(x)
valid_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/labels/{}.npy".format(x)

#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################

    # train_img_list = [train_img_path(i) for i in train_range]
    # train_lbl_list = [train_lbl_path(i) for i in train_range]
    data_from = 0
    data_to = 109867
    valid_data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)
    valid_range = np.arange(data_from, data_to)

    valid_img_list = [valid_img_path(i) for i in valid_range]
    valid_lbl_list = [valid_lbl_path(i) for i in valid_range]

    input_images = tf.placeholder(shape=[batch_size, img_size, img_size, 3], dtype=tf.float32)
    input_relation_table = tf.placeholder(shape=[batch_size, nJoints, nJoints], dtype=tf.float32, name="relation_table")
    input_loss_table_log = tf.placeholder(shape=[batch_size, nJoints, nJoints], dtype=tf.float32, name="loss_table_log")
    input_loss_table_pow = tf.placeholder(shape=[batch_size, nJoints, nJoints], dtype=tf.float32, name="loss_table_pow")
    ordinal_model = ordinal_3_1.mOrdinal_3_1(nJoints, img_size, batch_size, is_training=True)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_no_gt(input_relation_table, input_loss_table_log, input_loss_table_pow, learning_rate)

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
            batch_relation_table_np = np.zeros([batch_size, nJoints, nJoints], dtype=np.float32)
            batch_loss_table_log_np = np.zeros([batch_size, nJoints, nJoints], dtype=np.float32)
            batch_loss_table_pow_np = np.zeros([batch_size, nJoints, nJoints], dtype=np.float32)

            # batch_size is always 1 when evaluate
            # Generate the data batch
            cur_img = cv2.imread(valid_img_list[valid_data_index.val])
            cur_label = np.load(valid_lbl_list[valid_data_index.val]).tolist()
            valid_data_index.val += 1

            cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)
            # cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints)

            batch_relation_table_np[0], batch_loss_table_log_np[0], batch_loss_table_pow_np[0] = preprocessor.get_relation_table(cur_joints[:, 2])
            batch_images_np[0] = preprocessor.img2train(cur_img, [-1, 1])

            print(batch_relation_table_np[0], batch_loss_table_log_np[0], batch_loss_table_pow_np[0])

            depth, loss, summary  = sess.run([ordinal_model.result, ordinal_model.loss, ordinal_model.merged_summary],
                    feed_dict={input_images: batch_images_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np})
            valid_log_writer.add_summary(summary, global_steps)

            print("Loss : {:07f} . Result: {}".format(loss, depth))

            # Then do other evaluations
