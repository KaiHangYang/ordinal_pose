import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

##################### Setting for training ######################

nJoints = 17
batch_size = 4
img_size = 256
############# path parameters
train_log_dir = "../logs/train/3_1_ord/train"
valid_log_dir = "../logs/train/3_1_ord/valid"
model_dir = "../models/3_1_ord/"
model_name = "ordinal_3_1_ordinal"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

is_restore = False
restore_model_path = ""

valid_iter = 5
train_iter = 300000
learning_rate = 2.5e-6

train_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images/{}.jpg".format(x)
train_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/labels/{}.npy".format(x)

valid_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/images/{}.jpg".format(x)
valid_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/labels/{}.npy".format(x)

#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    train_range = np.arange(0, 312188)
    valid_range = np.arange(0, 109867)

    train_img_list = [train_img_path(i) for i in train_range]
    train_lbl_list = [train_lbl_path(i) for i in train_range]

    valid_img_list = [valid_img_path(i) for i in valid_range]
    valid_lbl_list = [valid_lbl_path(i) for i in valid_range]

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = ordinal_3_1_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = ordinal_3_1_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=batch_size, name="valid_reader")

    input_images = tf.placeholder(shape=[batch_size, img_size, img_size, 3], dtype=tf.float32)
    input_relation_table = tf.placeholder(shape=[batch_size, nJoints, nJoints], dtype=tf.float32)
    input_loss_table_log = tf.placeholder(shape=[batch_size, nJoints, nJoints], dtype=tf.float32)
    input_loss_table_pow = tf.placeholder(shape=[batch_size, nJoints, nJoints], dtype=tf.float32)
    ordinal_model = ordinal_3_1.mOrdinal_3_1(nJoints, img_size, batch_size, is_training=True)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_no_gt(input_relation_table, input_loss_table_log, input_loss_table_pow, learning_rate)

        print("Network built!")

        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)

        model_saver = tf.train.Saver(max_to_keep=70)
        net_init = tf.global_variables_initializer()

        sess.run([train_data_init_op, valid_data_init_op, net_init])

        # reload the model
        if is_restore:
            if os.path.exists(restore_model_path+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, restore_model_path)
            else:
                print("The prev model is not existing!")
                quit()

        is_valid = False
        valid_count = 0

        while True:
            global_steps = sess.run(ordinal_model.global_steps)

            if valid_count == valid_iter:
                valid_count = 0
                is_valid = True
            else:
                valid_count += 1
                is_valid = False

            # get the data path
            if is_valid:
                cur_data_batch = sess.run(valid_data_iter)
            else:
                cur_data_batch = sess.run(train_data_iter)

            batch_images_np = np.zeros([batch_size, img_size, img_size, 3], dtype=np.float32)
            batch_relation_table_np = np.zeros([batch_size, nJoints, nJoints], dtype=np.float32)
            batch_loss_table_log_np = np.zeros([batch_size, nJoints, nJoints], dtype=np.float32)
            batch_loss_table_pow_np = np.zeros([batch_size, nJoints, nJoints], dtype=np.float32)

            # Generate the data batch
            for b in range(batch_size):
                cur_img = cv2.imread(cur_data_batch[0][b])
                cur_label = np.load(cur_data_batch[1][b]).tolist()
                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)
                cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints)

                batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints[:, 2])
                batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])

                ############### Visualize the augmentated datas
                # show_img = cur_img.copy().astype(np.uint8)
                # show_img = display_utils.drawLines(show_img, cur_joints[:, 0:2])
                # show_img = display_utils.drawPoints(show_img, cur_joints[:, 0:2])

                # aug_relation_table = batch_relation_table_np[b].copy()
                # aug_relation_table -= np.transpose(aug_relation_table.copy())

                # cur_depth = cur_joints[:, 2]
                # cur_depth_row = np.repeat(cur_depth[:, np.newaxis], nJoints, axis=1)
                # cur_depth_col = np.repeat(cur_depth[np.newaxis], nJoints, axis=0)

                # cur_rel_t = cur_depth_row - cur_depth_col
                # cur_rel_t[np.abs(cur_rel_t) < 100] = 0
                # cur_rel_t[cur_rel_t >= 100] = -1
                # cur_rel_t[cur_rel_t <= -100] = 1

                # assert((aug_relation_table == cur_rel_t).all())

                # cv2.imshow("img_show", show_img)
                # cv2.waitKey()
                ###############################################

            if is_valid:
                loss, summary  = sess.run([ordinal_model.loss, ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                _, loss, summary  = sess.run([ordinal_model.train_op, ordinal_model.loss, ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np})
                train_log_writer.add_summary(summary, global_steps)

            print("Iteration: {:07d} Loss : {:07f} ".format(global_steps, loss))

            if global_steps % 50000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=os.path.join(model_dir, model_name), global_step=global_steps)

            if global_steps >= train_iter and not is_valid:
                break

