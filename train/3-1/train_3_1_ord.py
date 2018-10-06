import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_3_1
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
from utils.visualize_utils import display_utils

##################### Setting for training ######################
import configs

# t means gt(0) or ord(1)
configs.parse_configs(1)
configs.print_configs()

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
is_reset_global_steps = False
#################################################################

if __name__ == "__main__":

    ############################ Train and valid data list ##########################
    train_range = np.load(configs.train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.valid_range_file)
    train_img_list = [configs.train_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = ordinal_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=configs.train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = ordinal_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=configs.valid_batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_relation_table = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_relation_table")
    input_loss_table_log = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_log")
    input_loss_table_pow = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_pow")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool)
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32)

    ordinal_model = ordinal_3_1.mOrdinal_3_1(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, depth_scale=configs.depth_scale, loss_weight=configs.loss_weight)

    with tf.Session() as sess:

        ordinal_model.build_model(input_images)
        ordinal_model.build_loss_no_gt(relation_table=input_relation_table, loss_table_log=input_loss_table_log, loss_table_pow=input_loss_table_pow, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

        print("Network built!")
        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)

        model_saver = tf.train.Saver(max_to_keep=70)
        net_init = tf.global_variables_initializer()

        sess.run([train_data_init_op, valid_data_init_op, net_init])

        # reload the model
        if restore_model_iteration is not None:
            if os.path.exists(configs.model_path_fn(restore_model_iteration)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(restore_model_iteration))
            else:
                print("The prev model is not existing!")
                quit()

        if is_reset_global_steps:
            sess.run(tf.train.get_or_create_global_step().assign(0))

        is_valid = False
        valid_count = 0

        while True:
            global_steps = sess.run(ordinal_model.global_steps)

            if valid_count == configs.valid_iter:
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

            batch_size = len(cur_data_batch[0])
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_relation_table_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
            batch_loss_table_log_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
            batch_loss_table_pow_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]
            label_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[1][b])

                cur_img = cv2.imread(cur_data_batch[0][b])
                cur_label = np.load(cur_data_batch[1][b]).tolist()

                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

                # Cause the dataset is to large, test no augment
                cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints)
                batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])
                batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints[:, 2])

            if is_valid:
                loss, \
                lr, \
                summary  = sess.run(
                        [ordinal_model.loss,
                         ordinal_model.lr,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                _,\
                loss,\
                lr,\
                summary  = sess.run(
                        [ordinal_model.train_op,
                         ordinal_model.loss,
                         ordinal_model.lr,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, global_steps)

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nLoss : {:07f}\n\n\n".format(global_steps, lr, loss))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 25000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
