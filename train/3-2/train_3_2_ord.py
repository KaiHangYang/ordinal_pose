import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_3_2
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.preprocess_utils import ordinal_3_2 as preprocessor
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
#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
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
    input_coords_2d = tf.placeholder(shape=[None, configs.nJoints, 2], dtype=tf.float32, name="input_coords_2d")
    input_relation_table = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_relation_table")
    input_loss_table_log = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_log")
    input_loss_table_pow = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_pow")
    input_is_training = tf.placeholder(shape=[], dtype=tf.bool)
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32)

    ordinal_model = ordinal_3_2.mOrdinal_3_2(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, coords_2d_scale=configs.coords_2d_scale, rank_loss_weight=1.0, keyp_loss_weight=1000.0)

    with tf.Session() as sess:

        ordinal_model.build_model(input_images)
        ordinal_model.build_loss_no_gt(input_coords_2d=input_coords_2d, relation_table=input_relation_table, loss_table_log=input_loss_table_log, loss_table_pow=input_loss_table_pow, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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
            batch_coords_2d_np = np.zeros([batch_size, configs.nJoints, 2], dtype=np.float32)
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

                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"]], axis=1)

                # cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, do_rotate=True)
                cur_joints_2d = cur_joints[:, 0:2].copy()
                cur_joints_3d = cur_joints[:, 2:5].copy()

                batch_coords_2d_np[b] = (cur_joints_2d / configs.coords_2d_scale).copy()
                batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints_3d[:, 2])
                batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])

                ############### Visualize the augmentated datas
                # show_img = cur_img.copy().astype(np.uint8)
                # show_img = display_utils.drawLines(show_img, cur_joints[:, 0:2])
                # show_img = display_utils.drawPoints(show_img, cur_joints[:, 0:2])

                # print((cur_joints[:, 2] == cur_label["joints_3d"][:, 2]).all())

                # cv2.imshow("img_show", show_img)
                # cv2.waitKey()
                ###############################################

            if is_valid:
                loss, \
                rank_loss, \
                coord2d_loss, \
                lr, \
                accuracy_2d, \
                summary  = sess.run(
                        [ordinal_model.loss,
                         ordinal_model.rank_loss,
                         ordinal_model.coord2d_loss,
                         ordinal_model.lr,
                         ordinal_model.accuracy_2d,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_coords_2d: batch_coords_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                _,\
                loss,\
                rank_loss,\
                coord2d_loss,\
                lr,\
                accuracy_2d, \
                summary  = sess.run(
                        [ordinal_model.train_op,
                         ordinal_model.loss,
                         ordinal_model.rank_loss,
                         ordinal_model.coord2d_loss,
                         ordinal_model.lr,
                         ordinal_model.accuracy_2d,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_coords_2d: batch_coords_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, global_steps)

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nAccuracy 2D: {:07f} \nLoss : {:07f}\nRank Loss: {:07f}\nCoord2d Loss: {:07f}\n\n".format(global_steps, lr, accuracy_2d, loss, rank_loss, coord2d_loss))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 25000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
