import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import ordinal_3_3
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.preprocess_utils import ordinal_3_3 as preprocessor
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
    input_centers_2d = tf.placeholder(shape=[None, configs.nJoints, 2], dtype=tf.float32, name="input_centers_2d")
    input_relation_table = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_relation_table")
    input_loss_table_log = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_log")
    input_loss_table_pow = tf.placeholder(shape=[None, configs.nJoints, configs.nJoints], dtype=tf.float32, name="input_loss_table_pow")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    ordinal_model = ordinal_3_3.mOrdinal_3_3(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, rank_loss_weight=1.0, hm_loss_weight=1000.0)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
            input_heatmaps = ordinal_model.build_input_heatmaps(input_centers_2d, stddev=2.0, gaussian_coefficient=True, name="input_heatmaps")
        ordinal_model.build_loss_no_gt(input_heatmaps=input_heatmaps, relation_table=input_relation_table, loss_table_log=input_loss_table_log, loss_table_pow=input_loss_table_pow, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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
        write_log_iter = configs.valid_iter

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
            batch_centers_2d_np = np.zeros([batch_size, configs.nJoints, 2], dtype=np.float32)

            batch_relation_table_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
            batch_loss_table_log_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
            batch_loss_table_pow_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]
            label_path_for_show = [[] for i in range(len(img_path_for_show))]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[1][b])

                cur_img = cv2.imread(cur_data_batch[0][b])
                display_imgs_raw = cur_img.copy()
                cur_label = np.load(cur_data_batch[1][b]).tolist()

                # cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy() # cause lua is from 1 to n not 0 to n-1
                # cur_joints = np.concatenate([cur_label["joints_2d"], cur_joints_zidx[:, np.newaxis]], axis=1)

                # Now use the 100mm range to generate the relation table
                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)
                cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, is_training=not is_valid, is_rotate=False)

                # currently the cur_img is in range [0, 1]
                batch_images_np[b] = cur_img
                hm_joint_2d = np.round(cur_joints[:, 0:2] / configs.coords_2d_scale)

                batch_centers_2d_np[b] = hm_joint_2d
                batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints[:, 2])

            acc_hm = 0

            if is_valid:
                acc_hm, \
                vol_loss, \
                rank_loss, \
                hm_loss, \
                lr, \
                summary  = sess.run(
                        [
                         ordinal_model.accuracy_hm,
                         ordinal_model.vol_loss,
                         ordinal_model.rank_loss,
                         ordinal_model.hm_loss,
                         ordinal_model.lr,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers_2d: batch_centers_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                if global_steps % write_log_iter == 0:
                    acc_hm, \
                    _,\
                    vol_loss, \
                    rank_loss, \
                    hm_loss, \
                    lr,\
                    summary  = sess.run(
                            [
                             ordinal_model.accuracy_hm,
                             ordinal_model.train_op,
                             ordinal_model.vol_loss,
                             ordinal_model.rank_loss,
                             ordinal_model.hm_loss,
                             ordinal_model.lr,
                             ordinal_model.merged_summary],
                            feed_dict={input_images: batch_images_np, input_centers_2d: batch_centers_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                    train_log_writer.add_summary(summary, global_steps)
                else:
                    _,\
                    vol_loss,\
                    rank_loss, \
                    hm_loss, \
                    lr = sess.run(
                            [
                             ordinal_model.train_op,
                             ordinal_model.vol_loss,
                             ordinal_model.rank_loss,
                             ordinal_model.hm_loss,
                             ordinal_model.lr],
                            feed_dict={input_images: batch_images_np, input_centers_2d: batch_centers_2d_np, input_relation_table: batch_relation_table_np, input_loss_table_log: batch_loss_table_log_np, input_loss_table_pow: batch_loss_table_pow_np, input_is_training: True, input_batch_size: configs.train_batch_size})


            # assert((gt_joints_vol == batch_centers_np).all())
            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}\nRank Loss : {:07f}\nHeatmap Loss : {:07f}\nAccuracy : {:07f}\n\n".format(global_steps, lr, vol_loss, rank_loss, hm_loss, acc_hm))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 20000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
