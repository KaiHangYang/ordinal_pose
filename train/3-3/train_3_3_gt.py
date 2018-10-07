import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
configs.parse_configs(0)
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

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    input_centers = tf.placeholder(shape=[None, configs.nJoints, 3], dtype=tf.float32)
    input_is_training = tf.placeholder(shape=[], dtype=tf.bool)
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32)

    ordinal_model = ordinal_3_3.mOrdinal_3_3(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_volume=configs.loss_weight_volume)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
            input_volumes = ordinal_model.build_input_volumes(input_centers, stddev=2.0, name="input_volumes")
        ordinal_model.build_loss_gt(input_volumes=input_volumes, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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
            batch_centers_np = np.zeros([batch_size, configs.nJoints, 3], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]
            label_path_for_show = [[] for i in range(len(img_path_for_show))]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[1][b])

                cur_img = cv2.imread(cur_data_batch[0][b])
                display_imgs_raw = cur_img.copy()
                cur_label = np.load(cur_data_batch[1][b]).tolist()

                cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy() # cause lua is from 1 to n not 0 to n-1

                cur_joints = np.concatenate([cur_label["joints_2d"], cur_joints_zidx[:, np.newaxis]], axis=1)

                cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints, is_training=not is_valid)

                # currently the cur_img is in range [0, 1]
                batch_images_np[b] = cur_img
                hm_joint_2d = np.round(cur_joints[:, 0:2] / configs.coords_2d_scale)
                batch_centers_np[b] = np.concatenate([hm_joint_2d, cur_joints[:, 2][:, np.newaxis]], axis=1)

                ############# Visualize and debug the augmented datas #############
                # display_imgs = batch_images_np[b].copy()
                # display_imgs = display_utils.drawLines(display_imgs, batch_centers_np[b][:, 0:2]*configs.coords_2d_scale)
                # display_imgs = display_utils.drawPoints(display_imgs, batch_centers_np[b][:, 0:2]*configs.coords_2d_scale)

                # display_imgs_raw = display_utils.drawLines(display_imgs_raw, cur_label["joints_2d"])
                # display_imgs_raw = display_utils.drawPoints(display_imgs_raw, cur_label["joints_2d"])

                # print("augged: {}".format(batch_centers_np[b][:, 2]))
                # print("raw: {}".format(cur_joints_zidx))
                # cv2.imshow("raw_img", display_imgs_raw)
                # cv2.imshow("cur_img", display_imgs)
                # cv2.waitKey()

            acc = 0

            if is_valid:
                acc, \
                loss, \
                lr, \
                summary  = sess.run(
                        [
                         ordinal_model.accuracy,
                         ordinal_model.loss,
                         ordinal_model.lr,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers: batch_centers_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                if global_steps % write_log_iter == 0:
                    acc, \
                    _,\
                    loss,\
                    lr,\
                    summary  = sess.run(
                            [
                             ordinal_model.accuracy,
                             ordinal_model.train_op,
                             ordinal_model.loss,
                             ordinal_model.lr,
                             ordinal_model.merged_summary],
                            feed_dict={input_images: batch_images_np, input_centers: batch_centers_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                    train_log_writer.add_summary(summary, global_steps)
                else:
                    _,\
                    loss,\
                    lr = sess.run(
                            [
                             ordinal_model.train_op,
                             ordinal_model.loss,
                             ordinal_model.lr],
                            feed_dict={input_images: batch_images_np, input_centers: batch_centers_np, input_is_training: True, input_batch_size: configs.train_batch_size})


            # assert((gt_joints_vol == batch_centers_np).all())

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nLoss : {:07f}\nAccuracy : {:07f}\n\n".format(global_steps, lr, loss, acc))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 25000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
