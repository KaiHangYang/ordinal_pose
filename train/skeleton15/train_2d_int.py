import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import net_int_2d as net_int
from utils.dataread_utils import syn_reader as data_reader
from utils.preprocess_utils import int_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils

##################### Setting for training ######################
configs = mConfigs("../train.conf", "net_int_l2")
################ Reseting  #################
configs.loss_weight_heatmap = 1.0
configs.loss_weight_integral = 1.0
configs.pose_2d_scale = 4.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)
configs.is_use_bn = True

configs.learning_rate = 2.5e-4
configs.lr_decay_rate = 0.10
configs.lr_decay_step = 300000
configs.nFeats = 256
configs.nModules = 2
configs.nStacks = 1
################### Initialize the data reader ####################
configs.printConfig()
preprocessor = int_preprocess.INTProcessor(skeleton=skeleton, img_size=configs.img_size)

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
#################################################################

if __name__ == "__main__":
    train_range = np.load(configs.h36m_train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.h36m_valid_range_file)

    train_img_list = [configs.h36m_train_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.h36m_train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.h36m_valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.h36m_valid_lbl_path_fn(i) for i in valid_range]

    mpii_range = np.load(configs.mpii_range_file)
    lsp_range = np.load(configs.lsp_range_file)

    mpii_lsp_img_list = [configs.mpii_img_path_fn(i) for i in mpii_range] + [configs.lsp_img_path_fn(i) for i in lsp_range]
    mpii_lsp_lbl_list = [configs.mpii_lbl_path_fn(i) for i in mpii_range] + [configs.lsp_lbl_path_fn(i) for i in lsp_range]

    # increase the mpii_lsp datas
    mpii_lsp_img_list = mpii_lsp_img_list * 10
    mpii_lsp_lbl_list = mpii_lsp_lbl_list * 10

    train_img_list = train_img_list + mpii_lsp_img_list
    train_lbl_list = train_lbl_list + mpii_lsp_lbl_list
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = data_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=configs.train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = data_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=configs.valid_batch_size, name="valid_reader", is_shuffle=False)

    # now test the classification
    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_joints_2d = tf.placeholder(shape=[None, skeleton.n_joints, 2], dtype=tf.float32, name="input_joints_2d")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    int_model = net_int.mINTNet(skeleton=skeleton, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_integral=configs.loss_weight_integral, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nModules=configs.nModules, nStacks=configs.nStacks)

    train_valid_counter = my_utils.mTrainValidCounter(train_steps=configs.valid_iter, valid_steps=1)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            int_model.build_model(input_images)
            input_heatmap = int_model.build_input_heatmaps(input_center=input_joints_2d, stddev=1.0, gaussian_coefficient=True, name="input_heatmap")

        int_model.build_loss(input_heatmap=input_heatmap, input_joints_2d=input_joints_2d, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)
        print("Network built!")

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

        while True:
            global_steps = sess.run(int_model.global_steps)

            # get the data path
            cur_img_batch, cur_lbl_batch = sess.run(train_data_iter if train_valid_counter.is_training else valid_data_iter)

            batch_size = len(cur_img_batch)
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)

            img_path_for_show = [[] for i in range(batch_size)]
            lbl_path_for_show = [[] for i in range(batch_size)]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_img_batch[b])
                lbl_path_for_show[b] = os.path.basename(cur_lbl_batch[b])

                cur_img = cv2.imread(cur_img_batch[b])
                cur_label = np.load(cur_lbl_batch[b]).tolist()

                if "joints_3d" in cur_label.keys():
                    # human3.6m datas
                    cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                else:
                    cur_joints_2d = cur_label["joints_2d"].copy()

                cur_img, cur_joints_2d = preprocessor.preprocess(img=cur_img, joints_2d=cur_joints_2d, is_training=train_valid_counter.is_training)

                # generate the heatmaps
                batch_images_np[b] = cur_img
                batch_joints_2d_np[b] = cur_joints_2d / configs.pose_2d_scale

                ########## Visualize the datas ###########
                # cv2.imshow("img", cur_img)
                # cv2.imshow("test", display_utils.drawLines((255.0 * cur_img).astype(np.uint8), cur_joints_2d, indices=skeleton.bone_indices, color_table=skeleton.bone_colors * 255))

                # cv2.waitKey()
                ##########################################

            if train_valid_counter.is_training:
                _, \
                acc_hm, \
                acc_int, \
                total_loss,\
                heatmaps_loss, \
                integral_loss, \
                lr,\
                summary  = sess.run(
                        [
                         int_model.train_op,
                         int_model.heatmaps_acc,
                         int_model.integral_acc,
                         int_model.total_loss,
                         int_model.heatmap_losses,
                         int_model.integral_losses,
                         int_model.lr,
                         int_model.merged_summary],
                        feed_dict={input_images: batch_images_np,
                                   input_joints_2d: batch_joints_2d_np,
                                   input_is_training: True,
                                   input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, global_steps)
            else:
                acc_hm, \
                acc_int, \
                total_loss, \
                heatmaps_loss, \
                integral_loss, \
                lr, \
                summary  = sess.run(
                        [
                         int_model.heatmaps_acc,
                         int_model.integral_acc,
                         int_model.total_loss,
                         int_model.heatmap_losses,
                         int_model.integral_losses,
                         int_model.lr,
                         int_model.merged_summary],
                        feed_dict={input_images: batch_images_np,
                                   input_joints_2d: batch_joints_2d_np,
                                   input_is_training: False,
                                   input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)

            print("Train Iter:\n" if train_valid_counter.is_training else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}".format(global_steps, lr, total_loss))
            for l_idx in range(len(heatmaps_loss)):
                print("Heatmap loss level {}: {}".format(l_idx, heatmaps_loss[l_idx]))

            for l_idx in range(len(integral_loss)):
                print("Integral loss level {}: {}".format(l_idx, integral_loss[l_idx]))

            print("Heatmap Accuracy: {}\n".format(acc_hm))
            print("Integral Accuracy: {}\n".format(acc_int))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, lbl_path_for_show)))
            print("\n\n")

            if global_steps % 20000 == 0 and train_valid_counter.is_training:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and train_valid_counter.is_training:
                break

            ################### Pay attention to this !!!!!!################
            train_valid_counter.next()