import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import fb_net
from utils.dataread_utils import syn_reader
from utils.preprocess_utils import fb_preprocess
from utils.visualize_utils import display_utils

##################### Setting for training ######################
import configs

# t means gt(0) or ord(1)
# ver means version
configs.parse_configs(t=0, ver=3)
configs.print_configs()

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
#################################################################

if __name__ == "__main__":
    # assume the train_batch_size and valid_batch_size is 4
    batch_size = 4
    h36m_batch_size = 2
    mpii_lsp_batch_size = 2

    configs.nJoints = 13

    # the corresponding bone_status selected array is (index_arr[1:] - 1)
    h36m_selected_index = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
    mpii_lsp_selected_index = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14])

    ################### Initialize the data reader ####################
    train_range = np.load(configs.train_range_file)
    np.random.shuffle(train_range)

    lsp_range = np.load(configs.lsp_range_file)
    np.random.shuffle(lsp_range)
    mpii_range = np.load(configs.mpii_range_file)
    np.random.shuffle(mpii_range)

    valid_range = np.load(configs.valid_range_file)

    train_img_list = [configs.train_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.train_lbl_path_fn(i) for i in train_range]

    lsp_img_list = [configs.lsp_img_path_fn(i) for i in lsp_range]
    lsp_lbl_list = [configs.lsp_lbl_path_fn(i) for i in lsp_range]

    mpii_img_list = [configs.mpii_img_path_fn(i) for i in mpii_range]
    mpii_lbl_list = [configs.mpii_lbl_path_fn(i) for i in mpii_range]

    mpii_lsp_img_list = mpii_img_list + lsp_img_list
    mpii_lsp_lbl_list = mpii_lbl_list + lsp_lbl_list

    valid_img_list = [configs.valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = syn_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=h36m_batch_size, name="h36m_train_reader")
        mpii_lsp_data_iter, mpii_lsp_data_init_op = syn_reader.get_data_iterator(mpii_lsp_img_list, mpii_lsp_lbl_list, batch_size=mpii_lsp_batch_size, name="mpii_lsp_reader")
        valid_data_iter, valid_data_init_op = syn_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_center_2d = tf.placeholder(shape=[None, configs.nJoints, 2], dtype=tf.float32, name="input_center_2d")

    # both the fb_info and are one-hot arrays
    input_fb_info = tf.placeholder(shape=[None, configs.nJoints-1, 3], dtype=tf.float32, name="input_fb_info")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")

    fb_model = fb_net.mFBNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=batch_size, is_training=input_is_training, loss_weight_heatmaps=1.0, loss_weight_fb=1.0, is_use_bn=False)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            fb_model.build_model(input_images)
            input_heatmaps = fb_model.build_input_heatmaps(input_center_2d, name="input_heatmaps", stddev=2.0, gaussian_coefficient=False)

        fb_model.build_loss(input_heatmaps=input_heatmaps, input_fb=input_fb_info, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

        print("Network built!")
        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)

        model_saver = tf.train.Saver(max_to_keep=70)
        net_init = tf.global_variables_initializer()

        sess.run([train_data_init_op, valid_data_init_op, mpii_lsp_data_init_op, net_init])
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
            global_steps = sess.run(fb_model.global_steps)

            if valid_count == configs.valid_iter:
                valid_count = 0
                is_valid = True
            else:
                valid_count += 1
                is_valid = False

            # get the data path
            if is_valid:
                cur_data_batch = sess.run(valid_data_iter)
                cur_data_types = np.zeros(batch_size)
            else:
                tmp_data_batch = sess.run([train_data_iter, mpii_lsp_data_iter])
                cur_data_batch = np.concatenate([np.array(tmp_data_batch[0]), np.array(tmp_data_batch[1])], axis=1)
                cur_data_types = np.array([0, 0, 1, 1])

            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_center_2d = np.zeros([batch_size, configs.nJoints, 2], dtype=np.float32)
            batch_fb_info = np.zeros([batch_size, configs.nJoints-1, 3], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(batch_size)]
            label_path_for_show = [[] for i in range(batch_size)]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[1][b])

                if not is_valid and cur_data_types[b] == 0:
                    cur_img_num = int(img_path_for_show[b].split(".")[0])
                    cur_mask = cv2.imread(configs.train_mask_path_fn(cur_img_num))
                else:
                    cur_mask = None

                if cur_data_types[b] == 0:
                    # the human3.6m data
                    cur_selected_index = h36m_selected_index
                    cur_bone_selected_index = h36m_selected_index[1:] - 1
                else:
                    # the mpii and lsp data
                    cur_selected_index = mpii_lsp_selected_index
                    cur_bone_selected_index = mpii_lsp_selected_index[1:] - 1

                cur_img = cv2.imread(cur_data_batch[0][b])
                cur_label = np.load(cur_data_batch[1][b]).tolist()

                cur_joints_2d = cur_label["joints_2d"].copy()[cur_selected_index]
                cur_bone_status = cur_label["bone_status"].copy()[cur_bone_selected_index]

                cur_img, cur_joints_2d, cur_bone_status = fb_preprocess.preprocess(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, is_training=not is_valid, mask=cur_mask)

                batch_images_np[b] = cur_img
                batch_center_2d[b] = np.round(cur_joints_2d / configs.joints_2d_scale)
                batch_fb_info[b] = np.eye(3)[cur_bone_status]

            if is_valid:
                loss, \
                heatmaps_loss, \
                fb_loss, \
                hm_acc,\
                fb_acc,\
                lr, \
                summary  = sess.run(
                        [
                         fb_model.total_loss,
                         fb_model.heatmaps_loss,
                         fb_model.fb_loss,
                         fb_model.heatmaps_acc,
                         fb_model.fb_acc,
                         fb_model.lr,
                         fb_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_center_2d: batch_center_2d, input_fb_info: batch_fb_info, input_is_training: False})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                _, \
                loss,\
                heatmaps_loss,\
                fb_loss,\
                hm_acc,\
                fb_acc,\
                lr,\
                summary  = sess.run(
                        [
                         fb_model.train_op,
                         fb_model.total_loss,
                         fb_model.heatmaps_loss,
                         fb_model.fb_loss,
                         fb_model.heatmaps_acc,
                         fb_model.fb_acc,
                         fb_model.lr,
                         fb_model.merged_summary],

                        feed_dict={input_images: batch_images_np, input_center_2d: batch_center_2d, input_fb_info: batch_fb_info, input_is_training: True})
                train_log_writer.add_summary(summary, global_steps)

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}\nHeatmaps Loss: {:07f}\nFB Loss: {:07f}\n\n".format(global_steps, lr, loss, heatmaps_loss, fb_loss))
            print("Heatmap acc: {:07f}\nFB acc: {:07f}\n\n".format(hm_acc, fb_acc))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 20000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
