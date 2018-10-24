import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import syn_net
from utils.dataread_utils import syn_reader
from utils.preprocess_utils import syn_preprocess as preprocessor
from utils.visualize_utils import display_utils

##################### Setting for training ######################
import configs

# t means gt(0) or ord(1)
# ver means version
configs.parse_configs(t=0, ver=1)
configs.print_configs()

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ####################
    train_range = np.load(configs.train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.valid_range_file)

    train_img_list = [configs.train_img_path_fn(i) for i in train_range]
    train_syn_img_list = [configs.train_syn_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.valid_img_path_fn(i) for i in valid_range]
    valid_syn_img_list = [configs.valid_syn_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = syn_reader.get_data_iterator(train_img_list, train_syn_img_list, train_lbl_list, batch_size=configs.train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = syn_reader.get_data_iterator(valid_img_list, valid_syn_img_list, valid_lbl_list, batch_size=configs.valid_batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_synmaps = tf.placeholder(shape=[None, configs.syn_img_size, configs.syn_img_size, 3], dtype=tf.float32, name="input_synmaps")
    input_centers_hm = tf.placeholder(shape=[None, configs.nJoints, 2], dtype=tf.float32, name="input_centers_hm")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    syn_model = syn_net.mSynNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_synmap=configs.loss_weight_synmap)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)
            input_heatmaps = syn_model.build_input_heatmaps(input_centers_hm, stddev=2.0, gaussian_coefficient=False)

        syn_model.build_loss_gt(input_heatmaps=input_heatmaps, input_synmaps=input_synmaps, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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
            global_steps = sess.run(syn_model.global_steps)

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
            batch_synmaps_np = np.zeros([batch_size, configs.syn_img_size, configs.syn_img_size, 3], dtype=np.float32)
            batch_centers_np = np.zeros([batch_size, configs.nJoints, 3], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]
            syn_img_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]
            label_path_for_show = [[] for i in range(len(img_path_for_show))]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                syn_img_path_for_show[b] = os.path.basename(cur_data_batch[1][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[2][b])

                cur_img = cv2.imread(cur_data_batch[0][b])
                cur_syn_img = cv2.imread(cur_data_batch[1][b])
                cur_label = np.load(cur_data_batch[2][b]).tolist()

                cur_joints = cur_label["joints_2d"].copy()

                cur_img, cur_syn_img, cur_joints = preprocessor.preprocess(cur_img, cur_syn_img, cur_joints)
                # generate the heatmaps and synmaps
                batch_images_np[b] = cur_img
                batch_synmaps_np[b] = cur_syn_img

                hm_joint_2d = np.round(cur_joints[:, 0:2] / configs.coords_2d_scale)
                batch_centers_np[b] = hm_joint_3d

            acc_hm = 0

            if is_valid:
                acc_hm, \
                loss, \
                heatmap_loss, \
                synmap_loss, \
                lr, \
                summary  = sess.run(
                        [
                         syn_model.accuracy_hm,
                         syn_model.total_loss,
                         syn_model.heatmap_loss,
                         syn_model.synmap_loss,
                         syn_model.lr,
                         syn_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers_hm: batch_centers_np[:, :, 0:2], input_synmaps: batch_synmaps_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                if global_steps % write_log_iter == 0:
                    _, \
                    acc_hm, \
                    loss,\
                    heatmap_loss, \
                    synmap_loss, \
                    lr,\
                    summary  = sess.run(
                            [
                             syn_model.train_op,
                             syn_model.accuracy_hm,
                             syn_model.total_loss,
                             syn_model.heatmap_loss,
                             syn_model.synmap_loss,
                             syn_model.lr,
                             syn_model.merged_summary],
                            feed_dict={input_images: batch_images_np, input_centers_hm: batch_centers_np[:, :, 0:2], input_synmaps: batch_synmaps_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                    train_log_writer.add_summary(summary, global_steps)
                else:
                    _, \
                    loss,\
                    heatmap_loss, \
                    synmap_loss, \
                    lr = sess.run(
                            [
                             syn_model.train_op,
                             syn_model.total_loss,
                             syn_model.heatmap_loss,
                             syn_model.synmap_loss,
                             syn_model.lr,
                             ],
                            feed_dict={input_images: batch_images_np, input_centers_hm: batch_centers_np[:, :, 0:2], input_synmaps: batch_synmaps_np, input_is_training: True, input_batch_size: configs.train_batch_size})

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}\nHeatmap Loss: {:07f}\nSynmap Loss: {:07f}\n\n".format(global_steps, lr, loss, heatmap_loss, synmap_loss))
            print("Heatmap Accuracy: {}".format(acc_hm))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, syn_img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 20000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
