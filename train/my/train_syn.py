import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import syn_net
from utils.dataread_utils import syn_reader
from utils.preprocess_utils import syn_preprocess
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
    train_lbl_list = [configs.train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = syn_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=configs.train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = syn_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=configs.valid_batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")

    input_sep_synmaps = tf.placeholder(shape=[None, configs.sep_syn_img_size, configs.sep_syn_img_size, 3*(configs.nJoints - 1)], dtype=tf.float32, name="input_sep_synmaps")
    input_synmap = tf.placeholder(shape=[None, configs.syn_img_size, configs.syn_img_size, 3], dtype=tf.float32, name="input_synmap")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    syn_model = syn_net.mSynNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_sep_synmaps=10.0, loss_weight_synmap=1.0)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)

        syn_model.build_loss(input_sep_synmaps=input_sep_synmaps, input_synmap=input_synmap, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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

            batch_synmap_np = np.zeros([batch_size, configs.syn_img_size, configs.syn_img_size, 3], dtype=np.float32)
            batch_sep_synmaps_np = np.zeros([batch_size, configs.sep_syn_img_size, configs.sep_syn_img_size, 3 * (configs.nJoints - 1)], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]
            label_path_for_show = [[] for i in range(len(img_path_for_show))]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[1][b])

                cur_img = cv2.imread(cur_data_batch[0][b])
                cur_label = np.load(cur_data_batch[1][b]).tolist()

                # the joints_2d in the label_syn is resize in [64, 64]

                cur_joints_2d = cur_label["joints_2d"].copy() * configs.joints_2d_scale
                cur_bone_status = cur_label["bone_status"].copy()
                cur_bone_order = cur_label["bone_order"].copy()

                cur_img, cur_joints_2d, cur_bone_status, cur_bone_order = syn_preprocess.preprocess(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_order=cur_bone_order, is_training=not is_valid)

                cur_joints_2d = cur_joints_2d

                # draw the synsetic imgs as the ground truth
                cur_synmap, cur_sep_synmaps = syn_preprocess.draw_syn_img(cur_joints_2d, cur_bone_status, cur_bone_order, size=configs.syn_img_size, sep_size=configs.sep_syn_img_size, bone_width=4, joint_ratio=4)
                batch_images_np[b] = cur_img

                # make them [0, 1]
                batch_sep_synmaps_np[b] = np.concatenate(cur_sep_synmaps, axis=2) / 255.0
                batch_synmap_np[b] = cur_synmap / 255.0

            if is_valid:
                loss, \
                sep_synmaps_loss, \
                synmap_loss, \
                lr, \
                summary  = sess.run(
                        [
                         syn_model.total_loss,
                         syn_model.sep_synmaps_loss,
                         syn_model.synmap_loss,
                         syn_model.lr,
                         syn_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_sep_synmaps: batch_sep_synmaps_np, input_synmap: batch_synmap_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                if global_steps % write_log_iter == 0:
                    _, \
                    loss,\
                    sep_synmaps_loss, \
                    synmap_loss, \
                    lr,\
                    summary  = sess.run(
                            [
                             syn_model.train_op,
                             syn_model.total_loss,
                             syn_model.sep_synmaps_loss,
                             syn_model.synmap_loss,
                             syn_model.lr,
                             syn_model.merged_summary],
                            feed_dict={input_images: batch_images_np, input_sep_synmaps: batch_sep_synmaps_np, input_synmap: batch_synmap_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                    train_log_writer.add_summary(summary, global_steps)
                else:
                    _, \
                    loss,\
                    sep_synmaps_loss, \
                    synmap_loss, \
                    lr = sess.run(
                            [
                             syn_model.train_op,
                             syn_model.total_loss,
                             syn_model.sep_synmaps_loss,
                             syn_model.synmap_loss,
                             syn_model.lr,
                             ],
                            feed_dict={input_images: batch_images_np, input_sep_synmaps: batch_sep_synmaps_np, input_synmap: batch_synmap_np, input_is_training: True, input_batch_size: configs.train_batch_size})

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}\nSep synmaps Loss: {:07f}\nSynmap Loss: {:07f}\n\n".format(global_steps, lr, loss, sep_synmaps_loss, synmap_loss))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 20000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
