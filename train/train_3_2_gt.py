import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import ordinal_3_2
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.preprocess_utils import ordinal_3_2 as preprocessor
from utils.visualize_utils import display_utils

##################### Setting for training ######################

nJoints = 17
train_batch_size = 4
valid_batch_size = 1
img_size = 256

######################## To modify #############################
trash_log = ""

train_log_dir = "../"+trash_log+"logs/train/3_2_gt/train"
valid_log_dir = "../"+trash_log+"logs/train/3_2_gt/valid"
model_dir = "../models/3_2_gt/"
model_name = "ordinal_3_2_gt"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

is_restore = False
restore_model_path = "../models/3_2_gt/ordinal_3_2_gt-300000"
coords_scale = 1.0
################################################################


############### according to hourglass-tensorflow
valid_iter = 3
train_iter = 300000
learning_rate = 2.5e-4
lr_decay_rate = 0.96 # 0.96
lr_decay_step = 2000

train_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images/{}.jpg".format(x)
train_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/labels/{}.npy".format(x)

valid_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/images/{}.jpg".format(x)
valid_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/labels/{}.npy".format(x)

training_data_range_file = "./train_range/sec_3/train_range.npy"
validing_data_range_file = "./train_range/sec_3/valid_range.npy"

#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    ############################ range section 3 ##########################
    train_range = np.load(training_data_range_file)
    valid_range = np.load(validing_data_range_file)
    train_img_list = [train_img_path(i) for i in train_range]
    train_lbl_list = [train_lbl_path(i) for i in train_range]

    valid_img_list = [valid_img_path(i) for i in valid_range]
    valid_lbl_list = [valid_lbl_path(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = ordinal_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = ordinal_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=valid_batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, img_size, img_size, 3], dtype=tf.float32)
    input_coords = tf.placeholder(shape=[None, nJoints, 3], dtype=tf.float32)
    input_is_training = tf.placeholder(shape=[], dtype=tf.bool)
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32)

    ordinal_model = ordinal_3_2.mOrdinal_3_2(nJoints=nJoints, img_size=img_size, batch_size=input_batch_size, is_training=input_is_training, coords_scale=coords_scale)

    with tf.Session() as sess:

        ordinal_model.build_model(input_images)
        ordinal_model.build_loss_gt(input_coords, lr=learning_rate, lr_decay_step=lr_decay_step, lr_decay_rate=lr_decay_rate)

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

            batch_size = len(cur_data_batch[0])
            batch_images_np = np.zeros([batch_size, img_size, img_size, 3], dtype=np.float32)
            batch_coords_np = np.zeros([batch_size, nJoints, 3], dtype=np.float32)

            # Generate the data batch
            img_path_for_show = [[] for i in range(max(train_batch_size, valid_batch_size))]
            label_path_for_show = [[] for i in range(max(train_batch_size, valid_batch_size))]

            for b in range(batch_size):
                img_path_for_show[b] = os.path.basename(cur_data_batch[0][b])
                label_path_for_show[b] = os.path.basename(cur_data_batch[1][b])

                cur_img = cv2.imread(cur_data_batch[0][b])
                cur_label = np.load(cur_data_batch[1][b]).tolist()

                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"]], axis=1)
                cur_img, cur_joints = preprocessor.preprocess(cur_img, cur_joints)

                cur_joints_3d = cur_joints[:, 2:5]
                batch_coords_np[b] = (cur_joints_3d - cur_joints_3d[0]) / coords_scale# related to the root
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
                acc, \
                lr, \
                summary  = sess.run(
                        [ordinal_model.loss,
                         ordinal_model.accuracy,
                         ordinal_model.lr,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_coords: batch_coords_np, input_is_training: False, input_batch_size: valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                _,\
                loss,\
                acc, \
                lr,\
                summary  = sess.run(
                        [ordinal_model.train_op,
                         ordinal_model.loss,
                         ordinal_model.accuracy,
                         ordinal_model.lr,
                         ordinal_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_coords: batch_coords_np, input_is_training: True, input_batch_size: train_batch_size})
                train_log_writer.add_summary(summary, global_steps)

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nLoss : {:07f}\nCoords accuracy: {:07f}\n\n".format(global_steps, lr, loss, acc))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            print("\n\n")

            if global_steps % 10000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=os.path.join(model_dir, model_name), global_step=global_steps)

            if global_steps >= train_iter and not is_valid:
                break
