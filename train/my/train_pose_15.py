import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import pose_net
from utils.dataread_utils import pose_reader as data_reader
from utils.preprocess_utils import pose_preprocess as preprocessor
from utils.visualize_utils import display_utils

##################### Setting for training ######################
import configs

# t means gt(0) or ord(1)
# ver means version
configs.parse_configs(t=0, ver=4)
configs.print_configs()

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
#################################################################

def recalculate_bone_status(joints_z, bones_indices):
    bone_status = []
    for cur_bone_idx in bones_indices:
        if np.abs(joints_z[cur_bone_idx[0]] - joints_z[cur_bone_idx[1]]) < 100:
            bone_status.append(0)
        elif joints_z[cur_bone_idx[1]] < joints_z[cur_bone_idx[0]]:
            bone_status.append(1)
        else:
            bone_status.append(2)
    return np.array(bone_status)

if __name__ == "__main__":
    ################ Reseting  #################
    configs.nJoints = 15
    CUR_JOINTS_SELECTED = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16])
    preprocessor.bones_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [7, 9],
        [9, 10],
        [10, 11],
        [7, 12],
        [12, 13],
        [13, 14]
        ])

    preprocessor.bone_colors = np.array([
        [1.000000, 1.000000, 0.000000],
        [0.492543, 0.000000, 0.000000],
        [0.000000, 1.000000, 0.000000],
        [1.000000, 0.349454, 0.000000],
        [0.499439, 0.558884, 1.000000],
        [0.000000, 0.362774, 0.000000],
        [0.500312, 0.000000, 0.624406],
        [0.000000, 1.000000, 1.000000],
        [1.000000, 0.499433, 0.611793],
        [1.000000, 0.800000, 1.000000],
        [0.000000, 0.502502, 0.611632],
        [0.200000, 0.700000, 0.300000],
        [0.700000, 0.300000, 0.100000],
        [0.300000, 0.200000, 0.800000]
    ])
    preprocessor.flip_array = np.array([[1, 4], [2, 5], [3, 6], [9, 12], [10, 13], [11, 14]])

    ################### Initialize the data reader ####################
    train_range = np.load(configs.train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.valid_range_file)

    train_lbl_list = [configs.train_lbl_path_fn(i) for i in train_range]
    valid_lbl_list = [configs.valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = data_reader.get_data_iterator(train_lbl_list, batch_size=configs.train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = data_reader.get_data_iterator(valid_lbl_list, batch_size=configs.valid_batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_centers_hm = tf.placeholder(shape=[None, configs.nJoints, 2], dtype=tf.float32, name="input_centers_hm")
    input_centers_vol = tf.placeholder(shape=[None, configs.nJoints, 3], dtype=tf.float32, name="input_centers_vol")
    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    pose_model = pose_net.mPoseNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_heatmap=5.0, loss_weight_volume=1.0, is_use_bn=False)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            pose_model.build_model(input_images)
            input_heatmaps = pose_model.build_input_heatmaps(input_centers_hm, stddev=2.0, gaussian_coefficient=False)
            input_volumes = pose_model.build_input_volumes(input_centers_vol, stddev=2.0, gaussian_coefficient=False)

        pose_model.build_loss(input_heatmaps=input_heatmaps, input_volumes=input_volumes, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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
            global_steps = sess.run(pose_model.global_steps)

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

            batch_size = len(cur_data_batch)
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_centers_np = np.zeros([batch_size, configs.nJoints, 3], dtype=np.float32)

            # Generate the data batch
            label_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]

            for b in range(batch_size):
                label_path_for_show[b] = os.path.basename(cur_data_batch[b])

                cur_label = np.load(cur_data_batch[b]).tolist()

                cur_joints_3d = cur_label["joints_3d"].copy()[CUR_JOINTS_SELECTED]
                cur_joints_2d = cur_label["joints_2d"].copy()[CUR_JOINTS_SELECTED]
                cur_joints_zidx = (cur_label["joints_zidx"] - 1).copy()[CUR_JOINTS_SELECTED] # cause lua is from 1 to n not 0 to n-1

                # print(preprocessor.bones_indices)

                cur_bone_status = recalculate_bone_status(cur_joints_3d[:, 2], preprocessor.bones_indices)
                # cur_bone_status = cur_label["bone_status"].copy()
                # cur_bone_relations = cur_label["bone_relations"].copy()
                cur_bone_relations = None

                cur_img, cur_joints_2d, cur_joints_zidx = preprocessor.preprocess(joints_2d=cur_joints_2d, joints_zidx=cur_joints_zidx, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=not is_valid, bone_width=6, joint_ratio=6, bg_color=0.2, num_of_joints=configs.nJoints)
                # generate the heatmaps and volumes
                batch_images_np[b] = cur_img
                cur_joints_2d = np.round(cur_joints_2d / configs.joints_2d_scale)
                batch_centers_np[b] = np.concatenate([cur_joints_2d, cur_joints_zidx[:, np.newaxis]], axis=1)

                # cv2.imshow("img", cur_img)
                # print(preprocessor.bones_indices)
                # cv2.imshow("test", display_utils.drawLines((255.0 * cur_img).astype(np.uint8), cur_joints_2d * 4, indices=preprocessor.bones_indices))
                # print(batch_centers_np[b])
                # cv2.waitKey()

            acc_hm = 0
            acc_vol = 0

            if is_valid:
                acc_hm, \
                acc_vol, \
                loss, \
                heatmap_loss, \
                volume_loss, \
                lr, \
                summary  = sess.run(
                        [
                         pose_model.accuracy_hm,
                         pose_model.accuracy_vol,
                         pose_model.total_loss,
                         pose_model.heatmap_loss,
                         pose_model.volume_loss,
                         pose_model.lr,
                         pose_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers_hm: batch_centers_np[:, :, 0:2], input_centers_vol: batch_centers_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                # if global_steps % write_log_iter == 0:
                _, \
                acc_hm, \
                acc_vol, \
                loss,\
                heatmap_loss, \
                volume_loss, \
                lr,\
                summary  = sess.run(
                        [
                         pose_model.train_op,
                         pose_model.accuracy_hm,
                         pose_model.accuracy_vol,
                         pose_model.total_loss,
                         pose_model.heatmap_loss,
                         pose_model.volume_loss,
                         pose_model.lr,
                         pose_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers_hm: batch_centers_np[:, :, 0:2], input_centers_vol: batch_centers_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, global_steps)
                # else:
                    # _, \
                    # loss,\
                    # heatmap_loss, \
                    # volume_loss, \
                    # lr = sess.run(
                            # [
                             # pose_model.train_op,
                             # pose_model.total_loss,
                             # pose_model.heatmap_loss,
                             # pose_model.volume_loss,
                             # pose_model.lr,
                             # ],
                            # feed_dict={input_images: batch_images_np, input_centers_hm: batch_centers_np[:, :, 0:2], input_centers_vol: batch_centers_np, input_is_training: True, input_batch_size: configs.train_batch_size})

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}\nHeatmap Loss: {:07f}\nVolume Loss: {:07f}\n\n".format(global_steps, lr, loss, heatmap_loss, volume_loss))
            print("Heatmap Accuracy: {}\nVolume Accuracy: {}".format(acc_hm, acc_vol))
            print((len(label_path_for_show) * "{}\n").format(*label_path_for_show))
            print("\n\n")

            if global_steps % 20000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
