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
from utils.preprocess_utils import pose_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton

##################### Setting for training ######################
configs = mConfigs("../train.conf", "pose_net_br")
configs.printConfig()
preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, with_br=True, bone_width=6, joint_ratio=6, bg_color=0.2)

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
#################################################################

if __name__ == "__main__":
    ################ Reseting  #################
    configs.loss_weight_heatmap = 1
    configs.loss_weight_pose = 100
    configs.joints_2d_scale = 4.0
    configs.pose_scale = 1000.0
    configs.is_use_bn = False
    configs.learning_rate = 2.5e-5
    configs.lr_decay_rate = 0.80
    configs.lr_decay_step = 10000

    ################### Initialize the data reader ####################
    train_range = np.load(configs.h36m_train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.h36m_valid_range_file)

    train_lbl_list = [configs.h36m_train_lbl_path_fn(i) for i in train_range]
    valid_lbl_list = [configs.h36m_valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = data_reader.get_data_iterator(train_lbl_list, batch_size=configs.train_batch_size, name="train_reader")
        valid_data_iter, valid_data_init_op = data_reader.get_data_iterator(valid_lbl_list, batch_size=configs.valid_batch_size, name="valid_reader", is_shuffle=False)

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_centers_hm = tf.placeholder(shape=[None, skeleton.n_joints, 2], dtype=tf.float32, name="input_centers_hm")
    input_poses = tf.placeholder(shape=[None, skeleton.n_joints, 3], dtype=tf.float32, name="input_poses")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    pose_model = pose_net.mPoseNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_pose=configs.loss_weight_pose, pose_scale=configs.pose_scale, is_use_bn=configs.is_use_bn)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            pose_model.build_model(input_images)
            input_heatmaps = pose_model.build_input_heatmaps(input_centers_hm, stddev=2.0, gaussian_coefficient=False)

        pose_model.build_loss(input_heatmaps=input_heatmaps, input_poses=input_poses, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

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
            batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
            batch_joints_3d_np = np.zeros([batch_size, skeleton.n_joints, 3], dtype=np.float32)

            # Generate the data batch
            label_path_for_show = [[] for i in range(max(configs.train_batch_size, configs.valid_batch_size))]

            for b in range(batch_size):
                label_path_for_show[b] = os.path.basename(cur_data_batch[b])

                cur_label = np.load(cur_data_batch[b]).tolist()

                cur_joints_3d = cur_label["joints_3d"].copy()[skeleton.h36m_selected_index]
                cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                cur_scale = cur_label["scale"]
                cur_center = cur_label["center"]
                cur_cam_mat = cur_label["cam_mat"]

                # use the bone relations
                cur_img, cur_joints_2d, cur_joints_3d = preprocessor.preprocess(joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, scale=cur_scale, center=cur_center, cam_mat=cur_cam_mat, is_training=not is_valid)
                # generate the heatmaps
                batch_images_np[b] = cur_img
                cur_joints_2d = cur_joints_2d / configs.joints_2d_scale
                cur_joints_3d = cur_joints_3d / configs.pose_scale

                batch_joints_2d_np[b] = cur_joints_2d.copy()
                batch_joints_3d_np[b] = cur_joints_3d.copy()

                cv2.imshow("img", cur_img)
                cv2.imshow("test", display_utils.drawLines((255.0 * cur_img).astype(np.uint8), cur_joints_2d * configs.joints_2d_scale, indices=skeleton.bone_indices))
                cv2.waitKey()

            acc_hm = 0
            acc_pose = 0

            if is_valid:
                acc_hm, \
                acc_pose, \
                loss, \
                heatmap_loss, \
                pose_loss, \
                lr, \
                summary  = sess.run(
                        [
                         pose_model.accuracy_hm,
                         pose_model.accuracy_pose,
                         pose_model.total_loss,
                         pose_model.heatmap_loss,
                         pose_model.pose_loss,
                         pose_model.lr,
                         pose_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers_hm: batch_joints_2d_np, input_poses: batch_joints_3d_np, input_is_training: False, input_batch_size: configs.valid_batch_size})
                valid_log_writer.add_summary(summary, global_steps)
            else:
                # if global_steps % write_log_iter == 0:
                _, \
                acc_hm, \
                acc_pose, \
                loss,\
                heatmap_loss, \
                pose_loss, \
                lr,\
                summary  = sess.run(
                        [
                         pose_model.train_op,
                         pose_model.accuracy_hm,
                         pose_model.accuracy_pose,
                         pose_model.total_loss,
                         pose_model.heatmap_loss,
                         pose_model.pose_loss,
                         pose_model.lr,
                         pose_model.merged_summary],
                        feed_dict={input_images: batch_images_np, input_centers_hm: batch_joints_2d_np, input_poses: batch_joints_3d_np, input_is_training: True, input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, global_steps)

            print("Train Iter:\n" if not is_valid else "Valid Iter:\n")
            print("Iteration: {:07d} \nlearning_rate: {:07f} \nTotal Loss : {:07f}\nHeatmap Loss: {:07f}\nPose Loss: {:07f}\n\n".format(global_steps, lr, loss, heatmap_loss, pose_loss))
            print("Heatmap Accuracy: {}\nPose Accuracy: {}".format(acc_hm, acc_pose))
            print((len(label_path_for_show) * "{}\n").format(*label_path_for_show))
            print("\n\n")

            if global_steps % 20000 == 0 and not is_valid:
                model_saver.save(sess=sess, save_path=configs.model_path, global_step=global_steps)

            if global_steps >= configs.train_iter and not is_valid:
                break
