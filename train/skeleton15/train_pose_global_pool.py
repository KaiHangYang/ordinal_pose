import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import math

sys.path.append("../../")
from net import pose_net
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import pose_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorPose3D

##################### Setting for training ######################
configs = mConfigs("../train.conf", "pose_net_aug_global")

################ Reseting  #################
configs.loss_weight_heatmap = 1
configs.loss_weight_pose = 100
configs.pose_2d_scale = 4.0
configs.pose_3d_scale = 1000.0

configs.is_use_bn = False

configs.n_epoches = 50
configs.learning_rate = 2.5e-5
configs.gamma = 0.1
configs.schedule = [5, 25]
configs.valid_steps = 0 # every training epoch valid the network

configs.nFeats = 256
configs.nModules = 3
configs.nRegModules = 2

configs.extra_log_dir = "../train_log/" + configs.prefix

################### Initialize the data reader ####################

configs.printConfig()
preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, with_br=True, with_fb=True, bone_width=6, joint_ratio=6, overlap_threshold=6, bg_color=0.2, pad_scale=0.4)

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_epoch = 10
#################################################################
def get_learning_rate(configs, epoch):
    decay = 0
    for i in range(len(configs.schedule)):
        if epoch >= configs.schedule[i]:
            decay = 1 + i
    return configs.learning_rate * math.pow(configs.gamma, decay)

if __name__ == "__main__":
    #################################Set the train and valid datas##################################
    training_data_dir = "/home/kaihang/DataSet_2/Ordinal/syn/train"
    validing_data_dir = "/home/kaihang/DataSet_2/Ordinal/syn/valid"

    training_angles = np.load(os.path.join(training_data_dir, "angles.npy"))
    training_bonelengths = np.load(os.path.join(training_data_dir, "bone_lengths.npy"))
    training_cammat = np.load(os.path.join(training_data_dir, "cam_mat.npy"))
    training_root_pos = np.load(os.path.join(training_data_dir, "root_pos.npy"))
    assert(len(training_bonelengths) == len(training_cammat) == len(training_root_pos))

    training_extra_sum = len(training_bonelengths)
    print("Training Extra Data: {}".format(training_extra_sum))

    train_lbl_list = np.arange(0, len(training_angles), 1)
    valid_lbl_list = [os.path.join(validing_data_dir, "{}.npy".format(i)) for i in range(len(os.listdir(validing_data_dir)))]


    #################### Just for test ###################
    # train_lbl_list = train_lbl_list[0:100]
    # valid_lbl_list = valid_lbl_list[0:100]
    ###################################################################
    train_data_reader = epoch_reader.EPOCHReader(img_path_list=None, lbl_path_list=train_lbl_list, is_shuffle=True, batch_size=configs.train_batch_size, name="Train DataSet")
    valid_data_reader = epoch_reader.EPOCHReader(img_path_list=None, lbl_path_list=valid_lbl_list, is_shuffle=False, batch_size=configs.valid_batch_size, name="Valid DataSet")

    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_centers_hm = tf.placeholder(shape=[None, skeleton.n_joints, 2], dtype=tf.float32, name="input_centers_hm")
    input_poses = tf.placeholder(shape=[None, skeleton.n_joints, 3], dtype=tf.float32, name="input_poses")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")
    input_lr = tf.placeholder(shape=[], dtype=tf.float32, name="input_lr")

    pose_model = pose_net.mPoseNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_pose=configs.loss_weight_pose, pose_2d_scale=configs.pose_2d_scale, pose_3d_scale=configs.pose_3d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nRegModules=configs.nRegModules, nModules=configs.nModules)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            pose_model.build_model(input_images, global_average_pooling=True)
            input_heatmaps = pose_model.build_input_heatmaps(input_centers_hm, stddev=2.0, gaussian_coefficient=False)

        pose_model.build_loss(input_heatmaps=input_heatmaps, input_poses=input_poses, lr=input_lr)

        print("Network built!")
        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)

        model_saver = tf.train.Saver(max_to_keep=configs.n_epoches)
        net_init = tf.global_variables_initializer()

        sess.run([net_init])
        # reload the model
        if restore_model_epoch is not None:
            if os.path.exists(configs.model_path_fn(restore_model_epoch)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(restore_model_epoch))
            else:
                print("The prev model is not existing!")
                quit()

        cur_train_global_steps = 0
        cur_valid_global_steps = 0

        cur_min_mpje = float("inf")

        for cur_epoch in range(0 if restore_model_epoch is None else restore_model_epoch, configs.n_epoches):
            cur_learning_rate = get_learning_rate(configs, cur_epoch)

            ############################ Train first #############################
            train_data_reader.reset()
            is_epoch_finished = False
            train_pose3d_evaluator = mEvaluatorPose3D(nJoints=skeleton.n_joints)

            while not is_epoch_finished:
                cur_batch, is_epoch_finished = train_data_reader.get()

                batch_size = len(cur_batch)
                batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
                batch_joints_3d_np = np.zeros([batch_size, skeleton.n_joints, 3], dtype=np.float32)

                for b in range(batch_size):
                    ##### Here I random choose the cam_mat and root_pos and bone_lengths from the huamn3.6m training set #####
                    cur_angles = training_angles[cur_batch[b]].copy()
                    cur_cam_mat = training_cammat[int(np.random.uniform(0, training_extra_sum))][0:3, 0:3].copy()
                    cur_bonelengths = training_bonelengths[int(np.random.uniform(0, training_extra_sum))].copy()
                    cur_root_pos = training_root_pos[int(np.random.uniform(0, training_extra_sum))].copy()

                    # use the bone relations
                    cur_img, cur_joints_2d, cur_joints_3d = preprocessor.preprocess(angles=cur_angles, bone_lengths=cur_bonelengths, root_pos=cur_root_pos, cam_mat=cur_cam_mat, is_training=True)
                    # generate the heatmaps
                    batch_images_np[b] = cur_img

                    cur_joints_2d = cur_joints_2d / configs.pose_2d_scale
                    cur_joints_3d = cur_joints_3d / configs.pose_3d_scale

                    batch_joints_2d_np[b] = cur_joints_2d.copy()
                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                    ########################### Visualize the datas ###########################
                    # cv2.imshow("synimg", cur_img.copy())
                    # cv2.imshow("synimg_skeleton", display_utils.drawLines(cur_img.copy(), batch_joints_2d_np[b], indices=skeleton.bone_indices, color_table=skeleton.bone_colors))
                    # cv2.waitKey()
                    ###########################################################################

                _, \
                pd_poses, \
                acc_hm, \
                acc_pose, \
                total_loss,\
                heatmap_loss, \
                pose_loss, \
                lr,\
                summary  = sess.run(
                        [
                         pose_model.train_op,
                         pose_model.pd_poses,
                         pose_model.accuracy_hm,
                         pose_model.accuracy_pose,
                         pose_model.total_loss,
                         pose_model.heatmap_loss,
                         pose_model.pose_loss,
                         pose_model.lr,
                         pose_model.merged_summary],
                        feed_dict={
                                   input_images: batch_images_np,
                                   input_centers_hm: batch_joints_2d_np,
                                   input_poses: batch_joints_3d_np,
                                   input_is_training: True,
                                   input_batch_size: configs.train_batch_size,
                                   input_lr:cur_learning_rate
                                  })

                train_log_writer.add_summary(summary, cur_train_global_steps)
                train_pose3d_evaluator.add(gt_coords=batch_joints_3d_np * configs.pose_3d_scale, pd_coords=pd_poses)

                print("Training | Epoch: {:05d}/{:05d}. Iteration: {:08d}/{:08d}".format(cur_epoch, configs.n_epoches, *train_data_reader.progress()))
                print("Learning_rate: {:07f}".format(lr))
                print("Heatmap pixel error: {}".format(acc_hm))
                print("Pose error: {}".format(acc_pose))
                print("Total loss: {:.08f}".format(total_loss))
                print("Heatmap loss: {:.08f}".format(heatmap_loss))
                print("Pose loss: {:.08f}".format(pose_loss))
                train_pose3d_evaluator.printMean()
                print("\n\n")
                cur_train_global_steps += 1

            train_pose3d_evaluator.save(os.path.join(configs.extra_log_dir, "train"), prefix="train", epoch=cur_epoch)

            ############################ Next Evaluate #############################
            valid_data_reader.reset()
            is_epoch_finished = False
            valid_pose3d_evaluator = mEvaluatorPose3D(nJoints=skeleton.n_joints)

            while not is_epoch_finished:
                cur_batch, is_epoch_finished = valid_data_reader.get()

                batch_size = len(cur_batch)
                batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
                batch_joints_3d_np = np.zeros([batch_size, skeleton.n_joints, 3], dtype=np.float32)

                for b in range(batch_size):
                    cur_label = np.load(cur_batch[b]).tolist()
                    cur_angles = cur_label["angles"].copy()
                    cur_bonelengths = cur_label["bone_lengths"].copy()
                    cur_root_pos = cur_label["root_pos"].copy()
                    cur_cam_mat = cur_label["cam_mat"][0:3, 0:3].copy()
                    # use the bone relations
                    cur_img, cur_joints_2d, cur_joints_3d = preprocessor.preprocess(angles=cur_angles, bone_lengths=cur_bonelengths, root_pos=cur_root_pos, cam_mat=cur_cam_mat, is_training=False)
                    # generate the heatmaps
                    batch_images_np[b] = cur_img

                    cur_joints_2d = cur_joints_2d / configs.pose_2d_scale
                    cur_joints_3d = cur_joints_3d / configs.pose_3d_scale

                    batch_joints_2d_np[b] = cur_joints_2d.copy()
                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                    ########################### Visualize the datas ###########################
                    # cv2.imshow("synimg", cur_img.copy())
                    # cv2.imshow("synimg_skeleton", display_utils.drawLines(cur_img.copy(), batch_joints_2d_np[b], indices=skeleton.bone_indices, color_table=skeleton.bone_colors))
                    # cv2.waitKey()
                    ###########################################################################

                pd_poses, \
                acc_hm, \
                acc_pose, \
                total_loss,\
                heatmap_loss, \
                pose_loss, \
                lr,\
                summary  = sess.run(
                        [
                         pose_model.pd_poses,
                         pose_model.accuracy_hm,
                         pose_model.accuracy_pose,
                         pose_model.total_loss,
                         pose_model.heatmap_loss,
                         pose_model.pose_loss,
                         pose_model.lr,
                         pose_model.merged_summary],
                        feed_dict={
                                   input_images: batch_images_np,
                                   input_centers_hm: batch_joints_2d_np,
                                   input_poses: batch_joints_3d_np,
                                   input_is_training: False,
                                   input_batch_size: configs.valid_batch_size,
                                   input_lr:cur_learning_rate
                                  })

                valid_log_writer.add_summary(summary, cur_valid_global_steps)
                valid_pose3d_evaluator.add(gt_coords=batch_joints_3d_np * configs.pose_3d_scale, pd_coords=pd_poses)

                print("Validing | Epoch: {:05d}/{:05d}. Iteration: {:08d}/{:08d}".format(cur_epoch, configs.n_epoches, *valid_data_reader.progress()))
                print("Learning_rate: {:07f}".format(lr))
                print("Heatmap pixel error: {}".format(acc_hm))
                print("Pose error: {}".format(acc_pose))
                print("Total loss: {:.08f}".format(total_loss))
                print("Heatmap loss: {:.08f}".format(heatmap_loss))
                print("Pose loss: {:.08f}".format(pose_loss))
                valid_pose3d_evaluator.printMean()
                print("\n\n")
                cur_valid_global_steps += 1

            valid_pose3d_evaluator.save(os.path.join(configs.extra_log_dir, "valid"), prefix="valid", epoch=cur_epoch)

            #################### Save the models #####################
            cur_mpje = valid_pose3d_evaluator.mean()
            if cur_min_mpje > cur_mpje:
                cur_min_mpje = cur_mpje
                with open(os.path.join(configs.model_dir, "best_model.txt"), "w") as f:
                    f.write("{}".format(cur_epoch))
            model_saver.save(sess=sess, save_path=configs.model_path, global_step=cur_epoch)
