import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import math

sys.path.append("../../")
from net import lin_net
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import pose_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorPose3D

##################### Setting for training ######################
configs = mConfigs("../train.conf", "pose_lin")

################ Reseting  #################
configs.pose_2d_scale = 4
configs.pose_3d_scale = 1000.0

configs.is_use_bn = False

configs.n_epoches = 200
configs.learning_rate = 2.5e-4
configs.gamma = 0.1
configs.schedule = [10, 20]

configs.extra_log_dir = "../train_log/" + configs.prefix

################### Initialize the data reader ####################

configs.printConfig()

preprocessor = pose_preprocess.PoseProcessor(skeleton=skeleton, img_size=configs.img_size, with_br=True, with_fb=True, bone_width=6, joint_ratio=6, overlap_threshold=6, bg_color=0.2, pad_scale=0.4, aug_bone_status=True, bone_status_threshold=120)

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_epoch = 70
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

    input_joints_2d = tf.placeholder(shape=[None, skeleton.n_joints, 2], dtype=tf.float32, name="input_joints_2d")
    input_bone_status = tf.placeholder(shape=[None, skeleton.n_bones, 3], dtype=tf.float32, name="input_bone_status")
    input_bone_relation = tf.placeholder(shape=[None, skeleton.n_bones * (skeleton.n_bones - 1) / 2, 3], dtype=tf.float32, name="input_bone_relation")
    input_arr = tf.concat([tf.layers.flatten(input_joints_2d), tf.layers.flatten(input_bone_status), tf.layers.flatten(input_bone_relation)], axis=1, name="input_all")

    input_poses = tf.placeholder(shape=[None, skeleton.n_joints, 3], dtype=tf.float32, name="input_poses")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")
    input_lr = tf.placeholder(shape=[], dtype=tf.float32, name="input_lr")

    lin_model = lin_net.mLinNet(nJoints=skeleton.n_joints, is_training=input_is_training, batch_size=input_batch_size, pose_3d_scale=configs.pose_3d_scale)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            lin_model.build_model(input_arr)

        lin_model.build_loss(input_poses=input_poses, lr=input_lr)

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
                batch_bone_status_np = np.zeros([batch_size, skeleton.n_bones, 3], dtype=np.float32)
                batch_bone_relation_np = np.zeros([batch_size, (skeleton.n_bones - 1) * skeleton.n_bones / 2, 3], dtype=np.float32)
                batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)

                batch_joints_3d_np = np.zeros([batch_size, skeleton.n_joints, 3], dtype=np.float32)

                for b in range(batch_size):
                    ##### Here I random choose the cam_mat and root_pos and bone_lengths from the huamn3.6m training set #####
                    cur_angles = training_angles[cur_batch[b]].copy()
                    cur_cam_mat = training_cammat[int(np.random.uniform(0, training_extra_sum))][0:3, 0:3].copy()
                    cur_bonelengths = training_bonelengths[int(np.random.uniform(0, training_extra_sum))].copy()
                    cur_root_pos = training_root_pos[int(np.random.uniform(0, training_extra_sum))].copy()

                    # use the bone relations
                    cur_joints_2d, cur_joints_3d, cur_bone_status, cur_bone_relation = preprocessor.preprocess_vec(angles=cur_angles, bone_lengths=cur_bonelengths, root_pos=cur_root_pos, cam_mat=cur_cam_mat, is_training=True)

                    cur_joints_2d = np.round(cur_joints_2d / configs.pose_2d_scale)
                    cur_joints_3d = cur_joints_3d / configs.pose_3d_scale

                    batch_joints_2d_np[b] = cur_joints_2d.copy()
                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                    batch_bone_status_np[b] = np.eye(3)[cur_bone_status]
                    batch_bone_relation_np[b] = np.eye(3)[cur_bone_relation[np.triu_indices(skeleton.n_bones, k=1)]]

                _, \
                pd_poses, \
                acc_pose, \
                total_loss,\
                pose_loss, \
                lr,\
                summary  = sess.run(
                        [
                         lin_model.train_op,
                         lin_model.pd_poses,
                         lin_model.accuracy_pose,
                         lin_model.total_loss,
                         lin_model.pose_loss,
                         lin_model.lr,
                         lin_model.merged_summary],
                        feed_dict={
                                   input_joints_2d: batch_joints_2d_np,
                                   input_bone_status: batch_bone_status_np,
                                   input_bone_relation: batch_bone_relation_np,
                                   input_poses: batch_joints_3d_np,
                                   input_is_training: True,
                                   input_batch_size: configs.train_batch_size,
                                   input_lr:cur_learning_rate
                                  })

                train_log_writer.add_summary(summary, cur_train_global_steps)
                train_pose3d_evaluator.add(gt_coords=batch_joints_3d_np * configs.pose_3d_scale, pd_coords=pd_poses)

                print("Training | Epoch: {:05d}/{:05d}. Iteration: {:08d}/{:08d}".format(cur_epoch, configs.n_epoches, *train_data_reader.progress()))
                print("Learning_rate: {:07f}".format(lr))
                print("Pose error: {}".format(acc_pose))
                print("Total loss: {:.08f}".format(total_loss))
                for idx, cur_pose_loss in enumerate(pose_loss):
                    print("Pose loss {}: {:.08f}".format(idx, cur_pose_loss))

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
                batch_bone_status_np = np.zeros([batch_size, skeleton.n_bones, 3], dtype=np.float32)
                batch_bone_relation_np = np.zeros([batch_size, (skeleton.n_bones - 1) * skeleton.n_bones / 2, 3], dtype=np.float32)
                batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)

                batch_joints_3d_np = np.zeros([batch_size, skeleton.n_joints, 3], dtype=np.float32)

                for b in range(batch_size):
                    cur_label = np.load(cur_batch[b]).tolist()
                    cur_angles = cur_label["angles"].copy()
                    cur_bonelengths = cur_label["bone_lengths"].copy()
                    cur_root_pos = cur_label["root_pos"].copy()
                    cur_cam_mat = cur_label["cam_mat"][0:3, 0:3].copy()

                    # use the bone relations
                    cur_joints_2d, cur_joints_3d, cur_bone_status, cur_bone_relation = preprocessor.preprocess_vec(angles=cur_angles, bone_lengths=cur_bonelengths, root_pos=cur_root_pos, cam_mat=cur_cam_mat, is_training=False)

                    cur_joints_2d = np.round(cur_joints_2d / configs.pose_2d_scale)
                    cur_joints_3d = cur_joints_3d / configs.pose_3d_scale

                    batch_joints_2d_np[b] = cur_joints_2d.copy()
                    batch_joints_3d_np[b] = cur_joints_3d.copy()

                    batch_bone_status_np[b] = np.eye(3)[cur_bone_status]
                    batch_bone_relation_np[b] = np.eye(3)[cur_bone_relation[np.triu_indices(skeleton.n_bones, k=1)]]

                pd_poses, \
                acc_pose, \
                total_loss,\
                pose_loss, \
                lr,\
                summary  = sess.run(
                        [
                         lin_model.pd_poses,
                         lin_model.accuracy_pose,
                         lin_model.total_loss,
                         lin_model.pose_loss,
                         lin_model.lr,
                         lin_model.merged_summary],
                        feed_dict={
                                   input_joints_2d: batch_joints_2d_np,
                                   input_bone_status: batch_bone_status_np,
                                   input_bone_relation: batch_bone_relation_np,
                                   input_poses: batch_joints_3d_np,
                                   input_is_training: False,
                                   input_batch_size: configs.valid_batch_size,
                                   input_lr:cur_learning_rate
                                  })

                valid_log_writer.add_summary(summary, cur_valid_global_steps)
                valid_pose3d_evaluator.add(gt_coords=batch_joints_3d_np * configs.pose_3d_scale, pd_coords=pd_poses)

                print("Validing | Epoch: {:05d}/{:05d}. Iteration: {:08d}/{:08d}".format(cur_epoch, configs.n_epoches, *valid_data_reader.progress()))
                print("Learning_rate: {:07f}".format(lr))
                print("Pose error: {}".format(acc_pose))
                print("Total loss: {:.08f}".format(total_loss))
                for idx, cur_pose_loss in enumerate(pose_loss):
                    print("Pose loss {}: {:.08f}".format(idx, cur_pose_loss))
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
