import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import math

sys.path.append("../../")
from net import dlcm_net
from utils.dataread_utils import mpii_reader as data_reader
from utils.preprocess_utils import dlcm_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton16 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorPCK

##################### Setting for training ######################
configs = mConfigs("../train.conf", "dlcm_net_mpii")
################ Reseting  #################
configs.loss_weights = [5.0, 1.0, 1.0]
configs.pose_2d_scale = 4.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)
configs.is_use_bn = False
configs.n_epoches = 250

configs.data_range = [0.1, 0.25, 0.5]
configs.extra_log_dir = "../train_log/dlcm"
configs.schedule = [180, 225] # when at 180 epoch and 225 epoch, the learning rate is scaled by gamma
configs.gamma = 0.1
configs.learning_rate = 2.5e-4

configs.nFeats = 256
configs.nModules = 1
configs.valid_step = 4 # every 4 train epoch, valid once

configs.train_img_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/train/images"
configs.train_lbl_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/train/labels"
configs.valid_img_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/valid/images"
configs.valid_lbl_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/valid/labels"

################### Initialize the preprocessor ####################

configs.printConfig()
preprocessor = dlcm_preprocess.DLCMProcessor(skeleton=skeleton, img_size=configs.img_size, hm_size=configs.hm_size, sigma=1.0)

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_iteration = None
#################################################################

def get_learning_rate(configs, epoch):
    decay = 0
    for i in range(len(configs.schedule)):
        if epoch >= configs.schedule[i]:
            decay = 1 + i
    return configs.learning_rate * math.pow(configs.gamma, decay)


if __name__ == "__main__":
    train_data_reader = data_reader.MPIIReader(img_dir=configs.train_img_dir, lbl_dir=configs.train_lbl_dir, is_shuffle=True, batch_size=configs.train_batch_size)
    valid_data_reader = data_reader.MPIIReader(img_dir=configs.valid_img_dir, lbl_dir=configs.valid_lbl_dir, is_shuffle=False, batch_size=configs.valid_batch_size)

    # now test the classification
    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_heatmaps_level_0 = tf.placeholder(shape=[None, configs.hm_size, configs.hm_size, skeleton.level_nparts[0]], dtype=tf.float32, name="heatmaps_level_0")
    input_heatmaps_level_1 = tf.placeholder(shape=[None, configs.hm_size, configs.hm_size, skeleton.level_nparts[1]], dtype=tf.float32, name="heatmaps_level_1")
    input_heatmaps_level_2 = tf.placeholder(shape=[None, configs.hm_size, configs.hm_size, skeleton.level_nparts[2]], dtype=tf.float32, name="heatmaps_level_2")
    input_maps = [input_heatmaps_level_0, input_heatmaps_level_1, input_heatmaps_level_2]

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")
    input_lr = tf.placeholder(shape=[], dtype=tf.float32, name="input_learning_rate")

    dlcm_model = dlcm_net.mDLCMNet(skeleton=skeleton, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weights=configs.loss_weights, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nModules=configs.nModules)

    train_valid_counter = my_utils.mTrainValidCounter(train_steps=configs.valid_step, valid_steps=1)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            dlcm_model.build_model(input_images)

        dlcm_model.build_loss(input_maps=input_maps, lr=input_lr)

        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)
        print("Network built!")

        model_saver = tf.train.Saver(max_to_keep=10)
        net_init = tf.global_variables_initializer()

        sess.run([net_init])
        # reload the model
        if restore_model_iteration is not None:
            if os.path.exists(configs.model_path_fn(restore_model_iteration)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(restore_model_iteration))
            else:
                print("The prev model is not existing!")
                quit()

        cur_train_global_steps = 0
        cur_valid_global_steps = 0

        cur_max_acc = 0

        for cur_epoch in range(configs.n_epoches):

            cur_learning_rate = get_learning_rate(configs, cur_epoch)

            train_pck_evaluator = mEvaluatorPCK(skeleton=skeleton, data_range=configs.data_range)
            train_data_reader.reset()
            ################### Train #################
            is_epoch_finished = False
            while not is_epoch_finished:
                # get the data path
                cur_batch, is_epoch_finished = train_data_reader.get()

                batch_size = len(cur_batch)

                batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
                batch_heatmaps_level_0 = np.zeros([batch_size, configs.hm_size, configs.hm_size, skeleton.level_nparts[0]], dtype=np.float32)
                batch_heatmaps_level_1 = np.zeros([batch_size, configs.hm_size, configs.hm_size, skeleton.level_nparts[1]], dtype=np.float32)
                batch_heatmaps_level_2 = np.zeros([batch_size, configs.hm_size, configs.hm_size, skeleton.level_nparts[2]], dtype=np.float32)

                for b in range(batch_size):

                    cur_img = cv2.imread(cur_batch[b][0])
                    cur_lbl = np.load(cur_batch[b][1]).tolist()
                    cur_joints_2d = cur_lbl["cropped_joints_2d"]

                    cur_img, cur_maps, cur_joints_2d = preprocessor.preprocess(img=cur_img, joints_2d=cur_joints_2d, is_training=True)

                    # generate the heatmaps
                    batch_joints_2d_np[b] = cur_joints_2d * configs.pose_2d_scale
                    batch_images_np[b] = cur_img
                    batch_heatmaps_level_0[b] = cur_maps[0]
                    batch_heatmaps_level_1[b] = cur_maps[1]
                    batch_heatmaps_level_2[b] = cur_maps[2]

                    ########## Visualize the datas ###########
                    # cv2.imshow("img", cur_img)
                    # cv2.imshow("test", display_utils.drawLines((255.0 * cur_img).astype(np.uint8), cur_joints_2d * configs.pose_2d_scale, indices=skeleton.bone_indices, color_table=skeleton.bone_colors * 255))

                    # hm_level_0 = np.concatenate(np.transpose(cur_maps[0], axes=[2, 0, 1]), axis=1)
                    # hm_level_1 = np.concatenate(np.transpose(cur_maps[1], axes=[2, 0, 1]), axis=1)
                    # hm_level_2 = np.concatenate(np.transpose(cur_maps[2], axes=[2, 0, 1]), axis=1)

                    # cv2.imshow("hm_0", hm_level_0)
                    # cv2.imshow("hm_1", hm_level_1)
                    # cv2.imshow("hm_2", hm_level_2)

                    # cv2.waitKey()
                    ##########################################

                _, \
                pd_2d, \
                acc_hm, \
                total_loss,\
                heatmaps_loss, \
                lr,\
                summary  = sess.run(
                        [
                         dlcm_model.train_op,
                         dlcm_model.pd_2d,
                         dlcm_model.heatmaps_acc,
                         dlcm_model.total_loss,
                         dlcm_model.losses,
                         dlcm_model.lr,
                         dlcm_model.merged_summary],
                        feed_dict={input_images: batch_images_np,
                                   input_heatmaps_level_0: batch_heatmaps_level_0,
                                   input_heatmaps_level_1: batch_heatmaps_level_1,
                                   input_heatmaps_level_2: batch_heatmaps_level_2,
                                   input_lr: cur_learning_rate,
                                   input_is_training: True,
                                   input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, cur_train_global_steps)
                train_pck_evaluator.add(gt_2d=np.round(batch_joints_2d_np), pd_2d=pd_2d, norm=configs.img_size / 10.0)

                print("Training | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(cur_epoch, configs.n_epoches, *train_data_reader.progress()))
                print("learning_rate: {:07f}".format(lr))
                print("Heatmap pixel error: {}".format(acc_hm))
                print("Total loss: {:.08f}".format(total_loss))
                for l_idx in range(len(heatmaps_loss)):
                    print("Heatmap loss level {}: {}".format(l_idx, heatmaps_loss[l_idx]))
                train_pck_evaluator.printMean()

                print("\n\n")
                cur_train_global_steps += 1

            train_pck_evaluator.save(os.path.join(configs.extra_log_dir, "train"), prefix="train", epoch=cur_epoch)

            train_valid_counter.next()
            ############## Evaluate ################

            if not train_valid_counter.is_training:
                valid_data_reader.reset()
                valid_pck_evaluator = mEvaluatorPCK(skeleton=skeleton, data_range=configs.data_range)

                is_epoch_finished = False
                while not is_epoch_finished:
                    # get the data path
                    cur_batch, is_epoch_finished = valid_data_reader.get()

                    batch_size = len(cur_batch)

                    batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                    batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)

                    batch_heatmaps_level_0 = np.zeros([batch_size, configs.hm_size, configs.hm_size, skeleton.level_nparts[0]], dtype=np.float32)
                    batch_heatmaps_level_1 = np.zeros([batch_size, configs.hm_size, configs.hm_size, skeleton.level_nparts[1]], dtype=np.float32)
                    batch_heatmaps_level_2 = np.zeros([batch_size, configs.hm_size, configs.hm_size, skeleton.level_nparts[2]], dtype=np.float32)

                    for b in range(batch_size):

                        cur_img = cv2.imread(cur_batch[b][0])
                        cur_lbl = np.load(cur_batch[b][1]).tolist()
                        cur_joints_2d = cur_lbl["cropped_joints_2d"]

                        cur_img, cur_maps, cur_joints_2d = preprocessor.preprocess(img=cur_img, joints_2d=cur_joints_2d, is_training=False)

                        # generate the heatmaps
                        batch_images_np[b] = cur_img
                        batch_joints_2d_np[b] = cur_joints_2d * configs.pose_2d_scale
                        batch_heatmaps_level_0[b] = cur_maps[0]
                        batch_heatmaps_level_1[b] = cur_maps[1]
                        batch_heatmaps_level_2[b] = cur_maps[2]

                    acc_hm, \
                    pd_2d, \
                    total_loss,\
                    heatmaps_loss, \
                    lr,\
                    summary  = sess.run(
                            [
                             dlcm_model.heatmaps_acc,
                             dlcm_model.pd_2d,
                             dlcm_model.total_loss,
                             dlcm_model.losses,
                             dlcm_model.lr,
                             dlcm_model.merged_summary],
                            feed_dict={input_images: batch_images_np,
                                       input_heatmaps_level_0: batch_heatmaps_level_0,
                                       input_heatmaps_level_1: batch_heatmaps_level_1,
                                       input_heatmaps_level_2: batch_heatmaps_level_2,
                                       input_lr: cur_learning_rate,
                                       input_is_training: False,
                                       input_batch_size: configs.valid_batch_size})

                    valid_log_writer.add_summary(summary, cur_valid_global_steps)

                    valid_pck_evaluator.add(gt_2d=np.round(batch_joints_2d_np), pd_2d=pd_2d, norm=configs.img_size / 10.0)

                    print("Validing | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(cur_epoch, configs.n_epoches, *valid_data_reader.progress()))
                    print("learning_rate: {:07f}".format(lr))
                    print("Heatmap pixel error: {}".format(acc_hm))
                    print("Total loss: {:.08f}".format(total_loss))
                    for l_idx in range(len(heatmaps_loss)):
                        print("Heatmap loss level {}: {}".format(l_idx, heatmaps_loss[l_idx]))
                    valid_pck_evaluator.printMean()
                    print("\n\n")
                    cur_valid_global_steps += 1

                valid_pck_evaluator.save(os.path.join(configs.extra_log_dir, "valid"), prefix="valid", epoch=cur_epoch)
                valid_score_mean, _ = valid_pck_evaluator.mean()

                if cur_max_acc < valid_score_mean[-1]:
                    #### Only save the higher score models
                    with open(os.path.join(configs.model_dir, "best_model.txt"), "w") as f:
                        f.write("{}".format(cur_epoch))

                    cur_max_acc = valid_score_mean[-1]
                    model_saver.save(sess=sess, save_path=configs.model_path, global_step=cur_epoch)
