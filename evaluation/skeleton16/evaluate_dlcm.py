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
from utils.preprocess_utils import common as common_preprocess

from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton16 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorPCK

##################### Setting for training ######################
configs = mConfigs("../eval.conf", "dlcm_net_mpii")
################ Reseting  #################
configs.data_range = [0.1, 0.25, 0.5]
configs.loss_weights = []
configs.valid_scale = [0.0]
configs.pose_2d_scale = 4.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)
configs.is_use_bn = True

configs.extra_log_dir = "../eval_result/dlcm_mpii"

configs.nFeats = 256
configs.nModules = 1

configs.train_img_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/train/images"
configs.train_lbl_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/train/labels"
configs.valid_img_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/valid/images"
configs.valid_lbl_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/valid/labels"

################### Initialize the preprocessor ####################
configs.printConfig()
preprocessor = dlcm_preprocess.DLCMProcessor(skeleton=skeleton, img_size=configs.img_size, hm_size=configs.hm_size, sigma=1.0)

restore_model_epoch = 128
#################################################################

if __name__ == "__main__":
    # train_data_reader = data_reader.MPIIReader(img_dir=configs.train_img_dir, lbl_dir=configs.train_lbl_dir, is_shuffle=True, batch_size=configs.train_batch_size)
    valid_data_reader = data_reader.MPIIReader(img_dir=configs.valid_img_dir, lbl_dir=configs.valid_lbl_dir, is_shuffle=False, batch_size=1)

    # now test the classification
    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    dlcm_model = dlcm_net.mDLCMNet(skeleton=skeleton, img_size=configs.img_size, batch_size=input_batch_size, is_training=False, loss_weights=configs.loss_weights, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nModules=configs.nModules)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            dlcm_model.build_model(input_images)

        dlcm_model.build_evaluation(skeleton.flip_array)

        model_saver = tf.train.Saver()
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

        cur_valid_global_steps = 0

        valid_data_reader.reset()
        raw_pck_evaluator = mEvaluatorPCK(skeleton=skeleton, data_range=configs.data_range)
        mean_pck_evaluator = mEvaluatorPCK(skeleton=skeleton, data_range=configs.data_range)

        is_epoch_finished = False
        while not is_epoch_finished:
            # get the data path
            cur_batch, is_epoch_finished = valid_data_reader.get()

            batch_size = len(cur_batch)
            valid_batch_size = len(configs.valid_scale)

            batch_images_np = np.zeros([valid_batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_joints_2d_np = np.zeros([valid_batch_size, skeleton.n_joints, 2], dtype=np.float32)

            batch_images_flipped_np = np.zeros([valid_batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

            cur_img = cv2.imread(cur_batch[0][0])
            cur_lbl = np.load(cur_batch[0][1]).tolist()
            cur_joints_2d = cur_lbl["cropped_joints_2d"]

            multiscale_data = preprocessor.preprocess_multiscale(cur_img, cur_joints_2d, scale_range=configs.valid_scale, size=configs.img_size, pad_color=[128, 128, 128])

            for cur_batch in range(valid_batch_size):
                batch_images_np[cur_batch] = multiscale_data[cur_batch][0] / 255.0
                batch_images_flipped_np[cur_batch] = cv2.flip(batch_images_np[cur_batch], 1)

                batch_joints_2d_np[cur_batch] = np.round(multiscale_data[cur_batch][1])

                ############################# Visualize the result #############################
                # raw_imgs = np.concatenate([batch_images_np[cur_batch], batch_images_flipped_np[cur_batch]], axis=0)
                # raw_skt_img = display_utils.drawLines((255*batch_images_np[cur_batch]).astype(np.uint8), batch_joints_2d_np[cur_batch], indices=skeleton.bone_indices, color_table=skeleton.bone_colors)
                # flip_skt_img = display_utils.drawLines((255*batch_images_flipped_np[cur_batch]).astype(np.uint8), common_preprocess._flip_annot(batch_joints_2d_np[cur_batch], flip_array=skeleton.flip_array, size=configs.img_size), indices=skeleton.bone_indices, color_table=skeleton.bone_colors)

                # skt_imgs = np.concatenate([raw_skt_img, flip_skt_img], axis=0)

                # display_img = np.concatenate([raw_imgs, skt_imgs], axis=0)
                # cv2.imshow("display_img", display_img)
                # cv2.waitKey()
                ################################################################################

            raw_pd_2d,\
            mean_pd_2d = sess.run(
                    [
                    dlcm_model.raw_pd_2d,
                    dlcm_model.mean_pd_2d
                    ],
                    feed_dict={
                        input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0),
                        input_batch_size: 2*valid_batch_size
                    })

            raw_pck_evaluator.add(gt_2d=batch_joints_2d_np, pd_2d=raw_pd_2d, norm=configs.img_size / 10.0)
            mean_pck_evaluator.add(gt_2d=batch_joints_2d_np, pd_2d=mean_pd_2d, norm=configs.img_size / 10.0)

            print("Current frame {}:".format(cur_valid_global_steps))
            print("Raw data:")
            raw_pck_evaluator.printMean()
            print("\n")
            print("Mean data:")
            mean_pck_evaluator.printMean()
            print("\n\n")
            cur_valid_global_steps += 1

        raw_pck_evaluator.save(os.path.join(configs.extra_log_dir, "eval"), prefix="raw", epoch=restore_model_epoch)
        mean_pck_evaluator.save(os.path.join(configs.extra_log_dir, "eval"), prefix="mean", epoch=restore_model_epoch)
