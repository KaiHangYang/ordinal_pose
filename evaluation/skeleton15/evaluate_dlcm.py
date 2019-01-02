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
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import dlcm_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorPCK
from utils.evaluate_utils.evaluators import mEvaluatorPose2D

##################### Setting for Evaluation ######################

####################### Setting the Evaluate protocols ########################
training_protocol = [
        {"prefix": "dlcm_h36m", "extra_data_scale": 0, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "dlcm_mixed-15000", "extra_data_scale": 5, "mpii_range_file": "mpii_range_1.2w.npy"}
        ][1]
###############################################################################
configs = mConfigs("../eval.conf", training_protocol["prefix"])
################ Reseting  #################
configs.loss_weights = [5.0, 1.0, 1.0]
configs.pose_2d_scale = 4.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)
configs.is_use_bn = True
configs.n_epoches = 150

configs.data_range = [0.1, 0.25, 0.5]
configs.extra_log_dir = "../eval_result/" + configs.prefix
configs.zero_debias_moving_mean = True

configs.nFeats = 256
configs.nModules = 1
configs.batch_size = 4

configs.h36m_train_range_file = os.path.join(configs.range_file_dir, "train_range.npy")
configs.h36m_valid_range_file = os.path.join(configs.range_file_dir, "valid_range.npy")
configs.mpii_range_file = os.path.join(configs.range_file_dir, training_protocol["mpii_range_file"])
configs.lsp_range_file = os.path.join(configs.range_file_dir, "lsp_range.npy")

################### Initialize the preprocessor ####################

configs.printConfig()
preprocessor = dlcm_preprocess.DLCMProcessor(skeleton=skeleton, img_size=configs.img_size, hm_size=configs.hm_size, sigma=1.0)

### h36m best 32
### mixed best 18
configs.eval_type = "train"
restore_model_epoch = 68

#################################################################

if __name__ == "__main__":
    ########################### Initialize the data list #############################
    train_range = np.load(configs.h36m_train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.h36m_valid_range_file)

    train_img_list = [configs.h36m_train_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.h36m_train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.h36m_valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.h36m_valid_lbl_path_fn(i) for i in valid_range]

    mpii_range = np.load(configs.mpii_range_file)
    lsp_range = np.load(configs.lsp_range_file)

    mpii_lsp_img_list = [configs.mpii_img_path_fn(i) for i in mpii_range] + [configs.lsp_img_path_fn(i) for i in lsp_range]
    mpii_lsp_lbl_list = [configs.mpii_lbl_path_fn(i) for i in mpii_range] + [configs.lsp_lbl_path_fn(i) for i in lsp_range]

    ############################## Selected the evaluate dataset ##############################
    if configs.eval_type == "train":
        img_list = train_img_list
        lbl_list = train_lbl_list
    elif configs.eval_type == "valid":
        img_list = valid_img_list
        lbl_list = valid_lbl_list
    else:
        print("eval_type not exists!")
        quit()
    ###################################################################
    data_reader = epoch_reader.EPOCHReader(img_path_list=img_list, lbl_path_list=lbl_list, is_shuffle=False, batch_size=configs.batch_size, name="Eval DataSet")

    input_images = tf.placeholder(shape=[configs.batch_size * 2, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    dlcm_model = dlcm_net.mDLCMNet(skeleton=skeleton, img_size=configs.img_size, batch_size=configs.batch_size * 2, is_training=False, loss_weights=configs.loss_weights, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nModules=configs.nModules, zero_debias_moving_mean=configs.zero_debias_moving_mean)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            dlcm_model.build_model(input_images)
            dlcm_model.build_evaluation(skeleton.flip_array)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()

        sess.run([net_init])
        # reload the model
        if os.path.exists(configs.model_path_fn(restore_model_epoch)+".index"):
            print("#######################Restored all weights ###########################")
            model_saver.restore(sess, configs.model_path_fn(restore_model_epoch))
        else:
            print("The prev model is not existing!")
            quit()

        data_reader.reset()

        raw_pck_evaluator = mEvaluatorPCK(skeleton=skeleton, data_range=configs.data_range)
        mean_pck_evaluator = mEvaluatorPCK(skeleton=skeleton, data_range=configs.data_range)

        raw_mpje_evaluator = mEvaluatorPose2D(nJoints=skeleton.n_joints)
        mean_mpje_evaluator = mEvaluatorPose2D(nJoints=skeleton.n_joints)

        data_count = 0
        is_epoch_finished = False
        while not is_epoch_finished:
            # get the data path
            cur_batch, is_epoch_finished = data_reader.get()

            batch_size = len(cur_batch)
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_images_flipped_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

            batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)

            for b in range(batch_size):

                cur_img = cv2.imread(cur_batch[b][0])
                cur_lbl = np.load(cur_batch[b][1]).tolist()

                if "joints_3d" in cur_lbl.keys():
                    cur_joints_2d = cur_lbl["joints_2d"].copy()[skeleton.h36m_selected_index]
                else:
                    cur_joints_2d = cur_lbl["joints_2d"].copy()

                # generate the heatmaps
                batch_images_np[b] = cur_img / 255.0
                batch_images_flipped_np[b] = cv2.flip(batch_images_np[b].copy(), 1)

                batch_joints_2d_np[b] = cur_joints_2d

            raw_pd_2d, \
            mean_pd_2d = sess.run(
                    [
                        dlcm_model.raw_pd_2d,
                        dlcm_model.mean_pd_2d
                     ],
                    feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

            for b in range(batch_size):
                np.save(os.path.join(configs.extra_log_dir, "datas/{}.npy".format(data_count)), {"mean_pd_2d": mean_pd_2d[b], "raw_pd_2d": raw_pd_2d[b], "gt_2d": batch_joints_2d_np[b]})
                data_count += 1

            raw_pck_evaluator.add(gt_2d=np.round(batch_joints_2d_np), pd_2d=raw_pd_2d, norm=configs.img_size / 10.0)
            mean_pck_evaluator.add(gt_2d=np.round(batch_joints_2d_np), pd_2d=mean_pd_2d, norm=configs.img_size / 10.0)

            raw_mpje_evaluator.add(gt_2d=np.round(batch_joints_2d_np), pd_2d=raw_pd_2d)
            mean_mpje_evaluator.add(gt_2d=np.round(batch_joints_2d_np), pd_2d=mean_pd_2d)

            print("Validing | Iteration: {:05d}/{:05d}".format(*data_reader.progress()))

            sys.stdout.write("Raw:\n")
            raw_pck_evaluator.printMean()
            raw_mpje_evaluator.printMean()

            sys.stdout.write("Mean\n")
            mean_pck_evaluator.printMean()
            mean_mpje_evaluator.printMean()

        raw_pck_evaluator.save(os.path.join(configs.extra_log_dir, configs.eval_type), prefix="raw_pck", epoch=0)
        mean_pck_evaluator.save(os.path.join(configs.extra_log_dir, configs.eval_type), prefix="mean_pck", epoch=0)

        raw_mpje_evaluator.save(os.path.join(configs.extra_log_dir, configs.eval_type), prefix="raw_mpje", epoch=0)
        mean_mpje_evaluator.save(os.path.join(configs.extra_log_dir, configs.eval_type), prefix="mean_mpje", epoch=0)
