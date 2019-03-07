import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import math

sys.path.append("../../")
from net import syn_net
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import syn_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorFB_BR

##################### Setting for training ######################

####################### Setting the training protocols ########################
training_protocol = [
        {"prefix": "syn_net_h36m", "extra_data_scale": 0, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "syn_net_mixed-5000", "extra_data_scale": 10, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "syn_net_mixed-11000", "extra_data_scale": 3, "mpii_range_file": "mpii_range.npy"},
        {"prefix": "syn_net_mixed-all", "extra_data_scale": 5, "mpii_range_file": "mpii_range_1.2w.npy"}
        ][3]
###############################################################################
configs = mConfigs("../eval.conf", training_protocol["prefix"])

################ Reseting  #################
configs.loss_weight_heatmap = 1.0
configs.loss_weight_br = 1.0
configs.loss_weight_fb = 2.0
configs.pose_2d_scale = 4.0
configs.is_use_bn = False
configs.extra_data_scale = training_protocol["extra_data_scale"]
configs.batch_size = 4

configs.n_epoches = 100
configs.learning_rate = 2.5e-4
configs.gamma = 0.1
configs.schedule = [30, 50, 80]
configs.valid_steps = 4

configs.nFeats = 256
configs.nModules = 2
configs.nStacks = 2
configs.nRegModules = 2

configs.extra_log_dir = "../eval_result/" + configs.prefix

### Use the smaller dataset to test and tune the hyper parameters
configs.train_or_valid = "valid"

configs.h36m_train_range_file = os.path.join(configs.range_file_dir, "train_range.npy")
configs.h36m_valid_range_file = os.path.join(configs.range_file_dir, "valid_range.npy")
configs.mpii_range_file = os.path.join(configs.range_file_dir, training_protocol["mpii_range_file"])
configs.lsp_range_file = os.path.join(configs.range_file_dir, "lsp_range.npy")

################### Initialize the data reader ####################
configs.printConfig()
preprocessor = syn_preprocess.SynProcessor(skeleton=skeleton, img_size=configs.img_size, bone_width=6, joint_ratio=6, overlap_threshold=6, bg_color=0.2, bone_status_threshold=100)

restore_model_epoch = 59

if __name__ == "__main__":
    ########################### Initialize the data list #############################
    if configs.train_or_valid == "train":
        valid_range = np.load(configs.h36m_train_range_file)
        valid_img_list = [configs.h36m_train_img_path_fn(i) for i in valid_range]
        valid_lbl_list = [configs.h36m_train_lbl_path_fn(i) for i in valid_range]
    else:
        valid_range = np.load(configs.h36m_valid_range_file)
        valid_img_list = [configs.h36m_valid_img_path_fn(i) for i in valid_range]
        valid_lbl_list = [configs.h36m_valid_lbl_path_fn(i) for i in valid_range]

    ###################################################################
    valid_data_reader = epoch_reader.EPOCHReader(img_path_list=valid_img_list, lbl_path_list=valid_lbl_list, is_shuffle=False, batch_size=configs.batch_size, name="Valid DataSet")

    # now test the classification
    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")

    syn_model = syn_net.mSynNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=configs.batch_size, is_training=False, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_fb=configs.loss_weight_fb, loss_weight_br=configs.loss_weight_br, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)
            syn_model.build_evaluation()

        print("Network built!")

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

        cur_valid_global_steps = 0

        valid_data_reader.reset()
        valid_relation_evaluator = mEvaluatorFB_BR(n_fb=skeleton.n_bones, n_br=(skeleton.n_bones-1) * skeleton.n_bones / 2)
        is_epoch_finished = False

        data_count = 0

        while not is_epoch_finished:
            cur_batch, is_epoch_finished = valid_data_reader.get()

            batch_size = len(cur_batch)
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
            batch_fb_np = np.zeros([batch_size, skeleton.n_bones], dtype=np.float32)
            batch_br_np = np.zeros([batch_size, (skeleton.n_bones-1) * skeleton.n_bones / 2], dtype=np.float32)

            for b in range(batch_size):
                assert(os.path.basename(cur_batch[b][0]).split(".")[0] == os.path.basename(cur_batch[b][1]).split(".")[0])

                cur_img = cv2.imread(cur_batch[b][0])
                cur_label = np.load(cur_batch[b][1]).tolist()

                if "joints_3d" in cur_label.keys():
                    ##!!!!! the joints_3d is the joints under the camera coordinates !!!!!##
                    ##!!!!! the joints_2d is the cropped ones !!!!!##
                    # the h36m datas
                    cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                    cur_joints_3d = cur_label["joints_3d"].copy()[skeleton.h36m_selected_index]
                    cur_scale = cur_label["scale"]
                    cur_center = cur_label["center"]
                    cur_cam_mat = cur_label["cam_mat"]

                    cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_h36m(img=cur_img, joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, scale=cur_scale, center=cur_center, cam_mat=cur_cam_mat, is_training=False)
                else:
                    # the mpii lsp datas
                    cur_joints_2d = cur_label["joints_2d"].copy()
                    cur_bone_status = cur_label["bone_status"].copy()
                    cur_bone_relations = cur_label["bone_relations"].copy()

                    cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_base(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=False)

                # generate the heatmaps
                batch_images_np[b] = cur_img
                cur_joints_2d = cur_joints_2d / configs.pose_2d_scale

                batch_joints_2d_np[b] = cur_joints_2d.copy()
                #### convert the bone_status and bone_relations to one-hot representation
                batch_fb_np[b] = cur_bone_status
                batch_br_np[b] = cur_bone_relations[np.triu_indices(skeleton.n_bones, k=1)] # only get the upper triangle

            pd_fb_result, \
            pd_br_result, \
            pd_br_belief = sess.run(
                    [
                     syn_model.pd_fb_result,
                     syn_model.pd_br_result,
                     syn_model.pd_br_belief
                    ],
                    feed_dict={input_images: batch_images_np,
                               input_batch_size: configs.batch_size})

            ################### Temporarily save the network outputs ###################
            for b in range(batch_size):
                np.save(os.path.join(configs.extra_log_dir, "{}_datas/{}.npy".format(configs.train_or_valid, data_count)), {"pd_fb": pd_fb_result[b], "pd_br": pd_br_result[b], "pd_br_belief": pd_br_belief[b], "gt_fb": batch_fb_np[b], "gt_br": batch_br_np[b]})
                data_count += 1
            ############################################################################

            valid_relation_evaluator.add(gt_fb=batch_fb_np, pd_fb=pd_fb_result, gt_br=batch_br_np, pd_br=pd_br_result)

            print("Validing | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(0, configs.n_epoches, *valid_data_reader.progress()))

            valid_relation_evaluator.printMean()
            print("\n\n")

            cur_valid_global_steps += 1

        valid_relation_evaluator.save(os.path.join(configs.extra_log_dir, "valid"), prefix="valid", epoch=0)
