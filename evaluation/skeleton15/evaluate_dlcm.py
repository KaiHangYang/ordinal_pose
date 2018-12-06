import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import dlcm_net
from utils.preprocess_utils import dlcm_preprocess
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton

configs = mConfigs("../eval.conf", "dlcm_net")
################ Reseting  #################
configs.loss_weights = [10.0, 1.0, 1.0]
configs.pose_2d_scale = 4.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)
configs.is_use_bn = True
configs.batch_size = 4

configs.learning_rate = 2.5e-4
configs.lr_decay_rate = 0.10
configs.lr_decay_step = 200000
configs.nFeats = 256
configs.nModules = 1

configs.range_file =  configs.h36m_valid_range_file
configs.img_path_fn = configs.h36m_valid_img_path_fn
configs.lbl_path_fn = configs.h36m_valid_lbl_path_fn
##################### Evaluation Configs ######################
configs.printConfig()
preprocessor = dlcm_preprocess.DLCMProcessor(skeleton=skeleton, img_size=configs.img_size, hm_size=configs.hm_size, sigma=1.0)

evaluation_models = [140000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################

    range_arr = np.load(configs.range_file)
    data_from = 0
    data_to = len(range_arr)
    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[2*configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    dlcm_model = dlcm_net.mDLCMNet(skeleton=skeleton, img_size=configs.img_size, batch_size=2*configs.batch_size, is_training=False, loss_weights=configs.loss_weights, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nModules=configs.nModules)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            dlcm_model.build_model(input_images)
            dlcm_model.build_evaluation(skeleton.flip_array)

        print("Network built!")

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])

        for cur_model_iterations in evaluation_models:
            raw_pckh_evaluator = evaluators.mEvaluatorPCKh(n_joints=skeleton.n_joints, head_indices=skeleton.head_indices, data_range=[0.10, 0.25, 0.5])
            mean_pckh_evaluator = evaluators.mEvaluatorPCKh(n_joints=skeleton.n_joints, head_indices=skeleton.head_indices, data_range=[0.10, 0.25, 0.5])

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)
            ################# Restore the model ################

            if os.path.exists(configs.model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(cur_model_iterations))
            else:
                print(configs.model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

                batch_joints_2d_np = np.zeros([configs.batch_size, skeleton.n_joints, 2], dtype=np.float32)

                img_path_for_show = []
                lbl_path_for_show = []

                for b in range(configs.batch_size):
                    img_path_for_show.append(os.path.basename(img_list[data_index.val]))
                    lbl_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_img = cv2.imread(img_list[data_index.val])
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    if "joints_3d" in cur_label.keys():
                        ##!!!!! the joints_3d is the joints under the camera coordinates !!!!!##
                        ##!!!!! the joints_2d is the cropped ones !!!!!##
                        # the h36m datas
                        cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                    else:
                        # the mpii lsp datas
                        cur_joints_2d = cur_label["joints_2d"].copy()

                    cur_img = cur_img / 255.0

                    batch_images_np[b] = cur_img.copy()
                    batch_images_flipped_np[b] = cv2.flip(cur_img, 1)

                    batch_joints_2d_np[b] = cur_joints_2d.copy()

                raw_pd_2d, \
                mean_pd_2d = sess.run([
                        dlcm_model.raw_pd_2d,
                        dlcm_model.mean_pd_2d
                        ],
                        feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                print((len(lbl_path_for_show) * "{}\n").format(*zip(img_path_for_show, lbl_path_for_show)))

                # ############# evaluate the coords recovered from the gt 2d and gt root depth
                raw_pckh_evaluator.add(gt_2d=batch_joints_2d_np, pd_2d=raw_pd_2d)
                mean_pckh_evaluator.add(gt_2d=batch_joints_2d_np, pd_2d=mean_pd_2d)

                print("Current evaluate: {}-{}".format(configs.prefix, cur_model_iterations))
                print("Raw 2D PCKh:")
                raw_pckh_evaluator.printMean()
                print("\n")

                print("Mean 2D PCKh:")
                mean_pckh_evaluator.printMean()
                print("\n")

            raw_pckh_evaluator.save("../eval_result/{}/raw_pckh_{}w.npy".format(configs.prefix, cur_model_iterations / 10000))
            mean_pckh_evaluator.save("../eval_result/{}/mean_pckh_{}w.npy".format(configs.prefix, cur_model_iterations / 10000))
