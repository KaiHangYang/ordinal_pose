import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import fb_net
from utils.dataread_utils import syn_reader
from utils.preprocess_utils import fb_preprocess
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils

##################### Setting for training ######################
import configs

# t means gt(0) or ord(1)
# ver means version
configs.parse_configs(t=0, d=0)
configs.print_configs()

restore_model_iteration = 580000
#################################################################

if __name__ == "__main__":
    # assume the train_batch_size and valid_batch_size is 4
    batch_size = 1
    configs.nJoints = 13

    ################### Initialize the data reader ####################
    data_range = np.load(configs.range_file)
    lsp_range = np.load(configs.lsp_range_file)
    mpii_range = np.load(configs.mpii_range_file)

    img_list = [configs.img_path_fn(i) for i in data_range]
    lbl_list = [configs.lbl_path_fn(i) for i in data_range]

    lsp_img_list = [configs.lsp_img_path_fn(i) for i in lsp_range]
    lsp_lbl_list = [configs.lsp_lbl_path_fn(i) for i in lsp_range]

    mpii_img_list = [configs.mpii_img_path_fn(i) for i in mpii_range]
    mpii_lbl_list = [configs.mpii_lbl_path_fn(i) for i in mpii_range]
    ###################################################################
    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")

    fb_model = fb_net.mFBNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=batch_size, is_training=False, loss_weight_heatmaps=1.0, loss_weight_fb=1.0, is_use_bn=False)


    ####################################### Selected the dataset for visualization ##########################################
    cur_img_list = img_list
    cur_lbl_list = lbl_list

    data_index = my_utils.mRangeVariable(min_val=0, max_val=len(cur_img_list), initial_val=0)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            fb_model.build_model(input_images)
            fb_model.build_evaluation(eval_batch_size = batch_size)

        model_saver = tf.train.Saver(max_to_keep=70)
        # reload the model
        if os.path.exists(configs.fb_restore_model_path_fn(restore_model_iteration)+".index"):
            print("#######################Restored all weights ###########################")
            model_saver.restore(sess, configs.fb_restore_model_path_fn(restore_model_iteration))
        else:
            print("The prev model is not existing!")
            quit()

        while not data_index.isEnd():
            # assume the batch_size is 1
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            cur_img = cv2.imread(cur_img_list[data_index.val])
            batch_images_np[0] = cur_img / 255.0

            pd_heatmaps_0,\
            pd_heatmaps_1,\
            pd_fb_result,\
            pd_joints_2d = sess.run(
                    [
                        fb_model.heatmaps[0],
                        fb_model.heatmaps[1],
                        fb_model.pd_fb_result,
                        fb_model.pd_joints_2d
                    ],
                    feed_dict={input_images: batch_images_np})

            pd_joints_2d *= configs.joints_2d_scale
            ########### visualize the heatmaps ############
            all_hms = []
            for b in range(batch_size):
                b_all_hms = []
                for i in range(configs.nJoints):
                    cur_pd_1 = cv2.copyMakeBorder(pd_heatmaps_0[0, :, :, i], top=1, bottom=1, left=1, right=1, value=[1.0, 1.0, 1.0], borderType=cv2.BORDER_CONSTANT)
                    cur_pd_2 = cv2.copyMakeBorder(pd_heatmaps_1[0, :, :, i], top=1, bottom=1, left=0, right=1, value=[1.0, 1.0, 1.0], borderType=cv2.BORDER_CONSTANT)

                    tmp_hms_row = np.concatenate([cur_pd_1, cur_pd_2], axis=1)
                    b_all_hms.append(tmp_hms_row)

                all_hms.append(np.concatenate(b_all_hms, axis=0))
            all_hms = np.concatenate(all_hms, axis=1)
            cv2.imshow("all_hms", all_hms)

            bone_indices = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [7, 8], [8, 9], [10, 11], [11, 12]])
            joints_colors = [[255, 255, 255]]
            for cur_bs in pd_fb_result[0]:
                if cur_bs == 0:
                    joints_colors.append([128, 128, 128])
                elif cur_bs == 1:
                    joints_colors.append([255, 255, 255])
                elif cur_bs == 2:
                    joints_colors.append([0, 0, 0])
            img_for_display = display_utils.drawLines(cur_img, pd_joints_2d[0], indices=bone_indices)
            img_for_display = display_utils.drawPoints(img_for_display, pd_joints_2d[0], point_color_table=joints_colors, text_scale=0.3, point_ratio=3)

            cv2.imshow("visual_result", img_for_display)

            while True:
                key = cv2.waitKey(2)
                if key == ord("j"):
                    data_index.val += 1
                    break
                elif key == ord("k"):
                    data_index.val -= 1
                    break
                elif key == ord("q"):
                    quit()
