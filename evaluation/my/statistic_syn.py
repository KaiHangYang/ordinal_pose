import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../../")
from net import syn_net
from utils.preprocess_utils import syn_preprocess
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators

##################### Setting for training ######################
import configs

# t means gt(0) or ord(1)
# ver means experiment version
# d means validset(0) or trainset(1)
configs.parse_configs(t=0, ver=1, d=0)
configs.print_configs()

evaluation_models = [20000]
special_case_save_dir = lambda x: "/home/kaihang/Desktop/test_dir/special_cases_256/{}".format(x)
#################################################################

keep_showing = False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        initial_index = int(sys.argv[1])
    else:
        initial_index = 0

    network_batch_size = configs.batch_size
    ################### Initialize the data reader ####################
    range_arr = np.load(configs.range_file)
    data_from = 0
    # data_from = 50000
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    ###################################################################
    input_images = tf.placeholder(shape=[network_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_center_2d = tf.placeholder(shape=[network_batch_size, configs.nJoints, 2], dtype=tf.float32, name="input_center_2d")

    # both the fb_info and br_info are one-hot arrays
    input_fb_info = tf.placeholder(shape=[network_batch_size, configs.nJoints-1, 3], dtype=tf.float32, name="input_fb_info")
    input_br_info = tf.placeholder(shape=[network_batch_size, (configs.nJoints-1) * (configs.nJoints-2) / 2, 3], dtype=tf.float32, name="input_br_info") # up-triangle of the matrix

    syn_model = syn_net.mSynNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=network_batch_size, is_training=False, loss_weight_heatmaps=1.0, loss_weight_fb=1.0, loss_weight_br=1.0, is_use_bn=False)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)
            input_heatmaps = syn_model.build_input_heatmaps(input_center_2d, name="input_heatmaps", stddev=2.0, gaussian_coefficient=False)

        syn_model.build_loss(input_heatmaps=input_heatmaps, input_fb=input_fb_info, input_br=input_br_info, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

        print("Network built!")

        model_saver = tf.train.Saver(max_to_keep=70)
        net_init = tf.global_variables_initializer()

        sess.run([net_init])

        for cur_model_iterations in evaluation_models:
            # reload the model
            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=initial_index)

            fb_evaluator = evaluators.mEvaluatorFB_BR(nData=configs.nJoints - 1)
            br_evaluator = evaluators.mEvaluatorFB_BR(nData=(configs.nJoints - 2) * (configs.nJoints - 1) / 2)

            ################# Restore the model ################
            if os.path.exists(configs.restore_model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.restore_model_path_fn(cur_model_iterations))
            else:
                print(configs.restore_model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():
                global_steps = sess.run(syn_model.global_steps)

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_center_2d = np.zeros([configs.batch_size, configs.nJoints, 2], dtype=np.float32)
                batch_fb_info = np.zeros([configs.batch_size, configs.nJoints-1, 3], dtype=np.float32)
                batch_br_info = np.zeros([configs.batch_size, (configs.nJoints-1)*(configs.nJoints-2) / 2, 3], dtype=np.float32)

                # Generate the data batch
                img_path_for_show = [[] for i in range(configs.batch_size)]
                label_path_for_show = [[] for i in range(len(img_path_for_show))]

                fb_for_assert = []
                br_for_assert = []

                for b in range(configs.batch_size):
                    img_path_for_show[b] = os.path.basename(img_list[data_index.val])
                    label_path_for_show[b] = os.path.basename(lbl_list[data_index.val])

                    cur_img = cv2.imread(img_list[data_index.val])
                    # cur_img = cv2.imread("/home/kaihang/Desktop/b.jpg")
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    # the joints_2d in the label_syn is resize in [64, 64]
                    data_index.val += 1

                    cur_joints_2d = cur_label["joints_2d"].copy()
                    cur_bone_status = cur_label["bone_status"].copy()
                    cur_bone_relations = cur_label["bone_relations"].copy()

                    cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = syn_preprocess.preprocess(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=False, mask=None)
                    batch_images_np[b] = cur_img
                    batch_center_2d[b] = np.round(cur_joints_2d / configs.joints_2d_scale)
                    batch_fb_info[b] = np.eye(3)[cur_bone_status]
                    batch_br_info[b] = np.eye(3)[cur_bone_relations[np.triu_indices(configs.nJoints-1, k=1)]] # only get the upper triangle

                    ############# Just for debug #############
                    fb_for_assert.append(cur_bone_status.copy())
                    br_for_assert.append(cur_bone_relations[np.triu_indices(configs.nJoints-1, k=1)].copy())

                pd_heatmaps_0,\
                pd_heatmaps_1,\
                gt_heatmaps, \
                loss, \
                heatmaps_loss, \
                fb_loss, \
                br_loss, \
                hm_acc,\
                fb_acc,\
                br_acc,\
                gt_joints_2d,\
                pd_joints_2d,\
                gt_fb_result,\
                gt_fb_belief,\
                pd_fb_result,\
                pd_fb_belief,\
                gt_br_result,\
                gt_br_belief,\
                pd_br_result,\
                pd_br_belief = sess.run(
                        [
                         syn_model.heatmaps[0],
                         syn_model.heatmaps[1],
                         input_heatmaps,
                         syn_model.total_loss,
                         syn_model.heatmaps_loss,
                         syn_model.fb_loss,
                         syn_model.br_loss,
                         syn_model.heatmaps_acc,
                         syn_model.fb_acc,
                         syn_model.br_acc,

                         syn_model.gt_joints_2d,
                         syn_model.pd_joints_2d,
                         syn_model.gt_fb_result,
                         syn_model.gt_fb_belief,
                         syn_model.pd_fb_result,
                         syn_model.pd_fb_belief,
                         syn_model.gt_br_result,
                         syn_model.gt_br_belief,
                         syn_model.pd_br_result,
                         syn_model.pd_br_belief],
                        feed_dict={input_images: batch_images_np, input_center_2d: batch_center_2d, input_fb_info: batch_fb_info, input_br_info: batch_br_info})

                # gt_synmap, _ = syn_preprocess.draw_syn_img(cur_joints_2d, cur_bone_status, cur_bone_order, size=256, sep_size=64, bone_width=6, joint_ratio=6)
                assert((gt_fb_result == np.array(fb_for_assert)).all())
                assert((gt_br_result == np.array(br_for_assert)).all())

                fb_evaluator.add(gt_fb_result[b], pd_fb_result[b])
                br_evaluator.add(gt_br_result[b], pd_br_result[b])

                sys.stdout.write("FB INFO: ")
                fb_evaluator.printMean()


