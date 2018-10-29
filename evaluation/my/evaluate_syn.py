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

evaluation_models = [320000]
#################################################################

if __name__ == "__main__":

    network_batch_size = configs.batch_size
    ################### Initialize the data reader ####################
    range_arr = np.load(configs.range_file)
    data_from = 13000
    data_to = len(range_arr)

    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    ###################################################################
    input_images = tf.placeholder(shape=[network_batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")

    input_sep_synmaps = tf.placeholder(shape=[network_batch_size, configs.syn_img_size, configs.syn_img_size, 3*(configs.nJoints - 1)], dtype=tf.float32, name="input_sep_synmaps")
    input_synmap = tf.placeholder(shape=[network_batch_size, configs.syn_img_size, configs.syn_img_size, 3], dtype=tf.float32, name="input_synmap")

    syn_model = syn_net.mSynNet(nJoints=configs.nJoints, img_size=configs.img_size, batch_size=network_batch_size, is_training=False, loss_weight_sep_synmaps=1.0, loss_weight_synmap=10.0)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)

        syn_model.build_loss(input_sep_synmaps=input_sep_synmaps, input_synmap=input_synmap, lr=configs.learning_rate, lr_decay_step=configs.lr_decay_step, lr_decay_rate=configs.lr_decay_rate)

        print("Network built!")

        model_saver = tf.train.Saver(max_to_keep=70)
        net_init = tf.global_variables_initializer()

        sess.run([net_init])

        for cur_model_iterations in evaluation_models:
            # reload the model
            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

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

                batch_synmap_np = np.zeros([configs.batch_size, configs.syn_img_size, configs.syn_img_size, 3], dtype=np.float32)
                batch_sep_synmaps_np = np.zeros([configs.batch_size, configs.syn_img_size, configs.syn_img_size, 3 * (configs.nJoints - 1)], dtype=np.float32)

                # Generate the data batch
                img_path_for_show = [[] for i in range(configs.batch_size)]
                label_path_for_show = [[] for i in range(len(img_path_for_show))]

                for b in range(configs.batch_size):
                    img_path_for_show[b] = os.path.basename(img_list[data_index.val])
                    label_path_for_show[b] = os.path.basename(lbl_list[data_index.val])

                    cur_img = cv2.imread(img_list[data_index.val])
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    # the joints_2d in the label_syn is resize in [64, 64]

                    cur_joints_2d = cur_label["joints_2d"].copy() * configs.joints_2d_scale
                    cur_bone_status = cur_label["bone_status"].copy()
                    cur_bone_order = cur_label["bone_order"].copy()

                    cur_img, cur_joints_2d, cur_bone_status, cur_bone_order = syn_preprocess.preprocess(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_order=cur_bone_order, is_training=False)

                    cur_joints_2d = cur_joints_2d / configs.joints_2d_scale

                    # draw the synsetic imgs as the ground truth
                    cur_synmap, cur_sep_synmaps = syn_preprocess.draw_syn_img(cur_joints_2d, cur_bone_status, cur_bone_order)
                    batch_images_np[b] = cur_img

                    # make them [0, 1]
                    batch_sep_synmaps_np[b] = np.concatenate(cur_sep_synmaps, axis=2) / 255.0
                    batch_synmap_np[b] = cur_synmap / 255.0

                sep_synmaps, \
                synmap, \
                sep_synmaps_loss, \
                synmap_loss = sess.run(
                        [
                         syn_model.sep_synmaps,
                         syn_model.synmap,
                         syn_model.sep_synmaps_loss,
                         syn_model.synmap_loss
                         ],
                        feed_dict={input_images: batch_images_np, input_sep_synmaps: batch_sep_synmaps_np, input_synmap: batch_synmap_np})

                cv2.imshow("raw_img", batch_images_np[0])
                cv2.imshow("pd_synmap", cv2.resize(synmap[0], (256, 256)))
                cv2.imshow("gt_synmap", cv2.resize(batch_synmap_np[0], (256, 256)))

                cv2.waitKey(2)

                for i in range(16):
                    cv2.imshow("pd_sepsynmap", cv2.resize(sep_synmaps[0, :, :, 3*i:3*i+3], (256, 256)))
                    cv2.imshow("gt_sepsynmap", cv2.resize(batch_sep_synmaps_np[0, :, :, 3*i:3*i+3], (256, 256)))
                    cv2.waitKey()

                print("Iteration: {:07d} \nSep synmaps Loss: {:07f}\nSynmap Loss: {:07f}\n\n".format(global_steps, sep_synmaps_loss, synmap_loss))
                print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
                print("\n\n")
