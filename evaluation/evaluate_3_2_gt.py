import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import ordinal_3_2
from utils.dataread_utils import ordinal_3_1_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators

##################### Setting for training ######################

nJoints = 17
batch_size = 1
img_size = 256

######################## To modify #############################
section = "3_2"

trash_log = "trash_"
valid_log_dir = "../"+trash_log+"logs/evaluate/"+section+"_gt/valid"
valid_data_source = "valid"
################################################################
restore_model_path = "../models/"+section+"_gt/ordinal_"+section+"_gt-300000"
learning_rate = 2.5e-4
lr_decay_rate = 1.0
lr_decay_step = 2000

valid_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+valid_data_source+"/images/{}.jpg".format(x)
valid_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+valid_data_source+"/labels/{}.npy".format(x)

valid_range_file = "../train/train_range/sec_3/"+valid_data_source+"_range.npy"

#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################

    valid_range = np.load(valid_range_file)
    data_from = 0
    data_to = len(valid_range)
    valid_data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)

    valid_img_list = [valid_img_path(i) for i in valid_range]
    valid_lbl_list = [valid_lbl_path(i) for i in valid_range]

    input_images = tf.placeholder(shape=[batch_size, img_size, img_size, 3], dtype=tf.float32)
    input_coords = tf.placeholder(shape=[batch_size, nJoints, 3], dtype=tf.float32)
    ordinal_model = ordinal_3_2.mOrdinal_3_2(nJoints, img_size=img_size, batch_size=batch_size, is_training=False)

    gt_3_2_eval = evaluators.mEvaluator3_2_gt(nJoints=nJoints)

    with tf.Session() as sess:

        with tf.device("/device:GPU:0"):
            ordinal_model.build_model(input_images)
        ordinal_model.build_loss_gt(input_coords, learning_rate, lr_decay_rate=lr_decay_rate, lr_decay_step=lr_decay_step)

        print("Network built!")
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])
        # reload the model
        if os.path.exists(restore_model_path+".index"):
            print("#######################Restored all weights ###########################")
            model_saver.restore(sess, restore_model_path)
        else:
            print("The prev model is not existing!")
            quit()

        while not valid_data_index.isEnd():
            global_steps = sess.run(ordinal_model.global_steps)

            batch_images_np = np.zeros([batch_size, img_size, img_size, 3], dtype=np.float32)
            batch_coords_np = np.zeros([batch_size, nJoints, 3], dtype=np.float32)

            img_path_for_show = []
            label_path_for_show = []

            for b in range(batch_size):
                img_path_for_show.append(os.path.basename(valid_img_list[valid_data_index.val]))
                label_path_for_show.append(os.path.basename(valid_lbl_list[valid_data_index.val]))

                cur_img = cv2.imread(valid_img_list[valid_data_index.val])
                cur_label = np.load(valid_lbl_list[valid_data_index.val]).tolist()
                valid_data_index.val += 1

                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"]], axis=1)

                cur_joints_3d = cur_joints[:, 2:5]
                batch_coords_np[b] = cur_joints_3d - cur_joints_3d[0] # related to the root
                batch_images_np[b] = preprocessor.img2train(cur_img, [-1, 1])

            acc, coords, loss, summary  = sess.run([ordinal_model.accuracy, ordinal_model.result, ordinal_model.loss, ordinal_model.merged_summary],
                    feed_dict={input_images: batch_images_np, input_coords: batch_coords_np})
            valid_log_writer.add_summary(summary, global_steps)

            gt_3_2_eval.add(batch_coords_np, coords)

            print("Iteration: {:07d} \nLoss : {:07f}\nCoords accuracy: {:07f}\n\n".format(global_steps, loss, acc))
            print((len(img_path_for_show) * "{}\n").format(*zip(img_path_for_show, label_path_for_show)))
            gt_3_2_eval.printMean()

        gt_3_2_eval.save("../eval_result/gt_3_2/eval")
