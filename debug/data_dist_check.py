import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.visualize_utils import display_utils

data_source = "train"

img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+data_source+"/images/{}.jpg".format(x)
lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+data_source+"/labels/{}.npy".format(x)

data_range_file = "../train/train_range/sec_3/"+data_source+"_range.npy"

test_iterations = 600000
batch_size = 4
save_file = "dist_debug"

#################################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################

    ############################ range section 3 ##########################
    data_range = np.load(data_range_file)
    np.random.shuffle(data_range)
    distrib_arr = np.zeros([len(data_range)], dtype=np.int32)

    img_list = [img_path(i) for i in data_range]
    lbl_list = [lbl_path(i) for i in data_range]

    ###################################################################

    with tf.device('/cpu:0'):
        data_iter, data_init_op = ordinal_reader.get_data_iterator(img_list, lbl_list, batch_size=batch_size, name="data_reader")

    global_step = 0
    with tf.Session() as sess:
        sess.run([data_init_op])

        while global_step < test_iterations:
            sys.stderr.write("\rCurrent Iterations: {:08d}".format(global_step))
            sys.stderr.flush()

            cur_data_batch = sess.run(data_iter)

            for b in range(batch_size):
                img_num = int(os.path.basename(cur_data_batch[0][b]).split(".")[0])
                label_num = int(os.path.basename(cur_data_batch[1][b]).split(".")[0])
                assert(img_num == label_num)

                distrib_arr[img_num] += 1

            global_step += 1

        np.save(save_file, distrib_arr)
