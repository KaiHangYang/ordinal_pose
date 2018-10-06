import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
from utils.visualize_utils import display_utils

data_source = "train"

img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+data_source+"/images/{}.jpg".format(x)
lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+data_source+"/labels/{}.npy".format(x)

data_range_file = "../data_range/sec_3/"+data_source+"_range.npy"

#################################################################

if __name__ == "__main__":

    ############################ range section 3 ##########################
    data_range = np.load(data_range_file)
    np.random.shuffle(data_range)

    total_data_sum = len(data_range)

    img_list = [img_path(i) for i in data_range]
    lbl_list = [lbl_path(i) for i in data_range]

    ###################################################################

    cur_index = 0
    with tf.Session() as sess:
        while cur_index < total_data_sum:
            print("\rCurrent Iterations: {:08d}".format(cur_index))

            cur_img = cv2.imread(img_list[cur_index])
            cur_label = np.load(lbl_list[cur_index]).tolist()

            cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"]], axis=1)
            print(cur_joints[:, 4])
            auged_img, auged_joints = preprocessor.preprocess(cur_img, cur_joints.copy())


            cur_img = display_utils.drawLines(cur_img, cur_joints[:, 0:2])
            cur_img = display_utils.drawPoints(cur_img, cur_joints[:, 0:2])

            auged_img = display_utils.drawLines(auged_img, auged_joints[:, 0:2])
            auged_img = display_utils.drawPoints(auged_img, auged_joints[:, 0:2])
            cv2.imshow("auged", auged_img.astype(np.uint8))
            cv2.imshow("raw", cur_img)

            cv2.waitKey()

            print(auged_joints[:, 4])

            cur_index += 1
