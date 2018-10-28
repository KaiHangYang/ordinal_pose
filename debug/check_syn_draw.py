import numpy as np
import os
import sys
import cv2
import time

sys.path.append("../")

from utils.preprocess_utils import syn_preprocess

if __name__ == "__main__":

    label_dir_fn = lambda x: os.path.join("/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/labels_syn", x)
    image_dir_fn = lambda x: os.path.join("/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images_syn_64x64", x)
    real_image_dir_fn = lambda x: os.path.join("/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images", x)

    label_lists = os.listdir(label_dir_fn(""))

    for cur_label_name in label_lists:
        cur_label = np.load(label_dir_fn(cur_label_name)).tolist()
        cur_image = cv2.imread(image_dir_fn("{}.jpg".format(cur_label_name.split(".")[0])))
        cur_real_image = cv2.imread(real_image_dir_fn("{}.jpg".format(cur_label_name.split(".")[0])))

        joints_2d = cur_label["joints_2d"] * 4
        bone_status = cur_label["bone_status"]
        bone_order = cur_label["bone_order"]

        print(bone_order)

        SYN_TIME = time.time()

        # Test the preprocess scripts
        nJoints = joints_2d.shape[0]

        cur_real_image, joints_2d, bone_status, bone_order = syn_preprocess.preprocess(cur_real_image, joints_2d=joints_2d, bone_status=bone_status, bone_order=bone_order, is_training=True)

        joints_2d = joints_2d / 4

        print(bone_order)

        synmap, sep_synmaps = syn_preprocess.draw_syn_img(joints_2d, bone_status, bone_order)
        SYN_TIME = time.time() - SYN_TIME

        cv2.imshow("painted_img", cv2.resize(cur_image, (256, 256)))
        cv2.imshow("synmap", cv2.resize(synmap, (256, 256)))
        cv2.imshow("real_img", cur_real_image)
        cv2.waitKey()
        print("SYN_TIME", SYN_TIME)

        print(cur_label["source"])
