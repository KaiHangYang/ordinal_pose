import numpy as np
import os
import sys
import h5py
import cv2

sys.path.append("../../")

from utils.preprocess_utils import common as img_utils
from utils.visualize_utils import display_utils

settings = {
        "img_dir": lambda x: os.path.join("/home/kaihang/Projects/ordinal-pose3d/data/h36m/images", x), # valid data path
        "img_list_file": "/home/kaihang/Projects/ordinal-pose3d/data/h36m/annot/valid_images.txt", # valid img list
        "annot_file": "/home/kaihang/Projects/ordinal-pose3d/data/h36m/annot/valid.h5",
        "target_image": lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/images/{}".format(x),
        "target_label": lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/valid/labels/{}".format(x)
        }

if __name__ == "__main__":
    ##################### Here I crop the human3.6m to size 256 first ####################
    # read the img list
    img_list = None
    with open(settings["img_list_file"]) as f:
        img_list_arr = f.readlines()
        img_list = [settings["img_dir"](i.strip()) for i in img_list_arr]

    # keys: S [:, 17, 3], center [:], imgname [:], index [:], normalize [:], part[:, 17, 2]
    img_annots = h5py.File(settings["annot_file"], "r")

    for i, img_path in enumerate(img_list):
        sys.stderr.write("\rCurrently process {}".format(i))
        sys.stderr.flush()

        img = cv2.imread(img_path)

        joints_2d = img_annots["part"][i]
        joints_3d = img_annots["S"][i]
        joints_zidx = img_annots["zind"][i]

        crop_center = img_annots["center"][i]
        crop_scale = img_annots["scale"][i]

        cropped_img, cropped_joints_2d, is_discard = img_utils.data_resize_with_center_cropped(img, joints_2d, crop_center, crop_scale, crop_box_size=256, target_size=256)

        cv2.imwrite(settings["target_image"]("{}.jpg".format(i)), cropped_img)
        np.save(settings["target_label"](i), {"source": os.path.basename(img_path), "joints_2d": joints_2d, "joints_3d": joints_3d, "joints_zidx": joints_zidx, "center": crop_center, "scale": crop_scale})

