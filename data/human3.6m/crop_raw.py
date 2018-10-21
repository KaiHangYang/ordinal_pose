import numpy as np
import os
import sys
import h5py
import cv2

sys.path.append("../../")

from utils.preprocess_utils import common as img_utils
from utils.visualize_utils import display_utils
from utils.common_utils import h36m_camera


train_or_valid = "train"

settings = {
        "img_dir": lambda x: os.path.join("/home/kaihang/DataSet_2/Ordinal/human3.6m/raw_datas/"+train_or_valid+"/images/", x), # data path
        "img_list_file": "/home/kaihang/Projects/ordinal-pose3d/data/h36m/annot/"+train_or_valid+"_images.txt", # img list
        "annot_file": "/home/kaihang/DataSet_2/Ordinal/human3.6m/raw_datas/"+train_or_valid+"/"+train_or_valid+".h5",
        "target_image": lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+train_or_valid+"/images/{}".format(x),
        "target_label": lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+train_or_valid+"/labels/{}".format(x),
        "target_label_syn": lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+train_or_valid+"/labels_syn/{}.txt".format(x)

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

        cropped_img, cropped_joints_2d, is_discard = img_utils.data_resize_with_center_cropped(img, joints_2d.copy(), crop_center.copy(), crop_scale, crop_box_size=256, target_size=256)

        # cv2.imwrite(settings["target_image"]("{}.jpg".format(i)), cropped_img)
        # np.save(settings["target_label"](i), {"source": os.path.basename(img_path), "joints_2d": cropped_joints_2d, "joints_3d": joints_3d, "joints_zidx": joints_zidx, "center": crop_center, "scale": crop_scale})
        with open(settings["target_label_syn"](i), "w") as f:
            img_name = os.path.basename(img_path)
            #################### Write img_name ###################
            f.write("{}\n".format(img_name)) # img_name

            # get the camera matrix
            subject_num = int(img_name.split("_")[0][1])
            camera_num = int(img_name.split(".")[1].split("_")[0])
            _, cam_matrix = h36m_camera.get_cam_mat(subject_num, camera_num)
            cam_matrix = cam_matrix.flatten().tolist()

            #################### Write cam_matrix ################
            f.write(("{} " * len(cam_matrix) + "\n").format(*cam_matrix))

            ################### Write center and scale ###############
            f.write("{} {} {}\n".format(crop_center[0], crop_center[1], crop_scale))
            ################### Write joints_2d_raw and joints_3d_raw ##################
            flattened_joints_2d = joints_2d.flatten().tolist()
            flattened_joints_3d = joints_3d.flatten().tolist()
            f.write(("{} " * len(flattened_joints_2d) + "\n").format(*flattened_joints_2d))
            f.write(("{} " * len(flattened_joints_3d) + "\n").format(*flattened_joints_3d))
