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
        "img_path": lambda x, y, z, n: "/home/kaihang/DataSet_2/Ordinal/human3.6m/raw_datas/"+train_or_valid+"/images/{}_{}.{}_{:06d}.jpg".format(x, y, z, n), # data path
        "img_list_file": "/home/kaihang/Projects/ordinal-pose3d/data/h36m/annot/"+train_or_valid+"_images.txt", # img list
        "mask_video_path": lambda x, y, z: "/home/kaihang/DataSet_2/DataSets/H3.6m/{}/MySegmentsMat/ground_truth_bs/{}.{}.mp4".format(x, y, z),
        "target_mask_path": lambda x, y, z, n: "/home/kaihang/DataSet_2/Ordinal/human3.6m/raw_datas/"+train_or_valid+"/masks/{}_{}.{}_{:06d}.jpg".format(x, y, z, n),
        }

if __name__ == "__main__":
    ##################### Here I crop the human3.6m to size 256 first ####################
    # read the img list
    img_list = None
    img_dict = {}
    with open(settings["img_list_file"]) as f:
        img_list_arr = f.readlines()
        img_list = [i.strip() for i in img_list_arr]

    for img_name in img_list:
        subject = img_name.split("_")[0]
        action, cam_n_index, _ = img_name.split(".")
        action = " ".join(action[len(subject)+1:].split("_"))
        cam_num, frame_index = cam_n_index.split("_")

        # print(subject, action, cam_num, frame_index)
        key = "_".join([subject, action, cam_num])

        if key not in img_dict.keys():
            img_dict[key] = []

        img_dict[key].append(int(frame_index))

    count = 0
    for cur_key in img_dict.keys():

        cur_mask_video_path = settings["mask_video_path"](*cur_key.split("_"))

        sub, act, cam = cur_key.split("_")
        act = "_".join(act.split(" "))

        cur_video = cv2.VideoCapture(cur_mask_video_path)

        for cur_frame_num in img_dict[cur_key]:
            sys.stderr.write("\rCurrent Process {}".format(count))
            sys.stderr.flush()
            count += 1
            cur_video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_num-1)
            _, cur_mask = cur_video.read()

            morph_kernel = np.ones([5, 5])
            cur_mask = cv2.morphologyEx(cur_mask, cv2.MORPH_OPEN, morph_kernel)

            cur_mask = np.invert(cur_mask.astype(np.bool))

            raw_img_path = settings["img_path"](sub, act, cam, cur_frame_num)
            target_mask_path = settings["target_mask_path"](sub, act, cam, cur_frame_num)
            # print(raw_img_path)
            # cur_img = cv2.imread(raw_img_path)
            # cur_img[cur_mask] = 0
            # cv2.imshow("frame", cur_img)
            # cv2.waitKey(3)
            cv2.imwrite(target_mask_path, cur_mask.astype(np.uint8) * 255)


    print("finished")
