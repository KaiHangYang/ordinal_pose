import numpy as np
import os
import sys
import cv2
sys.path.append("../../")

from utils.preprocess_utils import syn_preprocess

img_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images/{}.jpg".format(x)
mask_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/masks/{}.jpg".format(x)

if __name__ == "__main__":
    img_num = len(os.listdir("/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/images"))

    for i in range(img_num):
        cur_img = cv2.imread(img_path_fn(i))
        cur_mask = cv2.imread(mask_path_fn(i))

        cur_img = syn_preprocess.replace_bg(cur_img, cur_mask)

        cv2.imshow("img", cur_img)
        cv2.waitKey(1)
