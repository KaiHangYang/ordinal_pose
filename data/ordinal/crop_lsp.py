import numpy as np
import os
import sys
import cv2

lsp_data_dir = "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/raw_data/lsp_dataset_original"

source_img_path = lambda x: os.path.join(lsp_data_dir, "images/im{:04d}.jpg").format(x)
source_2d_path = lsp_data_dir 

target_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/images/{}.jpg"
target_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/labels/{}.npy"
