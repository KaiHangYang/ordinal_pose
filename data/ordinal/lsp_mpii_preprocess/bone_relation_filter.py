import numpy as np
import os
import sys
import cv2

sys.path.append("../../../")

from utils.visualize_utils import display_utils
from utils.preprocess_utils import syn_preprocess
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils.my_utils import mAverageCounter

from get_bone_overlap import get_bone_overlap

preprocessor = syn_preprocess.SynProcessor(skeleton=skeleton, img_size=256, bone_width=6, joint_ratio=6, bg_color=0.2, bone_status_threshold=80, overlap_threshold=6)

datasets = ["lsp", "mpii"]

source_img_dir_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/images/".format(x)
source_lbl_dir_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/full_labels".format(x)
target_lbl_dir_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/full_labels".format(x)

if __name__ == "__main__":

    total = 0
    labeled = 0
    cur_frame = 0
    for cur_dataset in datasets:
        cur_source_img_dir = source_img_dir_fn(cur_dataset)
        cur_source_lbl_dir = source_lbl_dir_fn(cur_dataset)
        cur_target_lbl_dir = target_lbl_dir_fn(cur_dataset)

        for cur_lbl in os.listdir(cur_source_lbl_dir):
            cur_source_lbl_path = os.path.join(cur_source_lbl_dir, cur_lbl)
            cur_target_lbl_path = os.path.join(cur_target_lbl_dir, cur_lbl)
            cur_source_img_path = os.path.join(cur_source_img_dir, cur_lbl.split(".")[0] + ".jpg")

            cur_img = cv2.imread(cur_source_img_path)
            cur_source_lbl = np.load(cur_source_lbl_path).tolist()

            joints_2d = cur_source_lbl["joints_2d"]
            bone_relations = cur_source_lbl["bone_relations"]
            bone_order = preprocessor.bone_order_from_bone_relations(bone_relations, np.ones_like(bone_relations))
            bone_status = cur_source_lbl["bone_status"]

            overlapped_bone_pairs = np.reshape(get_bone_overlap(joints_2d.flatten().tolist(), 6), [-1, 2])

            # new_bone_relations = np.zeros_like(bone_relations)

            # for cur_pair in overlapped_bone_pairs:
                # new_bone_relations[cur_pair[0], cur_pair[1]] = bone_relations[cur_pair[0], cur_pair[1]]
                # new_bone_relations[cur_pair[1], cur_pair[0]] = bone_relations[cur_pair[1], cur_pair[0]]

            # cur_source_lbl["bone_relations"] = new_bone_relations


            # np.save(cur_target_lbl_path, cur_source_lbl)

            ####################### Counter the unlabeled relations #########################
            # total += len(overlapped_bone_pairs)
            # for cur_pair in overlapped_bone_pairs:
                # if bone_relations[cur_pair[0], cur_pair[1]] != 0:
                    # labeled += 1

            # print("Current frame {}, {}/{}".format(cur_frame, labeled, total))
            # cur_frame += 1
            #################################################################################

            ####################### Visualize the Relations #########################
            syn_img = preprocessor.draw_syn_img(joints_2d, bone_status, bone_order)
            cv2.imshow("raw_img", display_utils.drawLines(cur_img, joints_2d, indices=skeleton.bone_indices, color_table=(skeleton.bone_colors * 255.0).astype(np.int)))
            cv2.imshow("syn_img", syn_img)
            cv2.waitKey()
            #################################################################################
