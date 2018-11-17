import ConfigParser
import os
import sys
import numpy as np

#### Parameters for preprocessor reconfigure ####
H36M_JOINTS_SELECTED = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16])
NEW_BONE_INDICES = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]])
NEW_FLIP_ARRAY = np.array([[1, 4], [2, 5], [3, 6], [9, 12], [10, 13], [11, 14]])
NEW_BONE_COLORS = np.array([[1.000000, 1.000000, 0.000000], [0.492543, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [1.000000, 0.349454, 0.000000], [0.499439, 0.558884, 1.000000], [0.000000, 0.362774, 0.000000], [0.500312, 0.000000, 0.624406], [0.000000, 1.000000, 1.000000], [1.000000, 0.499433, 0.611793], [1.000000, 0.800000, 1.000000], [0.000000, 0.502502, 0.611632], [0.200000, 0.700000, 0.300000], [0.700000, 0.300000, 0.100000], [0.300000, 0.200000, 0.800000]])

#### Evaluating Parameters ####
joints_3d_scale = None
joints_2d_scale = None

feature_map_size = None
loss_weight_heatmap = None
loss_weight_xyzmap = None
nJoints = None

batch_size = None
img_size = None
syn_img_size = None
sep_syn_img_size = None
learning_rate = None
lr_decay_rate = None
lr_decay_step = None
log_dir = None
range_file = None
img_path_fn = None
lbl_path_fn = None

model_dir = None
restore_model_path_fn = None
syn_restore_model_path_fn = None
pose_restore_model_path_fn = None
model_path = None

###############################

# t means gt(0) or ord(1)
# ver the version of the experiment
# d the data source valid(0) train(1)
def parse_configs(t, ver, d, all_type=0):

    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, syn_restore_model_path_fn, pose_restore_model_path_fn, model_dir, model_path, feature_map_size, joints_3d_scale, joints_2d_scale 

    eval_type = "gt"
    cur_prefix = "syn_{}".format(ver)
    data_source = ["valid", "train"][d]

    if all_type == 0:
        syn_prefix = "syn_1"
        pose_prefix = "syn_2"
    else:
        syn_prefix = "syn_3"
        pose_prefix = "syn_4"

    EVAL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ### The eval.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(EVAL_ROOT_DIR, "eval.conf"))

    joints_3d_scale = 1000.0
    joints_2d_scale = 4

    loss_weight_heatmap = 1.0
    loss_weight_xyzmap = 1.0
    feature_map_size = 64

    nJoints = 15
    batch_size = 4
    img_size = 256
    syn_img_size = 256
    sep_syn_img_size= 64

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "eval/{}_".format(cur_prefix) + eval_type)

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source+"_range.npy")

    # base_image_path = config_parser.get("dataset", "image_path")
    # base_label_path = config_parser.get("dataset", "label_path")

    base_image_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/{}/images/{}.jpg"
    base_label_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/{}/labels_syn/{}.npy"

    img_path_fn = lambda x: (base_image_path.format(data_source, "{}")).format(x)
    lbl_path_fn = lambda x: (base_label_path.format(data_source, "{}")).format(x)

    model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format(cur_prefix, eval_type))

    syn_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format(syn_prefix, eval_type))
    pose_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format(pose_prefix, eval_type))

    restore_model_path_fn = lambda x: os.path.join(model_dir, config_parser.get("model", "prefix").format(cur_prefix, eval_type, x))

    syn_restore_model_path_fn = lambda x: os.path.join(syn_model_dir, config_parser.get("model", "prefix").format(syn_prefix, eval_type, x))
    pose_restore_model_path_fn = lambda x: os.path.join(pose_model_dir, config_parser.get("model", "prefix").format(pose_prefix, eval_type, x))

    # remove the '-'
    model_path = restore_model_path_fn("")[0:-1]

def print_configs():
    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, syn_restore_model_path_fn, pose_restore_model_path_fn, model_dir, model_path, feature_map_size, joints_3d_scale, joints_2d_scale
    print("##################### Evaluation Parameters #####################")
    print("##### Data Parameters")
    print("loss_weight_heatmap: {}\nloss_weight_xyzmap: {}\nnJoints: {}\nbatch_size: {}\nimg_size: {}\nsyn_img_size: {}\nsep_syn_img_size: {}".format(loss_weight_heatmap, loss_weight_xyzmap, nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size))
    print("feature_map_size: {}".format(feature_map_size))
    print("joints_3d_scale: {}\njoints_2d_scale: {}".format(joints_3d_scale, joints_2d_scale))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("range_file: {}".format(range_file))
    print("img_path: {}".format(img_path_fn("{}")))
    print("lbl_path: {}".format(lbl_path_fn("{}")))

    print("model_dir: {}".format(model_dir))
    print("restore_model_path_fn: {}".format(restore_model_path_fn("{}")))
    print("syn_restore_model_path_fn: {}".format(syn_restore_model_path_fn("{}")))
    print("pose_restore_model_path_fn: {}".format(pose_restore_model_path_fn("{}")))
    print("model_path: {}".format(model_path))

