import ConfigParser
import os
import sys

#### Visualize Parameters ####
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

syn_restore_model_path_fn = None
pose_restore_model_path_fn = None

# t means gt(0) or ord(1)
# ver the version of the experiment
# d the data source valid(0) train(1)
def parse_configs(t, d):
    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, syn_restore_model_path_fn, pose_restore_model_path_fn, feature_map_size, joints_3d_scale, joints_2d_scale

    visual_type = "gt"
    data_source = ["valid", "train"][d]

    VISUAL_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    ### The visual.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(VISUAL_ROOT_DIR, "visual.conf"))

    joints_3d_scale = 1000.0
    joints_2d_scale = 4

    loss_weight_heatmap = 1.0
    loss_weight_xyzmap = 1.0
    feature_map_size = 64

    nJoints = config_parser.getint("data", "nJoints")
    batch_size = 4
    img_size = 256
    syn_img_size = 256
    sep_syn_img_size= 64

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source+"_range.npy")

    base_image_path = config_parser.get("dataset", "image_path")
    base_label_path = config_parser.get("dataset", "label_path")

    img_path_fn = lambda x: (base_image_path.format(data_source, "{}")).format(x)
    lbl_path_fn = lambda x: (base_label_path.format(data_source, "{}")).format(x)

    syn_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format("syn_1", visual_type))
    pose_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format("syn_2", visual_type))

    syn_restore_model_path_fn = lambda x: os.path.join(syn_model_dir, config_parser.get("model", "prefix").format("syn_1", visual_type, x))
    pose_restore_model_path_fn = lambda x: os.path.join(pose_model_dir, config_parser.get("model", "prefix").format("syn_2", visual_type, x))

def print_configs():
    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, syn_restore_model_path_fn, pose_restore_model_path_fn, feature_map_size, joints_3d_scale, joints_2d_scale
    print("##################### Visualize Parameters #####################")
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

    print("syn_restore_model_path_fn: {}".format(syn_restore_model_path_fn("{}")))
    print("pose_restore_model_path_fn: {}".format(pose_restore_model_path_fn("{}")))
