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
fb_nJoints = None
batch_size = None
img_size = None
syn_img_size = None
sep_syn_img_size = None
learning_rate = None
lr_decay_rate = None
lr_decay_step = None
log_dir = None
range_file = None
lsp_range_file = None
mpii_range_file = None

img_path_fn = None
lbl_path_fn = None

lsp_img_path_fn = None
lsp_lbl_path_fn = None

mpii_img_path_fn = None
mpii_lbl_path_fn = None

syn_restore_model_path_fn = None
pose_restore_model_path_fn = None
fb_restore_model_path_fn = None

# t means gt(0) or ord(1)
# ver the version of the experiment
# d the data source valid(0) train(1)
def parse_configs(t, d):
    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, fb_nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, lsp_range_file, mpii_range_file, range_file, mpii_img_path_fn, mpii_lbl_path_fn, lsp_img_path_fn, lsp_lbl_path_fn, img_path_fn, lbl_path_fn, syn_restore_model_path_fn, pose_restore_model_path_fn, fb_restore_model_path_fn, feature_map_size, joints_3d_scale, joints_2d_scale

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
    fb_nJoints = 13

    batch_size = 4
    img_size = 256
    syn_img_size = 256
    sep_syn_img_size= 64

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source+"_range.npy")
    lsp_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "lsp_range.npy")
    mpii_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "mpii_range.npy")

    base_image_path = config_parser.get("dataset", "image_path")
    base_label_path = config_parser.get("dataset", "label_path")

    img_path_fn = lambda x: (base_image_path.format(data_source, "{}")).format(x)
    lbl_path_fn = lambda x: (base_label_path.format(data_source, "{}")).format(x)

    lsp_img_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/images/{}.jpg".format(x)
    lsp_lbl_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/labels/{}.npy".format(x)

    mpii_img_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/mpii/images/{}.jpg".format(x)
    mpii_lbl_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/mpii/labels/{}.npy".format(x)

    syn_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format("syn_1", visual_type))
    pose_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format("syn_2", visual_type))
    fb_model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format("syn_3", visual_type))

    syn_restore_model_path_fn = lambda x: os.path.join(syn_model_dir, config_parser.get("model", "prefix").format("syn_1", visual_type, x))
    pose_restore_model_path_fn = lambda x: os.path.join(pose_model_dir, config_parser.get("model", "prefix").format("syn_2", visual_type, x))
    fb_restore_model_path_fn = lambda x: os.path.join(fb_model_dir, config_parser.get("model", "prefix").format("syn_3", visual_type, x))

def print_configs():
    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, fb_nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, lsp_range_file, mpii_range_file, range_file, mpii_img_path_fn, mpii_lbl_path_fn, lsp_img_path_fn, lsp_lbl_path_fn, img_path_fn, lbl_path_fn, syn_restore_model_path_fn, pose_restore_model_path_fn, fb_restore_model_path_fn, feature_map_size, joints_3d_scale, joints_2d_scale
    print("##################### Visualize Parameters #####################")
    print("##### Data Parameters")
    print("loss_weight_heatmap: {}\nloss_weight_xyzmap: {}\nnJoints: {}\nfb_nJoints: {}\nbatch_size: {}\nimg_size: {}\nsyn_img_size: {}\nsep_syn_img_size: {}".format(loss_weight_heatmap, loss_weight_xyzmap, nJoints, fb_nJoints, batch_size, img_size, syn_img_size, sep_syn_img_size))
    print("feature_map_size: {}".format(feature_map_size))
    print("joints_3d_scale: {}\njoints_2d_scale: {}".format(joints_3d_scale, joints_2d_scale))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("range_file: {}".format(range_file))
    print("lsp_range_file: {}".format(lsp_range_file))
    print("mpii_range_file: {}".format(mpii_range_file))

    print("img_path: {}".format(img_path_fn("{}")))
    print("lbl_path: {}".format(lbl_path_fn("{}")))

    print("lsp_img_path: {}".format(lsp_img_path_fn("{}")))
    print("lsp_lbl_path: {}".format(lsp_lbl_path_fn("{}")))

    print("mpii_img_path: {}".format(mpii_img_path_fn("{}")))
    print("mpii_lbl_path: {}".format(mpii_lbl_path_fn("{}")))

    print("syn_restore_model_path_fn: {}".format(syn_restore_model_path_fn("{}")))
    print("pose_restore_model_path_fn: {}".format(pose_restore_model_path_fn("{}")))
    print("fb_restore_model_path_fn: {}".format(fb_restore_model_path_fn("{}")))
