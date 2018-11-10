import ConfigParser
import os
import sys

#### Training Parameters ####
joints_3d_scale = None
joints_2d_scale = None

feature_map_size = None
loss_weight_heatmap = None
loss_weight_xyzmap = None
nJoints = None
train_batch_size = None
valid_batch_size = None
img_size = None
syn_img_size = None
sep_syn_img_size = None
learning_rate = None
lr_decay_rate = None
lr_decay_step = None
log_dir = None
train_range_file = None
lsp_range_file = None
mpii_range_file = None

lsp_img_path_fn = None
lsp_lbl_path_fn = None
mpii_img_path_fn = None
mpii_lbl_path_fn = None

train_img_path_fn = None
train_mask_path_fn = None

train_syn_img_path_fn = None
train_lbl_path_fn = None
valid_range_file = None
valid_img_path_fn = None
valid_syn_img_path_fn = None
valid_lbl_path_fn = None
model_dir = None
model_path_fn = None
model_path = None
train_iter = None
valid_iter = None

###############################

# t means gt(0) or ord(1)
def parse_configs(t, ver):

    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, train_batch_size, valid_batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, lsp_range_file, mpii_range_file, train_range_file, valid_range_file, lsp_img_path_fn, lsp_lbl_path_fn, mpii_img_path_fn, mpii_lbl_path_fn, train_img_path_fn, train_mask_path_fn, train_syn_img_path_fn, train_lbl_path_fn, valid_img_path_fn, valid_syn_img_path_fn, valid_lbl_path_fn, model_path_fn, model_dir, model_path, valid_iter, train_iter, feature_map_size, joints_3d_scale, joints_2d_scale

    train_type = ["gt", "ord"][t]
    cur_prefix = "syn_{}".format(ver)

    TRAIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ### The train.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(TRAIN_ROOT_DIR, "train.conf"))

    joints_3d_scale = 1000.0
    joints_2d_scale = 4.0

    loss_weight_heatmap = 3.0
    loss_weight_xyzmap = 10.0

    feature_map_size = 64

    nJoints = config_parser.getint("data", "nJoints")
    train_batch_size = config_parser.getint("data", "train_batch_size")
    valid_batch_size = config_parser.getint("data", "valid_batch_size")
    img_size = 256
    syn_img_size = 256
    sep_syn_img_size = 64

    train_iter = config_parser.getint("data", "train_iter")
    valid_iter = config_parser.getint("data", "valid_iter")

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "train/{}_".format(cur_prefix) + train_type)

    # Dataset Settings
    train_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "train_range.npy")
    lsp_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "lsp_range.npy")
    mpii_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "mpii_range.npy")

    valid_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "valid_range.npy")

    # base_image_path = config_parser.get("dataset", "image_path")
    # base_label_path = config_parser.get("dataset", "label_path")
    base_image_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/{}/images/{}.jpg"
    base_mask_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/{}/masks/{}.jpg"

    base_syn_image_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/{}/images_syn_256x256/{}.jpg"
    base_label_path = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/{}/labels_syn/{}.npy"

    train_img_path_fn = lambda x: (base_image_path.format("train", "{}")).format(x)
    train_mask_path_fn = lambda x: (base_mask_path.format("train", "{}")).format(x)
    train_syn_img_path_fn = lambda x: (base_syn_image_path.format("train", "{}")).format(x)
    train_lbl_path_fn = lambda x: (base_label_path.format("train", "{}")).format(x)

    lsp_img_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/images/{}.jpg".format(x)
    lsp_lbl_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/labels/{}.npy".format(x)

    mpii_img_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/mpii/images/{}.jpg".format(x)
    mpii_lbl_path_fn = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/mpii/labels/{}.npy".format(x)

    valid_img_path_fn = lambda x: (base_image_path.format("valid", "{}")).format(x)
    valid_syn_img_path_fn = lambda x: (base_syn_image_path.format("valid", "{}")).format(x)
    valid_lbl_path_fn = lambda x: (base_label_path.format("valid", "{}")).format(x)

    model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format(cur_prefix, train_type))
    model_path_fn = lambda x: os.path.join(model_dir, config_parser.get("model", "prefix").format(cur_prefix, train_type, x))

    # remove the '-'
    model_path = model_path_fn("")[0:-1]

def print_configs():
    global loss_weight_heatmap, loss_weight_xyzmap, nJoints, train_batch_size, valid_batch_size, img_size, syn_img_size, sep_syn_img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, train_range_file, valid_range_file, lsp_img_path_fn, lsp_lbl_path_fn, mpii_img_path_fn, mpii_lbl_path_fn, train_img_path_fn, train_mask_path_fn, train_syn_img_path_fn, train_lbl_path_fn, valid_img_path_fn, valid_syn_img_path_fn, valid_lbl_path_fn, model_path_fn, model_dir, model_path, valid_iter, train_iter, feature_map_size, joints_3d_scale, joints_2d_scale
    print("##################### Training Parameters #####################")
    print("##### Data Parameters")
    print("loss_weight_heatmap: {}\nloss_weight_xyzmap: {}\nnJoints: {}\ntrain_batch_size: {}\nvalid_batch_size: {}\nimg_size: {}\nsyn_img_size: {}\nsep_syn_img_size: {}\ntrain_iter: {}\nvalid_iter: {}".format(loss_weight_heatmap, loss_weight_xyzmap, nJoints, train_batch_size, valid_batch_size, img_size, syn_img_size, sep_syn_img_size, train_iter, valid_iter))
    print("feature_map_size: {}".format(feature_map_size))
    print("joints_3d_scale: {}\njoints_2d_scale: {}".format(joints_3d_scale, joints_2d_scale))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("train_range_file: {}".format(train_range_file))
    print("train_img_path: {}".format(train_img_path_fn("{}")))
    print("train_mask_path: {}".format(train_mask_path_fn("{}")))

    print("train_syn_img_path: {}".format(train_syn_img_path_fn("{}")))
    print("train_lbl_path: {}".format(train_lbl_path_fn("{}")))

    print("lsp_img_path: {}".format(lsp_img_path_fn("{}")))
    print("lsp_lbl_path: {}".format(lsp_lbl_path_fn("{}")))

    print("mpii_img_path: {}".format(mpii_img_path_fn("{}")))
    print("mpii_lbl_path: {}".format(mpii_lbl_path_fn("{}")))

    print("valid_range_file: {}".format(valid_range_file))
    print("valid_img_path: {}".format(valid_img_path_fn("{}")))
    print("valid_syn_img_path: {}".format(valid_syn_img_path_fn("{}")))
    print("valid_lbl_path: {}".format(valid_lbl_path_fn("{}")))

    print("model_dir: {}".format(model_dir))
    print("model_path_fn: {}".format(model_path_fn("{}")))
    print("model_path: {}".format(model_path))

