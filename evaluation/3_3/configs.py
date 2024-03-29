import ConfigParser
import os
import sys

#### Evaluating Parameters ####
depth_scale = None
coords_2d_scale = None

feature_map_size = None
loss_weight_heatmap = None
loss_weight_volume = None
nJoints = None
batch_size = None
img_size = None
learning_rate = None
lr_decay_rate = None
lr_decay_step = None
log_dir = None
range_file = None
img_path_fn = None
lbl_path_fn = None
model_dir = None
restore_model_path_fn = None
model_path = None

##### The parameters below only used in the ordinal mode
scale_batch_size = None
scale_range_file = None
scale_img_path_fn = None
scale_lbl_path_fn = None
###############################

###############################

# t means gt(0) or ord(1)
# d means validset(0) or trainset(1)
def parse_configs(t, d):

    global loss_weight_heatmap, loss_weight_volume, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, model_dir, model_path, feature_map_size, depth_scale, coords_2d_scale, scale_batch_size, scale_img_path_fn, scale_lbl_path_fn, scale_range_file

    eval_type = ["gt", "ord"][t]
    data_source = ["valid", "train"][d]

    EVAL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ### The eval.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(EVAL_ROOT_DIR, "eval.conf"))

    depth_scale = 1000.0 / 32.0
    coords_2d_scale = 4

    loss_weight_heatmap = 1.0
    loss_weight_volume = 1.0
    feature_map_size = 64

    nJoints = config_parser.getint("data", "nJoints")
    batch_size = config_parser.getint("data", "batch_size")
    scale_batch_size = batch_size

    # batch_size = 2 # use the big batch to speed up the evaluation
    img_size = config_parser.getint("data", "img_size")

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "eval/3_3_" + eval_type)

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source + "_range.npy")

    base_image_path = config_parser.get("dataset", "image_path")
    base_label_path = config_parser.get("dataset", "label_path")

    img_path_fn = lambda x: (base_image_path.format(data_source, "{}")).format(x)
    lbl_path_fn = lambda x: (base_label_path.format(data_source, "{}")).format(x)

    model_dir = os.path.join(config_parser.get("model", "base_dir"), "3_3_{}".format(eval_type))
    restore_model_path_fn = lambda x: os.path.join(model_dir, config_parser.get("model", "prefix").format("3_3", eval_type, x))

    # Parameters used in ordinal mode
    scale_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "scale_range.npy")
    scale_img_path_fn = lambda x: (base_image_path.format("train", "{}")).format(x)
    scale_lbl_path_fn = lambda x: (base_label_path.format("train", "{}")).format(x)

    # remove the '-'
    model_path = restore_model_path_fn("")[0:-1]

def print_configs():
    global loss_weight_heatmap, loss_weight_volume, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, model_dir, model_path, feature_map_size, depth_scale, coords_2d_scale, scale_batch_size, scale_img_path_fn, scale_lbl_path_fn, scale_range_file

    print("##################### Parameters #####################")
    print("##### Data Parameters")
    print("loss_weight_heatmap: {}\nloss_weight_volume: {}\nnJoints: {}\nbatch_size: {}\nscale_batch_size: {}\nimg_size: {}\n".format(loss_weight_heatmap, loss_weight_volume, nJoints, batch_size, scale_batch_size, img_size))
    print("feature_map_size: {}".format(feature_map_size))
    print("depth_scale: {}\ncoords_2d_scale: {}".format(depth_scale, coords_2d_scale))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("range_file: {}".format(range_file))
    print("img_path: {}".format(img_path_fn("{}")))
    print("lbl_path: {}".format(lbl_path_fn("{}")))

    print("model_dir: {}".format(model_dir))
    print("restore_model_path_fn: {}".format(restore_model_path_fn("{}")))
    print("model_path: {}".format(model_path))

