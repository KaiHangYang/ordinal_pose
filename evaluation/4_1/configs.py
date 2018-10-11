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

###############################

# t means gt(0) or ord(1)
# ver the version of the experiment
# d the data source valid(0) train(1)
def parse_configs(t, ver, d):

    global loss_weight_heatmap, loss_weight_volume, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, model_dir, model_path, feature_map_size, depth_scale, coords_2d_scale

    eval_type = ["gt", "ord"][t]
    cur_prefix = "f_{}".format(ver)
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
    img_size = config_parser.getint("data", "img_size")

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "eval/{}_".format(cur_prefix) + eval_type)

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source+"_range.npy")

    img_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), data_source) + "/images/{}.jpg".format(x)
    lbl_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), data_source) + "/labels/{}.npy".format(x)

    model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}_{}".format(cur_prefix, eval_type))
    restore_model_path_fn = lambda x: os.path.join(model_dir, config_parser.get("model", "prefix").format(cur_prefix, eval_type, x))

    # remove the '-'
    model_path = restore_model_path_fn("")[0:-1]

def print_configs():
    global loss_weight_heatmap, loss_weight_volume, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, model_dir, model_path, feature_map_size, depth_scale, coords_2d_scale
    print("##################### Evaluation Parameters #####################")
    print("##### Data Parameters")
    print("loss_weight_heatmap: {}\nloss_weight_volume: {}\nnJoints: {}\nbatch_size: {}\nimg_size: {}".format(loss_weight_heatmap, loss_weight_volume, nJoints, batch_size, img_size))
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

