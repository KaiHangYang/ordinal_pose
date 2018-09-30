import ConfigParser
import os
import sys

#### Evaluation Parameters ####
global coords_scale
global weight_scale
global nJoints
global batch_size
global img_size
global learning_rate
global lr_decay_rate
global lr_decay_step
global log_dir
global range_file
global img_path_fn
global lbl_path_fn
global restore_model_path_fn
###############################

# t means gt(0) or ord(1)
# d means validset(0) or trainset(1)
def parse_configs(t, d):
    global coords_scale, weight_scale, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn

    eval_type = ["gt", "ord"][t]
    data_source = ["valid", "train"][d]

    EVAL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ### The eval.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(EVAL_ROOT_DIR, "eval.conf"))

    coords_scale = 1000.0
    weight_scale = 1000.0

    nJoints = config_parser.getint("data", "nJoints")
    batch_size = config_parser.getint("data", "batch_size")
    img_size = config_parser.getint("data", "img_size")

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "eval/3_2_" + eval_type)

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source + "_range.npy")
    img_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), data_source) + "/images/{}.jpg".format(x)
    lbl_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), data_source) + "/labels/{}.npy".format(x)

    restore_model_path_fn = lambda x: os.path.join(config_parser.get("model", "base_dir"), config_parser.get("model", "prefix").format("3_2", eval_type, x))

def print_configs():
    global coords_scale, weight_scale, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn
    print("##################### Evaluation Parameters #####################")
    print("##### Data Parameters")
    print("coords_scale: {}\nweight_scale: {}\nnJoints: {}\nbatch_size: {}\nimg_size: {}".format(coords_scale, weight_scale, nJoints, batch_size, img_size))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("range_file: {}".format(range_file))
    print("img_path: {}".format(img_path_fn("{}")))
    print("lbl_path: {}".format(lbl_path_fn("{}")))
    print("restore_model_path: {}".format(restore_model_path_fn("{}")))