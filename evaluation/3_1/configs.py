import ConfigParser
import os
import sys

#### Evaluation Parameters ####
depth_scale = None
weight_scale = None
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
restore_model_path_fn = None

##### The parameters below only used in the ordinal mode
scale_batch_size = 4
scale_range_file = None
scale_img_path_fn = None
scale_lbl_path_fn = None
###############################

# t means gt(0) or ord(1)
# d means validset(0) or trainset(1)
def parse_configs(t, d):

    global depth_scale, weight_scale, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, scale_batch_size, scale_img_path_fn, scale_lbl_path_fn, scale_range_file

    eval_type = ["gt", "ord"][t]
    data_source = ["valid", "train"][d]

    EVAL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ### The eval.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(EVAL_ROOT_DIR, "eval.conf"))

    depth_scale = 1000.0
    weight_scale = 1000.0

    nJoints = config_parser.getint("data", "nJoints")
    batch_size = config_parser.getint("data", "batch_size")
    img_size = config_parser.getint("data", "img_size")

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "eval/3_1_" + eval_type)

    # Dataset Settings
    range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), data_source + "_range.npy")
    img_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), data_source) + "/images/{}.jpg".format(x)
    lbl_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), data_source) + "/labels/{}.npy".format(x)

    # Parameters used in ordinal mode
    scale_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "scale_range.npy")
    scale_img_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), "train") + "/images/{}.jpg".format(x)
    scale_lbl_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), "train") + "/labels/{}.npy".format(x)

    restore_model_path_fn = lambda x: os.path.join(config_parser.get("model", "base_dir"), "3_1_{}/".format(eval_type) + config_parser.get("model", "prefix").format("3_1", eval_type, x))

def print_configs():
    global depth_scale, weight_scale, nJoints, batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, range_file, img_path_fn, lbl_path_fn, restore_model_path_fn, scale_batch_size, scale_img_path_fn, scale_lbl_path_fn, scale_range_file
    print("##################### Evaluation Parameters #####################")
    print("##### Data Parameters")
    print("depth_scale: {}\nweight_scale: {}\nnJoints: {}\nbatch_size: {}\nimg_size: {}".format(depth_scale, weight_scale, nJoints, batch_size, img_size))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("range_file: {}".format(range_file))
    print("img_path: {}".format(img_path_fn("{}")))
    print("lbl_path: {}".format(lbl_path_fn("{}")))

    print("scale_range_file: {}".format(scale_range_file))
    print("scale_img_path: {}".format(scale_img_path_fn("{}")))
    print("scale_lbl_path: {}".format(scale_lbl_path_fn("{}")))
    print("restore_model_path: {}".format(restore_model_path_fn("{}")))
