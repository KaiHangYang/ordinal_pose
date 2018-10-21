import ConfigParser
import os
import sys

#### Training Parameters ####
coords_scale = None
coords_2d_scale = None
coords_2d_offset = None

nJoints = None
train_batch_size = None
valid_batch_size = None
img_size = None
learning_rate = None
lr_decay_rate = None
lr_decay_step = None
log_dir = None
train_range_file = None
train_img_path_fn = None
train_lbl_path_fn = None
valid_range_file = None
valid_img_path_fn = None
valid_lbl_path_fn = None
model_dir = None
model_path_fn = None
model_path = None
train_iter = None
valid_iter = None

###############################

# t means gt(0) or ord(1)
def parse_configs(t):

    global coords_scale, coords_2d_scale, coords_2d_offset, nJoints, train_batch_size, valid_batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, train_range_file, valid_range_file, train_img_path_fn, train_lbl_path_fn, valid_img_path_fn, valid_lbl_path_fn, model_path_fn, model_dir, model_path, valid_iter, train_iter 
    train_type = ["gt", "ord"][t]

    TRAIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ### The train.conf is in the same directory with this script
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read(os.path.join(TRAIN_ROOT_DIR, "train.conf"))

    coords_scale = 1000.0
    coords_2d_scale = 255.0
    coords_2d_offset = 0
    # coords_2d_scale = 1.0

    nJoints = config_parser.getint("data", "nJoints")
    train_batch_size = config_parser.getint("data", "train_batch_size")
    valid_batch_size = config_parser.getint("data", "valid_batch_size")
    img_size = config_parser.getint("data", "img_size")
    train_iter = config_parser.getint("data", "train_iter")
    valid_iter = config_parser.getint("data", "valid_iter")

    learning_rate = config_parser.getfloat("train", "learning_rate")
    lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
    lr_decay_step = config_parser.getint("train", "lr_decay_step")

    log_dir = os.path.join(config_parser.get("log", "base_dir"), "train/3_2_" + train_type)

    # Dataset Settings
    train_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "train_range.npy")
    valid_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "valid_range.npy")

    train_img_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), "train") + "/images_syn/{}.jpg".format(x)
    train_lbl_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), "train") + "/labels/{}.npy".format(x)

    valid_img_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), "valid") + "/images_syn/{}.jpg".format(x)
    valid_lbl_path_fn = lambda x: os.path.join(config_parser.get("dataset", "base_dir"), "valid") + "/labels/{}.npy".format(x)

    model_dir = os.path.join(config_parser.get("model", "base_dir"), "3_2_{}".format(train_type))
    model_path_fn = lambda x: os.path.join(model_dir, config_parser.get("model", "prefix").format("3_2", train_type, x))

    # remove the '-'
    model_path = model_path_fn("")[0:-1]

def print_configs():
    global coords_scale, coords_2d_scale, coords_2d_offset, nJoints, train_batch_size, valid_batch_size, img_size, learning_rate, lr_decay_rate, lr_decay_step, log_dir, train_range_file, valid_range_file, train_img_path_fn, train_lbl_path_fn, valid_img_path_fn, valid_lbl_path_fn, model_path_fn, model_dir, model_path, valid_iter, train_iter
    print("##################### Training Parameters #####################")
    print("##### Data Parameters")
    print("coords_scale: {}\ncoords_2d_scale: {}\ncoords_2d_offset: {}\nnJoints: {}\ntrain_batch_size: {}\nvalid_batch_size: {}\nimg_size: {}\ntrain_iter: {}\nvalid_iter: {}".format(coords_scale, coords_2d_scale, coords_2d_offset, nJoints, train_batch_size, valid_batch_size, img_size, train_iter, valid_iter))
    print("##### Learn Parameters")
    print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(learning_rate, lr_decay_rate, lr_decay_step))
    print("log_dir: {}".format(log_dir))
    print("train_range_file: {}".format(train_range_file))
    print("train_img_path: {}".format(train_img_path_fn("{}")))
    print("train_lbl_path: {}".format(train_lbl_path_fn("{}")))

    print("valid_range_file: {}".format(valid_range_file))
    print("valid_img_path: {}".format(valid_img_path_fn("{}")))
    print("valid_lbl_path: {}".format(valid_lbl_path_fn("{}")))

    print("model_dir: {}".format(model_dir))
    print("model_path_fn: {}".format(model_path_fn("{}")))
    print("model_path: {}".format(model_path))

