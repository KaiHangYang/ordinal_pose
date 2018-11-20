import ConfigParser
import os
import sys
import numpy as np

class mConfigs(object):
    def __init__(self, conf_file, prefix):
        ### The conf_file is in the same directory with this script
        config_parser = ConfigParser.SafeConfigParser()
        config_parser.read(conf_file)

        self.img_size = 256
        self.train_batch_size = config_parser.getint("data", "train_batch_size")
        self.valid_batch_size = config_parser.getint("data", "valid_batch_size")

        self.train_iter = config_parser.getint("data", "train_iter")
        self.valid_iter = config_parser.getint("data", "valid_iter")

        self.learning_rate = config_parser.getfloat("train", "learning_rate")
        self.lr_decay_rate = config_parser.getfloat("train", "lr_decay_rate")
        self.lr_decay_step = config_parser.getint("train", "lr_decay_step")

        self.log_dir = os.path.join(config_parser.get("log", "base_dir"), "train/{}".format(prefix))

        # Dataset Settings
        self.h36m_train_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "train_range.npy")
        self.h36m_valid_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "valid_range.npy")
        self.lsp_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "lsp_range.npy")
        self.mpii_range_file = os.path.join(config_parser.get("dataset", "range_file_dir"), "mpii_range.npy")

        h36m_base_image_path = config_parser.get("dataset", "h36m_image_path")
        h36m_base_label_path = config_parser.get("dataset", "h36m_label_path")
        h36m_base_mask_path = config_parser.get("dataset", "h36m_mask_path")

        self.h36m_train_img_path_fn = lambda x: (h36m_base_image_path.format("train", "{}")).format(x)
        self.h36m_train_mask_path_fn = lambda x: (h36m_base_mask_path.format("train", "{}")).format(x)
        self.h36m_train_lbl_path_fn = lambda x: (h36m_base_label_path.format("train", "{}")).format(x)
        self.h36m_valid_img_path_fn = lambda x: (h36m_base_image_path.format("valid", "{}")).format(x)
        self.h36m_valid_lbl_path_fn = lambda x: (h36m_base_label_path.format("valid", "{}")).format(x)

        _lsp_img_path = config_parser.get("dataset", "lsp_image_path")
        _lsp_lbl_path = config_parser.get("dataset", "lsp_label_path")
        self.lsp_img_path_fn = lambda x: _lsp_img_path.format(x)
        self.lsp_lbl_path_fn = lambda x: _lsp_lbl_path.format(x)

        _mpii_img_path = config_parser.get("dataset", "mpii_image_path")
        _mpii_lbl_path = config_parser.get("dataset", "mpii_label_path")
        self.mpii_img_path_fn = lambda x: _mpii_img_path.format(x)
        self.mpii_lbl_path_fn = lambda x: _mpii_lbl_path.format(x)

        self.model_dir = os.path.join(config_parser.get("model", "base_dir"), "{}".format(prefix))
        self.model_path_fn = lambda x: os.path.join(self.model_dir, "{}-{}".format(prefix, x))
        self.model_path = self.model_path_fn("")[0:-1]

    def printConfig(self):
        print("##################### Training Parameters #####################")
        print("##### Data Parameters #####")
        print("train_batch_size: {}\nvalid_batch_size: {}\nimg_size: {}\ntrain_iter: {}\nvalid_iter: {}".format(self.train_batch_size, self.valid_batch_size, self.img_size, self.train_iter, self.valid_iter))
        print("##### Learn Parameters")
        print("learning_rate: {}\nlr_decay_rate: {}\nlr_decay_step: {}".format(self.learning_rate, self.lr_decay_rate, self.lr_decay_step))
        print("log_dir: {}".format(self.log_dir))
        print("h36m_train_range_file: {}".format(self.h36m_train_range_file))
        print("h36m_train_img_path: {}".format(self.h36m_train_img_path_fn("{}")))
        print("h36m_train_lbl_path: {}".format(self.h36m_train_lbl_path_fn("{}")))
        print("h36m_train_mask_path: {}".format(self.h36m_train_mask_path_fn("{}")))

        print("h36m_valid_range_file: {}".format(self.h36m_valid_range_file))
        print("h36m_valid_img_path: {}".format(self.h36m_valid_img_path_fn("{}")))
        print("h36m_valid_lbl_path: {}".format(self.h36m_valid_lbl_path_fn("{}")))

        print("lsp_img_path: {}".format(self.lsp_img_path_fn("{}")))
        print("lsp_lbl_path: {}".format(self.lsp_lbl_path_fn("{}")))

        print("mpii_img_path: {}".format(self.mpii_img_path_fn("{}")))
        print("mpii_lbl_path: {}".format(self.mpii_lbl_path_fn("{}")))

        print("model_dir: {}".format(self.model_dir))
        print("model_path_fn: {}".format(self.model_path_fn("{}")))
