import numpy as np
import os
import sys
import cv2

class MPIIReader(object):
    def __init__(self, img_dir, lbl_dir, is_shuffle=True, batch_size=4):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size

        if not os.path.isdir(self.lbl_dir) or not os.path.isdir(self.img_dir):
            print("{} or {} is not existing!".format(self.img_dir, self.lbl_dir))
            quit()

        self.n_labels = 0
        for cur_lbl in os.listdir(self.lbl_dir):
            if cur_lbl.split(".")[1] == "npy":
                self.n_labels += 1

        self._cur_index = 0

        self.index_arrays = np.arange(0, self.n_labels, 1)
        if self.is_shuffle:
            np.random.shuffle(self.index_arrays)

        print("DataSet frame number : {}".format(self.n_labels))

    @property
    def cur_index(self):
        return self._cur_index

    @cur_index.setter
    def cur_index(self, val):
        self._cur_index = val
        if self._cur_index < 0:
            self._cur_index = 0
        elif self._cur_index >= self.n_labels:
            self._cur_index = self.n_labels - 1

    def reset(self):
        self.cur_index = 0
        if self.is_shuffle:
            np.random.shuffle(self.index_arrays)

    def progress(self):
        return self.cur_index, self.n_labels

    # get a batch of datas
    def get(self):
        batches = []
        finished_epoch = False
        for i in range(self.batch_size):
            cur_label_num = self.index_arrays[self.cur_index]
            cur_img_path = os.path.join(self.img_dir, "{}.jpg".format(cur_label_num))
            cur_lbl_path = os.path.join(self.lbl_dir, "{}.npy".format(cur_label_num))

            batches.append((cur_img_path, cur_lbl_path))
            self.cur_index += 1

        if self.cur_index == self.n_labels - 1:
            finished_epoch = True

        return batches, finished_epoch
