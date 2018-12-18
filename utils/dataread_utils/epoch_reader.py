import numpy as np
import os
import sys
import cv2

class EPOCHReader(object):
    def __init__(self, img_path_list, lbl_path_list, is_shuffle=True, batch_size=4, name="Train DataSet"):
        self.name = name
        self.img_path_list = img_path_list
        self.lbl_path_list = lbl_path_list
        assert(len(self.img_path_list) == len(self.lbl_path_list))
        self.index_arrays = np.arange(0, len(self.img_path_list), 1)

        self.only_label = False

        if self.img_path_list is None:
            # only labels
            self.only_label = True

        self.is_shuffle = is_shuffle
        self.batch_size = batch_size

        if not os.path.isdir(os.path.dirname(self.lbl_path_list[0])) or (not self.only_label and not os.path.isdir(os.path.dirname(self.img_path_list[0]))):
            print("img_dir or lbl_dir is not existing!")
            quit()

        self.n_labels = len(self.index_arrays)
        self._cur_index = 0

        if self.is_shuffle:
            np.random.shuffle(self.index_arrays)

        print("{} frame number : {}".format(name, self.n_labels))

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

            if self.only_label:
                cur_lbl_path = self.lbl_path_list[cur_label_num]
                batches.append((cur_lbl_path))
            else:
                cur_img_path = self.img_path_list[cur_label_num]
                cur_lbl_path = self.lbl_path_list[cur_label_num]
                batches.append((cur_img_path, cur_lbl_path))

            self.cur_index += 1

        if self.cur_index == self.n_labels - 1:
            finished_epoch = True

        return batches, finished_epoch
