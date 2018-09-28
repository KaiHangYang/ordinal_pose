import numpy as np
import os

class mRangeVariable(object):
    def __init__(self, min_val, max_val, initial_val = 0):
        self.min_val = min_val
        self.max_val = max_val
        self._val = initial_val
        self.end = False

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val
        self.end = False
        if self._val > self.max_val:
            self._val = self.max_val
            self.end = True
        elif self._val < self.min_val:
            self._val = self.min_val

    def isEnd(self):
        return self.end

class mAverageCounter(object):
    def __init__(self, shape=[]):
        self.cur_data_sum = 0
        self.cur_average = np.zeros(shape=shape)

    def add(self, one):
        self.cur_average = ((self.cur_data_sum * self.cur_average) + one ) / (self.cur_data_sum + 1)
        self.cur_data_sum += 1

    def mean(self, axis=0):
        return np.mean(self.cur_average, axis=axis)

# list all the files in the directorys
def list_all_files(root_dir):
    file_list = []
    dir_list = []

    for item in os.listdir(root_dir):
        cur_path = os.path.join(root_dir, item)
        if os.path.isdir(cur_path):
            dir_list.append(cur_path)
        elif os.path.isfile(cur_path):
            file_list.append(cur_path)

    for child_dir in dir_list:
        file_list += list_all_files(child_dir)

    return file_list

