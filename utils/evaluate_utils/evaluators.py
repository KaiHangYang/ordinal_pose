import numpy as np
import sys
import os

sys.path.append("../")
from common_utils import my_utils

class mEvaluator3_1_gt(object):
    def __init__(self, nJoints=17):
        # section 3_1 only evaluate the depth np.abs()
        self.nJoints = nJoints
        self.avg_counter = my_utils.mAverageCounter(shape=self.nJoints)

    def add(self, gt_depth, pd_depth):
        assert(gt_depth.shape == pd_depth.shape)
        if len(gt_depth.shape) >= 2:
            gt_depth = np.reshape(gt_depth, [gt_depth.shape[0], self.nJoints])
            pd_depth = np.reshape(pd_depth, [pd_depth.shape[0], self.nJoints])
        else:
            gt_depth = gt_depth[np.newaxis]
            pd_depth = pd_depth[np.newaxis]

        for batch_num in range(gt_depth.shape[0]):
            mpje_arr = np.abs(gt_depth[batch_num] - pd_depth[batch_num])
            self.avg_counter.add(mpje_arr)

    def mean(self):
        return self.avg_counter.mean()

    def get(self):
        return self.avg_counter.cur_average

    def printMean(self):
        print("MPJE(depth / mm): {}".format(self.mean()))

    def printAll(self):
        print("MPJE(depth / mm): {}".format(self.mean()))
        print(("MPJE(depth_joints / mm): " + "{}" * self.nJoints).format(*self.get()))

    def save(self, path):
        np.save(path, {"mean": self.mean(), "all": self.get()})



class mEvaluator3_2_gt(object):
    def __init__(self, nJoints=17):
        # section 3_2 evaluate the coords distance (mpje:mm)
        self.nJoints = nJoints
        self.avg_counter = my_utils.mAverageCounter(shape=self.nJoints)

    def add(self, gt_coords, pd_coords):
        assert(gt_coords.shape == pd_coords.shape)
        gt_depth = np.reshape(gt_depth, [-1, self.nJoints, 3])
        pd_depth = np.reshape(pd_depth, [-1, self.nJoints, 3])

        for batch_num in range(gt_depth.shape[0]):
            mpje_arr = np.linalg.norm(gt_depth[batch_num] - pd_depth[batch_num], axis=1)
            self.avg_counter.add(mpje_arr)

    def mean(self):
        return self.avg_counter.mean()

    def get(self):
        return self.avg_counter.cur_average

    def printMean(self):
        print("MPJE(distance / mm): {}".format(self.mean()))

    def printAll(self):
        print("MPJE(distance / mm): {}".format(self.mean()))
        print(("MPJE(distance_joints / mm): " + "{}" * self.nJoints).format(*self.get()))

    def save(self, path):
        np.save(path, {"mean": self.mean(), "all": self.get()})
