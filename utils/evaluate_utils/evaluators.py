import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import my_utils

########## Evaluation used only for evaluate depth ##########
class mEvaluatorDepth(object):
    def __init__(self, nJoints=17):
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
        print("MPJE(depth / mm): {}. Frame sum: {}".format(self.mean(), self.avg_counter.cur_data_sum))

    def printAll(self):
        print("MPJE(depth / mm): {}. Frame sum: {}".format(self.mean(), self.avg_counter.cur_data_sum))
        print(("MPJE(depth_joints / mm): " + "{}" * self.nJoints).format(*self.get()))

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        np.save(path, {"mean": self.mean(), "all": self.get(), "frame_sum": self.avg_counter.cur_data_sum})


############### Evaluator for Pose3D ############
class mEvaluatorPose3D(object):
    def __init__(self, nJoints=17):
        self.nJoints = nJoints
        self.avg_counter = my_utils.mAverageCounter(shape=self.nJoints)

    ### The root of each coords is the same
    def add(self, gt_coords, pd_coords):
        assert(gt_coords.shape == pd_coords.shape)
        gt_coords = np.reshape(gt_coords, [-1, self.nJoints, 3])
        pd_coords = np.reshape(pd_coords, [-1, self.nJoints, 3])

        for batch_num in range(gt_coords.shape[0]):
            mpje_arr = np.linalg.norm(gt_coords[batch_num] - pd_coords[batch_num], axis=1)
            self.avg_counter.add(mpje_arr)

    def mean(self):
        return self.avg_counter.mean()

    def get(self):
        return self.avg_counter.cur_average

    def printMean(self):
        print("MPJE(distance / mm): {}. Frame sum: {}".format(self.mean(), self.avg_counter.cur_data_sum))

    def printAll(self):
        print("MPJE(distance / mm): {}. Frame sum: {}".format(self.mean(), self.avg_counter.cur_data_sum))
        print(("MPJE(distance_joints / mm): " + "{}" * self.nJoints).format(*self.get()))

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.save(path, {"mean": self.mean(), "all": self.get(), "frame_sum": self.avg_counter.cur_data_sum})

############### Evaluator for BR and FB accuracy ############
class mEvaluatorFB_BR(object):
    def __init__(self, nData=16):
        self.nData = nData
        self.avg_counter = my_utils.mAverageCounter(shape=self.nData)

    ### The root of each data is the same
    def add(self, gt_info, pd_info):
        assert(gt_info.shape == pd_info.shape)
        gt_info = np.reshape(gt_info, [-1, self.nData])
        pd_info = np.reshape(pd_info, [-1, self.nData])

        for batch_num in range(gt_info.shape[0]):
            self.avg_counter.add((gt_info[batch_num] == pd_info[batch_num]).astype(np.float32))

    def mean(self):
        return self.avg_counter.mean()

    def get(self):
        return self.avg_counter.cur_average

    def printMean(self):
        print("Mean accuracy: {}. Frame sum: {}".format(self.mean(), self.avg_counter.cur_data_sum))

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.save(path, {"mean": self.mean(), "all": self.get(), "frame_sum": self.avg_counter.cur_data_sum})
