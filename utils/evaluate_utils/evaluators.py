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
        self.dist_avg_counter = my_utils.mAverageCounter(shape=self.nData)

    ### The root of each data is the same
    def add(self, gt_info, pd_info):
        assert(gt_info.shape == pd_info.shape)
        gt_info = np.reshape(gt_info, [-1, self.nData])
        pd_info = np.reshape(pd_info, [-1, self.nData])

        for batch_num in range(gt_info.shape[0]):
            self.avg_counter.add((gt_info[batch_num] == pd_info[batch_num]).astype(np.float32))
            self.dist_avg_counter.add(np.abs(gt_info[batch_num] - pd_info[batch_num]))

    def mean(self):
        return self.avg_counter.mean(), self.dist_avg_counter.mean()

    def get(self):
        return self.avg_counter.cur_average, self.dist_avg_counter.cur_average

    def printMean(self):
        acc_mean, dist_mean = self.mean()
        print("Mean Accuracy: {}, Mean Distance: {}. Frame sum: {}".format(acc_mean, dist_mean, self.avg_counter.cur_data_sum))

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        acc_mean, dist_mean = self.mean()
        acc_all, dist_all = self.get()
        np.save(path, {"mean_acc": acc_mean, "mean_dist": dist_mean, "all_acc": acc_all, "all_dist": dist_all, "frame_sum": self.avg_counter.cur_data_sum})

############## Evaluator for 2D PCKh ###############
""" Here the head length is estimated by the head and throat joints """
class mEvaluatorPCKh(object):
    def __init__(self, n_joints, head_indices, data_range):
        self.n_joints = n_joints
        self.head_indices = head_indices
        self.data_range = data_range

        self.frame_sum = 0
        self.pck_counter = np.zeros([len(self.data_range), self.n_joints])

    def get_head_length(self, gt_2d):
        head_size = np.linalg.norm(gt_2d[:, self.head_indices[0]] - gt_2d[:, self.head_indices[1]], axis=1)
        return head_size

    def add(self, gt_2d, pd_2d):
        assert(gt_2d.shape == pd_2d.shape)

        gt_2d = np.reshape(gt_2d, [-1, self.n_joints, 2])
        pd_2d = np.reshape(pd_2d, [-1, self.n_joints, 2])

        cur_head_size = self.get_head_length(gt_2d)[:, np.newaxis]
        dist_2d = np.linalg.norm(gt_2d - pd_2d, axis=2) / cur_head_size

        for idx in range(len(self.data_range)):
            self.pck_counter[idx] = self.pck_counter[idx] + np.sum((dist_2d <= self.data_range[idx]).astype(np.int32), axis=0)

        self.frame_sum += gt_2d.shape[0]

    def mean(self):
        mean_per_pckh = self.pck_counter / self.frame_sum
        mean_pckh = np.mean(mean_per_pckh, axis=1)

        return mean_per_pckh, mean_pckh

    def printMean(self):
        mean_per_pckh, mean_pckh = self.mean()
        print("Frame sum: {}".format(self.frame_sum))
        for idx in range(len(self.data_range)):
            print("Threshhold: {}, PCKh: {}. ".format(self.data_range[idx], mean_pckh[idx]))

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        mean_per_pckh, mean_pckh = self.mean()
        np.save(path, {"mean_per_pckh": mean_per_pckh, "mean_pckh": mean_pckh, "frame_sum": self.frame_sum})
