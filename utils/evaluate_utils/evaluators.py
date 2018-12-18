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
    def __init__(self, n_fb, n_br):
        self.n_fb = n_fb
        self.n_br = n_br

        self.fb_avg_counter = my_utils.mAverageCounter(shape=self.n_fb)
        self.br_avg_counter = my_utils.mAverageCounter(shape=self.n_br)

    def add(self, gt_fb, pd_fb, gt_br, pd_br):
        assert(gt_fb.shape == pd_fb.shape and gt_br.shape == pd_br.shape)
        gt_fb = np.reshape(gt_fb, [-1, self.n_fb])
        pd_fb = np.reshape(pd_fb, [-1, self.n_fb])

        gt_br = np.reshape(gt_br, [-1, self.n_br])
        pd_br = np.reshape(pd_br, [-1, self.n_br])

        assert(gt_fb.shape[0] == gt_br.shape[0])

        for batch_num in range(gt_fb.shape[0]):
            self.fb_avg_counter.add((gt_fb[batch_num] == pd_fb[batch_num]).astype(np.float32))
            self.br_avg_counter.add((gt_br[batch_num] == pd_br[batch_num]).astype(np.float32))

    def mean(self):
        return self.fb_avg_counter.mean(), self.br_avg_counter.mean()

    def get(self):
        return self.fb_avg_counter.cur_average, self.br_avg_counter.cur_average

    def printMean(self):
        fb_acc, br_acc = self.mean()
        print("Mean Bone Status Acc: {:05f}. Mean Bone Relation Acc: {:05f}".format(fb_acc, br_acc))

    def save(self, save_dir, prefix, epoch):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fb_acc, br_acc = self.mean()

        data_file = os.path.join(save_dir, "{}-{}.npy".format(prefix, epoch))
        np.save(data_file, {"fb_acc": fb_acc, "br_acc": br_acc, "fb_per_acc":self.fb_avg_counter.cur_average, "br_per_acc":self.br_avg_counter.cur_average, "frame_sum": self.br_avg_counter.cur_data_sum})

        log_file = os.path.join(save_dir, "{}-log.txt".format(prefix))
        with open(log_file, "aw") as f:
            f.write(("Epoch: {:05d} | Bone Status Acc: {:05f} | Bone Relation Acc: {:05f}\n").format(epoch, fb_acc, br_acc))

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

class mEvaluatorPCK(object):
    def __init__(self, skeleton, data_range):
        self.skeleton = skeleton
        self.n_joints = self.skeleton.n_joints
        self.data_range = data_range

        self.frame_sum = 0
        self.pck_counter = np.zeros([len(self.data_range), self.n_joints])
        self.valid_counter = np.zeros([self.n_joints])

    def add(self, gt_2d, pd_2d, norm):
        assert(gt_2d.shape == pd_2d.shape)

        gt_2d = np.reshape(gt_2d, [-1, self.n_joints, 2])
        pd_2d = np.reshape(pd_2d, [-1, self.n_joints, 2])

        ### Calculate the distance
        dist_2d = np.zeros([gt_2d.shape[0], gt_2d.shape[1]])
        for b in range(dist_2d.shape[0]):
            for j in range(dist_2d.shape[1]):
                # The valid labels
                if (gt_2d[b][j] > 0).all():
                    dist_2d[b][j] = np.linalg.norm(gt_2d[b][j] - pd_2d[b][j]) / float(norm)
                    self.valid_counter[j] += 1
                else:
                    dist_2d[b][j] = -1

        for idx in range(len(self.data_range)):
            # only count the valid labels in ground true label (not (0, 0))
            self.pck_counter[idx] = self.pck_counter[idx] + np.sum((np.logical_and(dist_2d <= self.data_range[idx], dist_2d >= 0)).astype(np.int32), axis=0)
        self.frame_sum += gt_2d.shape[0]

    def mean(self):
        # Now only print the mean result
        # total mean
        total_mean = np.sum(self.pck_counter, axis=1) / np.sum(self.valid_counter, axis=0)
        score_mean = np.sum(self.pck_counter[:, self.skeleton.score_joints_idx], axis=1) / np.sum(self.valid_counter[self.skeleton.score_joints_idx], axis=0)
        return score_mean, total_mean

    def printMean(self):
        score_mean, total_mean = self.mean()
        for idx in range(len(self.data_range)):
            print("Threshhold: {:0.2f}, All PCK: {:0.5f}, Score PCK: {:0.5f}".format(self.data_range[idx], total_mean[idx], score_mean[idx]))

    def save(self, save_dir, prefix, epoch):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        score_mean, total_mean = self.mean()

        data_file = os.path.join(save_dir, "{}-{}.npy".format(prefix, epoch))
        np.save(data_file, {"total_mean": total_mean, "score_mean": score_mean, "valid_counter": self.valid_counter, "pck_counter": self.pck_counter})

        log_file = os.path.join(save_dir, "{}-log.txt".format(prefix))
        with open(log_file, "aw") as f:
            f.write(("Epoch: {:05d} | Score pck: ("+", ".join(["{:0.2f}".format(i) + ":{:0.5f}" for i in self.data_range])+") | All pck: ("+", ".join(["{:0.2f}".format(i) + ":{:0.5f}" for i in self.data_range])+")\n").format(epoch, *(score_mean.tolist()+total_mean.tolist())))
