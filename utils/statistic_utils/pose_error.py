import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class mResultSaver(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.results = None

    def restore(self):
        self.results = np.load(self.save_path).tolist()

    def save(self):
        if self.results is not None:
            np.save(os.path.splitext(self.save_path)[0], self.results)

    # Notice: the pose is related to the root
    # and pose_gt, pose_pd shape is (nJoint, 3)
    def add_one(self, data_index, pose_gt, pose_pd, network_output):
        if self.results is None:
            self.results = []

        # calculate the error
        error_per_joint = np.sqrt(np.sum((pose_gt - pose_pd) ** 2, axis=1))
        error_mean = np.mean(error_per_joint)

        cur_result = {
                "index": data_index,
                "pose_gt": pose_gt,
                "pose_pd": pose_pd,
                "network_output": network_output,
                "error_per": error_per_joint,
                "error_mean": error_mean
                }

        self.results.append(cur_result)
