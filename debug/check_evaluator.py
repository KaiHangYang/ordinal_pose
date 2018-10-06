import numpy as np
import os
import sys

sys.path.append("../")

from utils.evaluate_utils import evaluators

eval_iterations = 100000
nJoints = 17

if __name__ == "__main__":
    eval_depth = evaluators.mEvaluatorDepth(nJoints)
    eval_coords = evaluators.mEvaluatorPose3D(nJoints)


    gt_depth = np.random.random([eval_iterations, nJoints]) * 100
    pd_depth = np.random.random([eval_iterations, nJoints]) * 100

    mean_depth = np.mean(np.abs(gt_depth - pd_depth))

    gt_coords = np.random.random([eval_iterations, nJoints, 3]) * 100
    pd_coords = np.random.random([eval_iterations, nJoints, 3]) * 100

    mean_coords = np.mean(np.sqrt(np.sum((gt_coords - pd_coords)**2, axis=2)))

    for i in range(eval_iterations):
        eval_depth.add(gt_depth[i], pd_depth[i])
        eval_coords.add(gt_coords[i], pd_coords[i])

    print("mean_depth({} | {}), mean_coords({} | {})".format(mean_depth, eval_depth.mean(), mean_coords, eval_coords.mean()))
