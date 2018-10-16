import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import sys
import time

sys.path.append("../")
from net import vnect_out

batch_size = 4
nJoints = 17
feature_size = 64
test_iter = 1000

if __name__ == "__main__":
    ############################### Check the joints_extractor ################################
    #### cpu implementation is fast
    # input_xyzmaps = tf.placeholder(shape=[batch_size, feature_size, feature_size, nJoints * 3], dtype=tf.float32)
    # input_joints_2d = tf.placeholder(shape=[batch_size, nJoints, 2], dtype=tf.float32)

    # net_model = vnect_out.mVNectOutput(nJoints, False, batch_size)

    # joints_xyz = net_model.get_joints_xyzm(input_joints_2d, input_xyzmaps, batch_size)

    # with tf.Session() as sess:
        # for _ in range(test_iter):
            # xyzmaps = np.random.random([batch_size, feature_size, feature_size, nJoints*3])
            # joints_2d = np.round(np.random.random([batch_size, nJoints, 2]) * (feature_size - 1))

            # joints_2d_int = joints_2d.astype(np.int32)

            # joints_3d_cpu = np.zeros([batch_size, nJoints, 3], dtype=np.float32)

            # cpu_time = time.clock()

            # for b in range(batch_size):
                # for j in range(nJoints):
                    # joints_3d_cpu[b, j, 0] = xyzmaps[b, joints_2d_int[b, j, 1], joints_2d_int[b, j, 0], j + 0*nJoints]
                    # joints_3d_cpu[b, j, 1] = xyzmaps[b, joints_2d_int[b, j, 1], joints_2d_int[b, j, 0], j + 1*nJoints]
                    # joints_3d_cpu[b, j, 2] = xyzmaps[b, joints_2d_int[b, j, 1], joints_2d_int[b, j, 0], j + 2*nJoints]

            # cpu_time = time.clock() - cpu_time

            # gpu_time = time.clock()
            # joints_3d_gpu = sess.run(joints_xyz, feed_dict={input_xyzmaps: xyzmaps, input_joints_2d: joints_2d})
            # gpu_time = time.clock() - gpu_time

            # assert((joints_3d_cpu == joints_3d_gpu).all())

            # print("GPU time: {}, CPU time: {}".format(gpu_time, cpu_time))
    ##########################################################################################

    ############################ Test the generation scripts of xyzmaps ##############################
    ######### GPU version is faster
    input_joints_3d = tf.placeholder(shape=[batch_size, nJoints, 3], dtype=tf.float32)

    net_model = vnect_out.mVNectOutput(nJoints, False, batch_size)
    xyzmaps = net_model.build_input_xyzmaps(input_joints_3d, batch_size)

    with tf.Session() as sess:
        for _ in range(test_iter):
            joints_3d = np.random.random([batch_size, nJoints, 3])

            cpu_time = time.clock()
            cpu_xyzmaps = np.ones([batch_size, feature_size, feature_size, nJoints*3], dtype=np.float32)

            for b in range(batch_size):
                for j in range(nJoints):
                    cpu_xyzmaps[b, :, :, j + 0 * nJoints] *= joints_3d[b, j, 0]
                    cpu_xyzmaps[b, :, :, j + 1 * nJoints] *= joints_3d[b, j, 1]
                    cpu_xyzmaps[b, :, :, j + 2 * nJoints] *= joints_3d[b, j, 2]

            cpu_time = time.clock() - cpu_time

            gpu_time = time.clock()
            gpu_xyzmaps = sess.run(xyzmaps, feed_dict={input_joints_3d: joints_3d})
            gpu_time = time.clock() - gpu_time

            assert((gpu_xyzmaps == cpu_xyzmaps).all())
            print("GPU time: {}, CPU time: {}".format(gpu_time, cpu_time))
