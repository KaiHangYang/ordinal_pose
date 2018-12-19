import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import math

sys.path.append("../../")
from net import syn_net
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import syn_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorFB_BR

##################### Setting for training ######################

####################### Setting the training protocols ########################
training_protocol = [
        {"prefix": "syn_net_h36m", "extra_data_scale": 0, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "syn_net_mixed-5000", "extra_data_scale": 10, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "syn_net_mixed-11000", "extra_data_scale": 5, "mpii_range_file": "mpii_range.npy"}
        ][2]
###############################################################################

configs = mConfigs("../train.conf", training_protocol["prefix"])

################ Reseting  #################
configs.loss_weight_heatmap = 1.0
configs.loss_weight_br = 1.0
configs.loss_weight_fb = 1.0
configs.pose_2d_scale = 4.0
configs.is_use_bn = True
configs.extra_data_scale = training_protocol["extra_data_scale"]

configs.n_epoches = 100
configs.learning_rate = 2.5e-4
configs.gamma = 0.1
configs.schedule = [30, 80]
configs.valid_steps = 4

configs.nFeats = 256
configs.nModules = 2
configs.nStacks = 2
configs.nRegModules = 2

configs.extra_log_dir = "../train_log/" + configs.prefix

### Use the smaller dataset to test and tune the hyper parameters

configs.h36m_train_range_file = os.path.join(configs.range_file_dir, "train_range.npy")
configs.h36m_valid_range_file = os.path.join(configs.range_file_dir, "valid_range_training.npy")
configs.mpii_range_file = os.path.join(configs.range_file_dir, training_protocol["mpii_range_file"])
configs.lsp_range_file = os.path.join(configs.range_file_dir, "lsp_range.npy")

################### Initialize the data reader ####################

configs.printConfig()
preprocessor = syn_preprocess.SynProcessor(skeleton=skeleton, img_size=configs.img_size, bone_width=6, joint_ratio=6, bg_color=0.2)

train_log_dir = os.path.join(configs.log_dir, "train")
valid_log_dir = os.path.join(configs.log_dir, "valid")

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_epoch = None
#################################################################
def get_learning_rate(configs, epoch):
    decay = 0
    for i in range(len(configs.schedule)):
        if epoch >= configs.schedule[i]:
            decay = 1 + i
    return configs.learning_rate * math.pow(configs.gamma, decay)

if __name__ == "__main__":
    ########################### Initialize the data list #############################
    train_range = np.load(configs.h36m_train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.h36m_valid_range_file)

    train_img_list = [configs.h36m_train_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.h36m_train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.h36m_valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.h36m_valid_lbl_path_fn(i) for i in valid_range]

    mpii_range = np.load(configs.mpii_range_file)
    lsp_range = np.load(configs.lsp_range_file)

    mpii_lsp_img_list = [configs.mpii_img_path_fn(i) for i in mpii_range] + [configs.lsp_img_path_fn(i) for i in lsp_range]
    mpii_lsp_lbl_list = [configs.mpii_lbl_path_fn(i) for i in mpii_range] + [configs.lsp_lbl_path_fn(i) for i in lsp_range]

    # increase the mpii_lsp datas
    mpii_lsp_img_list = mpii_lsp_img_list * configs.extra_data_scale
    mpii_lsp_lbl_list = mpii_lsp_lbl_list * configs.extra_data_scale

    train_img_list = train_img_list + mpii_lsp_img_list
    train_lbl_list = train_lbl_list + mpii_lsp_lbl_list

    ###################################################################

    train_data_reader = epoch_reader.EPOCHReader(img_path_list=train_img_list, lbl_path_list=train_lbl_list, is_shuffle=True, batch_size=configs.train_batch_size, name="Train DataSet")
    valid_data_reader = epoch_reader.EPOCHReader(img_path_list=valid_img_list, lbl_path_list=valid_lbl_list, is_shuffle=False, batch_size=configs.valid_batch_size, name="Valid DataSet")

    # now test the classification
    input_images = tf.placeholder(shape=[None, configs.img_size, configs.img_size, 3], dtype=tf.float32, name="input_images")
    input_centers_hm = tf.placeholder(shape=[None, skeleton.n_joints, 2], dtype=tf.float32, name="input_centers_hm")
    input_fb = tf.placeholder(shape=[None, skeleton.n_bones, 3], dtype=tf.float32, name="input_fb")
    input_br = tf.placeholder(shape=[None, (skeleton.n_bones-1) * skeleton.n_bones / 2, 3], dtype=tf.float32, name="input_br")

    input_is_training = tf.placeholder(shape=[], dtype=tf.bool, name="input_is_training")
    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="input_batch_size")
    input_lr = tf.placeholder(shape=[], dtype=tf.float32, name="input_lr")

    syn_model = syn_net.mSynNet(nJoints=skeleton.n_joints, img_size=configs.img_size, batch_size=input_batch_size, is_training=input_is_training, loss_weight_heatmap=configs.loss_weight_heatmap, loss_weight_fb=configs.loss_weight_fb, loss_weight_br=configs.loss_weight_br, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn)

    train_valid_counter = my_utils.mTrainValidCounter(train_steps=configs.valid_steps, valid_steps=1)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            syn_model.build_model(input_images)
            input_heatmaps = syn_model.build_input_heatmaps(input_centers_hm, stddev=1.0, gaussian_coefficient=False)

        syn_model.build_loss(input_heatmaps=input_heatmaps, input_fb=input_fb, input_br=input_br, lr=input_lr)

        train_log_writer = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
        valid_log_writer = tf.summary.FileWriter(logdir=valid_log_dir, graph=sess.graph)
        print("Network built!")

        model_saver = tf.train.Saver(max_to_keep=10)
        net_init = tf.global_variables_initializer()

        sess.run([net_init])
        # reload the model
        if restore_model_epoch is not None:
            if os.path.exists(configs.model_path_fn(restore_model_epoch)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(restore_model_epoch))
            else:
                print("The prev model is not existing!")
                quit()

        cur_train_global_steps = 0
        cur_valid_global_steps = 0

        cur_max_fb_acc = 0
        cur_max_br_acc = 0

        for cur_epoch in range(0 if restore_model_epoch is None else restore_model_epoch, configs.n_epoches):
            cur_learning_rate = get_learning_rate(configs, cur_epoch)

            #################### Train ####################
            train_relation_evaluator = mEvaluatorFB_BR(n_fb=skeleton.n_bones, n_br=(skeleton.n_bones-1) * skeleton.n_bones / 2)
            train_data_reader.reset()
            is_epoch_finished = False

            while not is_epoch_finished:
                cur_batch, is_epoch_finished = train_data_reader.get()

                batch_size = len(cur_batch)
                batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
                batch_fb_np = np.zeros([batch_size, skeleton.n_bones, 3], dtype=np.float32)
                batch_br_np = np.zeros([batch_size, (skeleton.n_bones-1) * skeleton.n_bones / 2, 3], dtype=np.float32)

                for b in range(batch_size):
                    assert(os.path.basename(cur_batch[b][0]).split(".")[0] == os.path.basename(cur_batch[b][1]).split(".")[0])

                    cur_img = cv2.imread(cur_batch[b][0])
                    cur_label = np.load(cur_batch[b][1]).tolist()

                    if "joints_3d" in cur_label.keys():
                        ##!!!!! the joints_3d is the joints under the camera coordinates !!!!!##
                        ##!!!!! the joints_2d is the cropped ones !!!!!##
                        # the h36m datas
                        cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                        cur_joints_3d = cur_label["joints_3d"].copy()[skeleton.h36m_selected_index]
                        cur_scale = cur_label["scale"]
                        cur_center = cur_label["center"]
                        cur_cam_mat = cur_label["cam_mat"]

                        cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_h36m(img=cur_img, joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, scale=cur_scale, center=cur_center, cam_mat=cur_cam_mat, is_training=True)
                    else:
                        # the mpii lsp datas
                        cur_joints_2d = cur_label["joints_2d"].copy()
                        cur_bone_status = cur_label["bone_status"].copy()
                        cur_bone_relations = cur_label["bone_relations"].copy()

                        cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_base(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=True)

                    # generate the heatmaps
                    batch_images_np[b] = cur_img
                    cur_joints_2d = cur_joints_2d / configs.pose_2d_scale

                    batch_joints_2d_np[b] = cur_joints_2d.copy()
                    #### convert the bone_status and bone_relations to one-hot representation
                    batch_fb_np[b] = np.eye(3)[cur_bone_status]
                    batch_br_np[b] = np.eye(3)[cur_bone_relations[np.triu_indices(skeleton.n_bones, k=1)]] # only get the upper triangle

                    ########## Visualize the datas ###########
                    # cv2.imshow("img", cur_img)
                    # cv2.imshow("test", display_utils.drawLines((255.0 * cur_img).astype(np.uint8), cur_joints_2d * configs.pose_2d_scale, indices=skeleton.bone_indices, color_table=skeleton.bone_colors * 255))
                    # cur_bone_order = preprocessor.bone_order_from_bone_relations(cur_bone_relations, np.ones_like(cur_bone_relations))
                    # cv2.imshow("syn_img_python_order", preprocessor.draw_syn_img(cur_joints_2d*configs.pose_2d_scale, cur_bone_status, cur_bone_order))
                    # cv2.waitKey()
                    ##########################################

                _, \
                gt_fb_result, \
                pd_fb_result, \
                gt_br_result, \
                pd_br_result, \
                acc_hm, \
                acc_fb, \
                acc_br, \
                total_loss,\
                heatmaps_loss, \
                fb_loss, \
                br_loss, \
                lr,\
                summary  = sess.run(
                        [
                         syn_model.train_op,
                         syn_model.gt_fb_result,
                         syn_model.pd_fb_result,
                         syn_model.gt_br_result,
                         syn_model.pd_br_result,
                         syn_model.heatmaps_acc,
                         syn_model.fb_acc,
                         syn_model.br_acc,
                         syn_model.total_loss,
                         syn_model.heatmaps_loss,
                         syn_model.fb_loss,
                         syn_model.br_loss,
                         syn_model.lr,
                         syn_model.merged_summary],
                        feed_dict={input_images: batch_images_np,
                                   input_centers_hm: batch_joints_2d_np,
                                   input_fb: batch_fb_np,
                                   input_br: batch_br_np,
                                   input_lr: cur_learning_rate,
                                   input_is_training: True,
                                   input_batch_size: configs.train_batch_size})
                train_log_writer.add_summary(summary, cur_train_global_steps)

                train_relation_evaluator.add(gt_fb=gt_fb_result, pd_fb=pd_fb_result, gt_br=gt_br_result, pd_br=pd_br_result)

                print("Training | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(cur_epoch, configs.n_epoches, *train_data_reader.progress()))
                print("learning_rate: {:07f}".format(lr))
                print("Heatmap pixel error: {}".format(acc_hm))
                print("Bone Status Accuracy: {:07f}".format(acc_fb))
                print("Bone Relation Accuracy: {:07f}".format(acc_br))
                print("Total loss: {:.08f}".format(total_loss))
                for l_idx in range(len(heatmaps_loss)):
                    print("Heatmap loss level {}: {}".format(l_idx, heatmaps_loss[l_idx]))
                print("Bone status loss: {:.08f}".format(fb_loss))
                print("Bone relation loss: {:.08f}".format(br_loss))

                train_relation_evaluator.printMean()
                print("\n\n")

                cur_train_global_steps += 1

            train_relation_evaluator.save(os.path.join(configs.extra_log_dir, "train"), prefix="train", epoch=cur_epoch)
            train_valid_counter.next()

            ######################## Evaluate ############################

            if not train_valid_counter.is_training:
                valid_data_reader.reset()
                valid_relation_evaluator = mEvaluatorFB_BR(n_fb=skeleton.n_bones, n_br=(skeleton.n_bones-1) * skeleton.n_bones / 2)
                is_epoch_finished = False

                while not is_epoch_finished:
                    cur_batch, is_epoch_finished = valid_data_reader.get()

                    batch_size = len(cur_batch)
                    batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                    batch_joints_2d_np = np.zeros([batch_size, skeleton.n_joints, 2], dtype=np.float32)
                    batch_fb_np = np.zeros([batch_size, skeleton.n_bones, 3], dtype=np.float32)
                    batch_br_np = np.zeros([batch_size, (skeleton.n_bones-1) * skeleton.n_bones / 2, 3], dtype=np.float32)

                    for b in range(batch_size):
                        assert(os.path.basename(cur_batch[b][0]).split(".")[0] == os.path.basename(cur_batch[b][1]).split(".")[0])

                        cur_img = cv2.imread(cur_batch[b][0])
                        cur_label = np.load(cur_batch[b][1]).tolist()

                        if "joints_3d" in cur_label.keys():
                            ##!!!!! the joints_3d is the joints under the camera coordinates !!!!!##
                            ##!!!!! the joints_2d is the cropped ones !!!!!##
                            # the h36m datas
                            cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                            cur_joints_3d = cur_label["joints_3d"].copy()[skeleton.h36m_selected_index]
                            cur_scale = cur_label["scale"]
                            cur_center = cur_label["center"]
                            cur_cam_mat = cur_label["cam_mat"]

                            cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_h36m(img=cur_img, joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, scale=cur_scale, center=cur_center, cam_mat=cur_cam_mat, is_training=False)
                        else:
                            # the mpii lsp datas
                            cur_joints_2d = cur_label["joints_2d"].copy()
                            cur_bone_status = cur_label["bone_status"].copy()
                            cur_bone_relations = cur_label["bone_relations"].copy()

                            cur_img, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_base(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=False)

                        # generate the heatmaps
                        batch_images_np[b] = cur_img
                        cur_joints_2d = cur_joints_2d / configs.pose_2d_scale

                        batch_joints_2d_np[b] = cur_joints_2d.copy()
                        #### convert the bone_status and bone_relations to one-hot representation
                        batch_fb_np[b] = np.eye(3)[cur_bone_status]
                        batch_br_np[b] = np.eye(3)[cur_bone_relations[np.triu_indices(skeleton.n_bones, k=1)]] # only get the upper triangle


                    gt_fb_result, \
                    pd_fb_result, \
                    gt_br_result, \
                    pd_br_result, \
                    acc_hm, \
                    acc_fb, \
                    acc_br, \
                    total_loss,\
                    heatmaps_loss, \
                    fb_loss, \
                    br_loss, \
                    lr,\
                    summary  = sess.run(
                            [
                             syn_model.gt_fb_result,
                             syn_model.pd_fb_result,
                             syn_model.gt_br_result,
                             syn_model.pd_br_result,
                             syn_model.heatmaps_acc,
                             syn_model.fb_acc,
                             syn_model.br_acc,
                             syn_model.total_loss,
                             syn_model.heatmaps_loss,
                             syn_model.fb_loss,
                             syn_model.br_loss,
                             syn_model.lr,
                             syn_model.merged_summary],
                            feed_dict={input_images: batch_images_np,
                                       input_centers_hm: batch_joints_2d_np,
                                       input_fb: batch_fb_np,
                                       input_br: batch_br_np,
                                       input_lr: cur_learning_rate,
                                       input_is_training: False,
                                       input_batch_size: configs.valid_batch_size})
                    valid_log_writer.add_summary(summary, cur_valid_global_steps)

                    valid_relation_evaluator.add(gt_fb=gt_fb_result, pd_fb=pd_fb_result, gt_br=gt_br_result, pd_br=pd_br_result)

                    print("Validing | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(cur_epoch, configs.n_epoches, *valid_data_reader.progress()))
                    print("learning_rate: {:07f}".format(lr))
                    print("Heatmap pixel error: {}".format(acc_hm))
                    print("Bone Status Accuracy: {:07f}".format(acc_fb))
                    print("Bone Relation Accuracy: {:07f}".format(acc_br))
                    print("Total loss: {:.08f}".format(total_loss))
                    for l_idx in range(len(heatmaps_loss)):
                        print("Heatmap loss level {}: {}".format(l_idx, heatmaps_loss[l_idx]))
                    print("Bone status loss: {:.08f}".format(fb_loss))
                    print("Bone relation loss: {:.08f}".format(br_loss))

                    valid_relation_evaluator.printMean()
                    print("\n\n")

                    cur_valid_global_steps += 1

                valid_relation_evaluator.save(os.path.join(configs.extra_log_dir, "valid"), prefix="valid", epoch=cur_epoch)
                valid_fb_acc, valid_br_acc = valid_relation_evaluator.mean()

                if cur_max_fb_acc < valid_fb_acc and cur_max_br_acc < valid_br_acc:
                    cur_max_fb_acc = valid_fb_acc
                    cur_max_br_acc = valid_br_acc

                    #### Only save the higher score models
                    with open(os.path.join(configs.model_dir, "best_model.txt"), "w") as f:
                        f.write("{}".format(cur_epoch))

                    model_saver.save(sess=sess, save_path=configs.model_path, global_step=cur_epoch)