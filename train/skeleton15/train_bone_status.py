import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import cv2
import time
import math

sys.path.append("../../")
from keras_net import relation_net
from utils.dataread_utils import epoch_reader
from utils.preprocess_utils import syn_preprocess
from utils.visualize_utils import display_utils
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton
from utils.common_utils import my_utils
from utils.evaluate_utils.evaluators import mEvaluatorRelation

####################### Setting the training protocols ########################
training_protocol = [
        {"prefix": "fb_h36m", "extra_data_scale": 0, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "fb_mixed-5000", "extra_data_scale": 10, "mpii_range_file": "mpii_range_3000.npy"},
        {"prefix": "fb_mixed-11000", "extra_data_scale": 3, "mpii_range_file": "mpii_range.npy"}
        ][0]
###############################################################################
configs = mConfigs("../train.conf", training_protocol["prefix"])

################ Reseting  #################
configs.relation_name = "Bone Status"
configs.pose_2d_scale = 4.0
configs.extra_data_scale = training_protocol["extra_data_scale"]

configs.n_epoches = 100
configs.learning_rate = 2.5e-4
configs.gamma = 0.1
configs.schedule = [30, 80]
configs.batch_size = 4

configs.n_relations = skeleton.n_bones

configs.extra_log_dir = "../train_log/" + configs.prefix

### Use the smaller dataset to test and tune the hyper parameters
configs.h36m_train_range_file = os.path.join(configs.range_file_dir, "train_range.npy")
configs.h36m_valid_range_file = os.path.join(configs.range_file_dir, "valid_range_training.npy")
configs.mpii_range_file = os.path.join(configs.range_file_dir, training_protocol["mpii_range_file"])
configs.lsp_range_file = os.path.join(configs.range_file_dir, "lsp_range.npy")

################### Initialize the data reader ####################
configs.printConfig()
preprocessor = syn_preprocess.SynProcessor(skeleton=skeleton, img_size=configs.img_size, bone_width=6, joint_ratio=6, bg_color=0.2)

if not os.path.exists(configs.model_dir):
    os.makedirs(configs.model_dir)

restore_model_epoch = 0
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
    train_data_reader = epoch_reader.EPOCHReader(img_path_list=train_img_list, lbl_path_list=train_lbl_list, is_shuffle=True, batch_size=configs.batch_size, name="Train DataSet")
    valid_data_reader = epoch_reader.EPOCHReader(img_path_list=valid_img_list, lbl_path_list=valid_lbl_list, is_shuffle=False, batch_size=configs.batch_size, name="Valid DataSet")

    model = relation_net.mRelationNet(img_size=configs.img_size, batch_size=configs.batch_size, skeleton=skeleton, n_relations=configs.n_relations, name="bone_status_net")
    model.build()
    model.build_loss(configs.learning_rate)

    cur_train_global_steps = 0
    cur_valid_global_steps = 0

    cur_max_acc = 0

    if restore_model_epoch is not None:
        model.restore_model(configs.model_path_fn(restore_model_epoch))

    for cur_epoch in range(0 if restore_model_epoch is None else restore_model_epoch, configs.n_epoches):
        cur_learning_rate = get_learning_rate(configs, cur_epoch)
        model.set_lr(cur_learning_rate)

        #################### Train ####################
        train_accuracy_evaluator = mEvaluatorRelation(n_relations=configs.n_relations, batch_size=configs.batch_size, name=configs.relation_name)
        train_data_reader.reset()
        is_epoch_finished = False

        while not is_epoch_finished:
            cur_batch, is_epoch_finished = train_data_reader.get()

            batch_size = len(cur_batch)
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_fb_np = np.zeros([batch_size, configs.n_relations, 3], dtype=np.float32)

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
                #### convert the bone_status and bone_relations to one-hot representation
                batch_fb_np[b] = np.eye(3)[cur_bone_status]

            loss, accuracy = model.train_on_batch(x=batch_images_np, y=batch_fb_np)
            lr = model.get_lr()

            train_accuracy_evaluator.add(pred_mean=accuracy)
            print("Training | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(cur_epoch, configs.n_epoches, *train_data_reader.progress()))
            print("learning_rate: {:07f}".format(lr))
            print("Loss: {:07f}".format(loss))
            print("Current Accuracy: {}".format(accuracy))
            train_accuracy_evaluator.printMean()
            print("\n\n")
            cur_train_global_steps += 1

        train_accuracy_evaluator.save(os.path.join(configs.extra_log_dir, "train"), "train", cur_epoch)

        ######################## Evaluate ############################

        valid_data_reader.reset()
        is_epoch_finished = False
        valid_accuracy_evaluator = mEvaluatorRelation(n_relations=configs.n_relations, batch_size=configs.batch_size, name=configs.relation_name)

        while not is_epoch_finished:
            cur_batch, is_epoch_finished = valid_data_reader.get()

            batch_size = len(cur_batch)
            batch_images_np = np.zeros([batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
            batch_fb_np = np.zeros([batch_size, configs.n_relations, 3], dtype=np.float32)

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
                #### convert the bone_status and bone_relations to one-hot representation
                batch_fb_np[b] = np.eye(3)[cur_bone_status]

            loss, accuracy = model.test_on_batch(x=batch_images_np, y=batch_fb_np)
            lr = model.get_lr()

            valid_accuracy_evaluator.add(pred_mean=accuracy)
            print("Validing | Epoch: {:05d}/{:05d}. Iteration: {:05d}/{:05d}".format(cur_epoch, configs.n_epoches, *valid_data_reader.progress()))
            print("learning_rate: {:07f}".format(lr))
            print("Loss: {:07f}".format(loss))
            print("Current Accuracy: {}".format(accuracy))
            valid_accuracy_evaluator.printMean()
            print("\n\n")
            cur_valid_global_steps += 1

        valid_accuracy_evaluator.save(os.path.join(configs.extra_log_dir, "valid"), "valid", cur_epoch)

        cur_valid_acc = valid_accuracy_evaluator.mean()

        if cur_valid_acc > cur_max_acc:
            cur_max_acc = cur_valid_acc
            #### Only save the higher score models
            with open(os.path.join(configs.model_dir, "best_model.txt"), "w") as f:
                f.write("{}".format(cur_epoch))
        model.save_model(configs.model_path_fn(cur_epoch))
