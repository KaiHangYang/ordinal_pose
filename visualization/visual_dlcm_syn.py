import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from net import dlcm_syn_net
from utils.preprocess_utils import dlcm_syn_preprocess
from utils.visualize_utils import display_utils
from utils.common_utils import my_utils
from utils.evaluate_utils import evaluators
from utils.defs.configs import mConfigs
from utils.defs.skeleton import mSkeleton15 as skeleton

##################### Evaluation Configs ######################
configs = mConfigs("./eval.conf", "dlcm_syn_net")
################ Reseting  #################
configs.loss_weights = [7.0, 1.0, 1.0]
configs.loss_weight_fb = 2.0
configs.loss_weight_br = 2.0
configs.pose_2d_scale = 4.0
configs.hm_size = int(configs.img_size / configs.pose_2d_scale)
configs.is_use_bn = True

configs.learning_rate = 2.5e-4
configs.lr_decay_rate = 0.10
configs.lr_decay_step = 200000
configs.nFeats = 256
configs.nModules = 1

configs.batch_size = 1
### train or valid
configs.range_file =  configs.h36m_valid_range_file
configs.img_path_fn = configs.h36m_valid_img_path_fn
configs.lbl_path_fn = configs.h36m_valid_lbl_path_fn

configs.printConfig()
preprocessor = dlcm_syn_preprocess.DLCMSynProcessor(skeleton=skeleton, img_size=configs.img_size, hm_size=configs.hm_size, sigma=1.0, bone_width=6, joint_ratio=6, bg_color=0.2)

evaluation_models = [740000]
###############################################################

if __name__ == "__main__":

    ################### Initialize the data reader ###################
    range_arr = np.load(configs.range_file)
    data_from = 50000
    data_to = len(range_arr)
    img_list = [configs.img_path_fn(i) for i in range_arr]
    lbl_list = [configs.lbl_path_fn(i) for i in range_arr]

    input_images = tf.placeholder(shape=[2*configs.batch_size, configs.img_size, configs.img_size, 3], dtype=tf.float32)
    dlcm_syn_model = dlcm_syn_net.mDLCMSynNet(skeleton=skeleton, img_size=configs.img_size, batch_size=2*configs.batch_size, is_training=False, loss_weights=configs.loss_weights, loss_weight_fb=configs.loss_weight_fb, loss_weight_br=configs.loss_weight_br, pose_2d_scale=configs.pose_2d_scale, is_use_bn=configs.is_use_bn, nFeats=configs.nFeats, nModules=configs.nModules)

    with tf.Session() as sess:
        with tf.device("/device:GPU:0"):
            dlcm_syn_model.build_model(input_images)
            dlcm_syn_model.build_evaluation(skeleton.flip_array)

        print("Network built!")

        model_saver = tf.train.Saver()
        net_init = tf.global_variables_initializer()
        sess.run([net_init])

        for cur_model_iterations in evaluation_models:

            data_index = my_utils.mRangeVariable(min_val=data_from, max_val=data_to-1, initial_val=data_from)
            ################# Restore the model ################

            if os.path.exists(configs.model_path_fn(cur_model_iterations)+".index"):
                print("#######################Restored all weights ###########################")
                model_saver.restore(sess, configs.model_path_fn(cur_model_iterations))
            else:
                print(configs.model_path_fn(cur_model_iterations))
                print("The prev model is not existing!")
                quit()
            ####################################################

            while not data_index.isEnd():

                batch_images_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)
                batch_images_flipped_np = np.zeros([configs.batch_size, configs.img_size, configs.img_size, 3], dtype=np.float32)

                batch_joints_2d_np = np.zeros([configs.batch_size, skeleton.n_joints, 2], dtype=np.float32)

                batch_fb_np = np.zeros([configs.batch_size, skeleton.n_bones], dtype=np.int32)
                batch_br_np = np.zeros([configs.batch_size, (skeleton.n_bones-1) * skeleton.n_bones / 2], dtype=np.int32)

                img_path_for_show = []
                lbl_path_for_show = []

                for b in range(configs.batch_size):
                    img_path_for_show.append(os.path.basename(img_list[data_index.val]))
                    lbl_path_for_show.append(os.path.basename(lbl_list[data_index.val]))

                    cur_img = cv2.imread(img_list[data_index.val])
                    cur_label = np.load(lbl_list[data_index.val]).tolist()
                    data_index.val += 1

                    if "joints_3d" in cur_label.keys():
                        ##!!!!! the joints_3d is the joints under the camera coordinates !!!!!##
                        ##!!!!! the joints_2d is the cropped ones !!!!!##
                        # the h36m datas
                        cur_joints_2d = cur_label["joints_2d"].copy()[skeleton.h36m_selected_index]
                        cur_joints_3d = cur_label["joints_3d"].copy()[skeleton.h36m_selected_index]
                        cur_scale = cur_label["scale"]
                        cur_center = cur_label["center"]
                        cur_cam_mat = cur_label["cam_mat"]

                        cur_img, cur_maps, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_h36m(img=cur_img, joints_2d=cur_joints_2d, joints_3d=cur_joints_3d, scale=cur_scale, center=cur_center, cam_mat=cur_cam_mat, is_training=False, is_drawmaps=False)
                    else:
                        # the mpii lsp datas
                        cur_joints_2d = cur_label["joints_2d"].copy()
                        cur_bone_status = cur_label["bone_status"].copy()
                        cur_bone_relations = cur_label["bone_relations"].copy()

                        cur_img, cur_maps, cur_joints_2d, cur_bone_status, cur_bone_relations = preprocessor.preprocess_base(img=cur_img, joints_2d=cur_joints_2d, bone_status=cur_bone_status, bone_relations=cur_bone_relations, is_training=False, is_drawmaps=False)

                    batch_images_np[b] = cur_img.copy()
                    batch_images_flipped_np[b] = cv2.flip(cur_img, 1)

                    batch_joints_2d_np[b] = cur_joints_2d.copy() * configs.pose_2d_scale
                    batch_fb_np[b] = cur_bone_status.copy()
                    batch_br_np[b] = cur_bone_relations[np.triu_indices(skeleton.n_bones, k=1)].copy()

                pd_fb, \
                pd_br, \
                pd_br_belief, \
                raw_2d, \
                mean_2d = sess.run([
                        dlcm_syn_model.pd_fb_result,
                        dlcm_syn_model.pd_br_result,
                        dlcm_syn_model.pd_br_belief,
                        dlcm_syn_model.raw_pd_2d,
                        dlcm_syn_model.mean_pd_2d
                        ],
                        feed_dict={input_images: np.concatenate([batch_images_np, batch_images_flipped_np], axis=0)})

                print((len(lbl_path_for_show) * "{}\n").format(*zip(img_path_for_show, lbl_path_for_show)))

                cur_img = (batch_images_np[0] * 255).astype(np.uint8)

                pd_2d = raw_2d[0]
                pd_fb = pd_fb[0]
                pd_br = pd_br[0]

                gt_2d = batch_joints_2d_np[0]
                gt_fb = batch_fb_np[0]
                gt_br = batch_br_np[0]

                gt_2d_img = display_utils.drawLines(cur_img.copy(), gt_2d, indices=skeleton.bone_indices, color_table=skeleton.bone_colors * 255)
                pd_2d_img = display_utils.drawLines(cur_img.copy(), pd_2d, indices=skeleton.bone_indices, color_table=skeleton.bone_colors * 255)

                gt_bone_order = preprocessor.bone_order_from_bone_relations(gt_br, np.ones_like(gt_br))
                pd_bone_order = preprocessor.bone_order_from_bone_relations(pd_br, pd_br_belief)

                gt_skeleton = (preprocessor.draw_syn_img(gt_2d, gt_fb, gt_bone_order)).astype(np.uint8)
                pd_skeleton = (preprocessor.draw_syn_img(pd_2d, pd_fb, pd_bone_order)).astype(np.uint8)

                gt_img = np.concatenate([cur_img, gt_2d_img, gt_skeleton], axis=1)
                pd_img = np.concatenate([cur_img, pd_2d_img, pd_skeleton], axis=1)

                img_for_show = np.concatenate([gt_img, pd_img], axis=0)

                cv2.imshow("result", img_for_show)
                cv2.waitKey(5)
