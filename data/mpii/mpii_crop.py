import numpy as np
import os
import sys
import h5py
import cv2

class MPIICropper(object):
    def __init__(self, img_dir, lbl_path, is_shuffle=True, batch_size=1, res=256, pad_color=[128, 128, 128]):
        self.pad_color = pad_color
        self.res = res
        self.box_size = 256
        self.img_dir = img_dir
        self.lbl_path = lbl_path
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size

        if not os.path.isfile(self.lbl_path):
            print("{} is not existing!".format(self.lbl_path))
            quit()

        self.labels = h5py.File(self.lbl_path)
        self.n_labels = len(self.labels["imgname"])
        self._cur_index = 0

        self.norms = lambda x: self.labels["normalize"][x]
        self.imgpathes = lambda x: os.path.join(self.img_dir, self.labels["imgname"][x])
        self.imgnames = lambda x: self.labels["imgname"][x]
        self.centers = lambda x: self.labels["center"][x]
        self.scales = lambda x: self.labels["scale"][x]
        self.joints_2d = lambda x: self.labels["part"][x]

        self.index_arrays = np.arange(0, self.n_labels, 1)
        if self.is_shuffle:
            np.random.shuffle(self.index_arrays)

    @property
    def cur_index(self):
        return self._cur_index

    @cur_index.setter
    def cur_index(self, val):
        self._cur_index = val
        if self._cur_index < 0:
            self._cur_index = 0
        elif self._cur_index >= self.n_labels:
            self._cur_index = self.n_labels - 1

    def progress(self):
        return self.cur_index+1, self.n_labels

    def reset(self):
        self.cur_index = 0
        if self.is_shuffle:
            np.random.shuffle(self.index_arrays)

    def crop_data(self, data, extra_scale=1.0):
        cur_img = data["img"].copy()
        cur_joints_2d = data["joints_2d"].copy()
        cur_center = data["center"]
        cur_scale = data["scale"]

        img_width = cur_img.shape[1]
        img_height = cur_img.shape[0]

        cen_x, cen_y = cur_center

        crop_box_size = int(cur_scale * self.box_size * extra_scale)

        min_x = int(cen_x - crop_box_size / 2)
        min_y = int(cen_y - crop_box_size / 2)
        max_x = min_x + crop_box_size
        max_y = min_y + crop_box_size

        raw_min_x = 0
        raw_max_x = img_width
        raw_min_y = 0
        raw_max_y = img_height

        if min_x > 0:
            raw_min_x = min_x
        if min_y > 0:
            raw_min_y = min_y
        if max_x < img_width:
            raw_max_x = max_x
        if max_y < img_height:
            raw_max_y = max_y

        cropped_img = cv2.copyMakeBorder(cur_img[raw_min_y:raw_max_y, raw_min_x:raw_max_x], top=raw_min_y-min_y, bottom=max_y - raw_max_y, left=raw_min_x - min_x, right=max_x - raw_max_x, value=self.pad_color, borderType=cv2.BORDER_CONSTANT)

        min_size = min(cropped_img.shape[0], cropped_img.shape[1])
        cropped_img = cropped_img[0:min_size, 0:min_size]

        final_scale = float(min_size) / self.res

        cropped_img = cv2.resize(cropped_img, (self.res, self.res))
        valid_joints_index = (cur_joints_2d != [0, 0]).any(axis=1)

        cur_joints_2d[valid_joints_index] = (cur_joints_2d[valid_joints_index] - [min_x, min_y]) / final_scale
        cropped_joints_2d = cur_joints_2d

        data["cropped_img"] = cropped_img
        data["cropped_joints_2d"] = cropped_joints_2d
        data["final_scale"] = final_scale

        return data

    # get a batch of datas
    def get(self):
        batches = []
        finished_epoch = False
        for i in range(self.batch_size):
            cur_label_num = self.index_arrays[self.cur_index]

            batches.append(self.crop_data(dict(
                name=self.imgnames(cur_label_num),
                img=cv2.imread(self.imgpathes(cur_label_num)),
                joints_2d=self.joints_2d(cur_label_num),
                center=self.centers(cur_label_num),
                scale=self.scales(cur_label_num),
                norm=self.norms(cur_label_num)
            )))
            self.cur_index += 1

        if self.cur_index == self.n_labels - 1:
            finished_epoch = True

        return batches, finished_epoch

if __name__ == "__main__":
    train_or_valid = "valid"
    data_reader = MPIICropper(img_dir="/home/kaihang/DataSet/PoseDataSets/2D/images", lbl_path="/home/kaihang/Projects/DLCM-release/data/mpii/"+train_or_valid+".h5", is_shuffle=False, batch_size=1)

    target_img_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/"+train_or_valid+"/images"
    target_lbl_dir = "/home/kaihang/DataSet_2/Ordinal/mpii/"+train_or_valid+"/labels"

    counter = 0

    finished_epoch = False
    while not finished_epoch:
        sys.stdout.write("\rCurrently process {}".format(counter))
        sys.stdout.flush()

        cur_data, finished_epoch = data_reader.get()
        cur_data = cur_data[0]

        cv2.imwrite(os.path.join(target_img_dir, "{}.jpg".format(counter)), cur_data["cropped_img"])
        np.save(os.path.join(target_lbl_dir, "{}.npy".format(counter)), dict(
            name=cur_data["name"],
            raw_joints_2d=cur_data["joints_2d"],
            cropped_joints_2d=cur_data["cropped_joints_2d"],
            final_scale=cur_data["final_scale"],
            center=cur_data["center"],
            scale=cur_data["scale"],
            norm=cur_data["norm"]
            ))
        counter += 1
