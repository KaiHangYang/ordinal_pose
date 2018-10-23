import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

cur_hist_n = None
cur_hist_text = None
cur_ax = None
cur_joint_index = None

class mResultSaver(object):
    def __init__(self):
        self.results = None

    def restore(self, save_path):
        assert(os.path.isdir(os.path.dirname(save_path)))
        self.results = np.load(save_path).tolist()["results"]

    def save(self, save_path):
        if self.results is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            np.save(save_path, {"results": self.results})

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

    # data_type, 0 means the error_mean histogram, 1 means the histogram of error_per
    def histdatas(self, save_path, data_type=0, data_step=5):
        if self.results is not None:
            if data_type == 0:
                # statistic the error distribution of every frame pose
                statistic_datas = []

                for i, cur_data in enumerate(self.results):
                    cur_index = float(cur_data["index"])
                    cur_mean = cur_data["error_mean"]
                    statistic_datas.append([cur_mean, cur_index])

                statistic_datas = np.array(statistic_datas)

                error_dist = statistic_datas[:, 0]
                bins = np.arange(0, error_dist.max() + data_step, data_step)

                data_arrays = [[] for _ in range(len(bins) - 1)]

                for d in statistic_datas:
                    data_index = int(d[0] / data_step)
                    data_arrays[data_index].append(d.copy())

                np.save(save_path, {"hist": np.array(data_arrays), "data_step": data_step})
            else:
                # the distribution of each joints error
                nJoints = len(self.results[0]["error_per"])
                results = {"data_step": data_step, "hist_arr": []}

                for j_idx in range(nJoints):
                    statistic_datas = []
                    for cur_data in self.results:
                        cur_index = float(cur_data["index"])
                        cur_error = cur_data["error_per"][j_idx]
                        statistic_datas.append([cur_error, cur_index])

                    statistic_datas = np.array(statistic_datas)

                    error_dist = statistic_datas[:, 0]
                    bins = np.arange(0, error_dist.max() + data_step, data_step)

                    data_arrays = [[] for _ in range(len(bins) - 1)]

                    for d in statistic_datas:
                        data_index = int(d[0] / data_step)
                        data_arrays[data_index].append(d.copy())

                    results["hist_arr"].append(data_arrays)

                np.save(save_path, results)

    def histogram(self, save_dir, data_type=0, data_step=5, callback=None, fig_size=[14, 10]):
        global cur_hist_n
        global cur_hist_text
        global cur_ax

        if self.results is not None:
            if data_type == 0:
                statistic_datas = []

                for i, cur_data in enumerate(self.results):
                    cur_index = float(cur_data["index"])
                    cur_mean = cur_data["error_mean"]
                    statistic_datas.append([cur_mean, cur_index])

                statistic_datas = np.array(statistic_datas)
                error_dist = statistic_datas[:, 0]
                bins = np.arange(0, error_dist.max() + data_step, data_step)

                fig = plt.figure()
                fig.set_figwidth(fig_size[0])
                fig.set_figheight(fig_size[1])

                ax = fig.add_subplot(1, 1, 1)
                text = ax.text(1, 1, '')

                plt.title("Error histogram for mean per frame")
                plt.xlabel("Error(mm)")
                plt.ylabel("The number of datas")

                n, bins, patches = plt.hist(error_dist, bins=bins, range=(bins.min(), bins.max()), density=False, lw=1, edgecolor=(0, 0, 0), histtype="bar")

                def save_callback(event):
                    if event.key == "a":
                        # save_current fig
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        fig.savefig(os.path.join(save_dir, "frame_mean_error.png"), bbox_inches=extent.expanded(1.2, 1.3))

                fig.canvas.mpl_connect("button_press_event", lambda event: callback(event, fig, text, data_step, n))
                fig.canvas.mpl_connect("key_press_event", save_callback)
                plt.show()

            else:
                # draw the buttons and add the callback
                fig = plt.figure()
                fig.set_figheight(fig_size[1])
                fig.set_figwidth(fig_size[0])
                nJoints = len(self.results[0]["error_per"])

                # fig.set_figheight(9) # 6 for histogram, 3 for buttons
                # fig.set_figwidth(7)
                fig_grid_row = 10
                fig_grid_col = 7
                btn_grid_start = 7

                # ax_hist = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start, colspan=fig_grid_col)
                btn_arrs = []

                joints_statistic_datas = []
                for j_idx in range(nJoints):
                    statistic_datas = []

                    for cur_data in self.results:
                        cur_index = float(cur_data["index"])
                        cur_error = cur_data["error_per"][j_idx]
                        statistic_datas.append([cur_error, cur_index])

                    statistic_datas = np.array(statistic_datas)
                    joints_statistic_datas.append(statistic_datas)

                def btn_callback(event):
                    global cur_hist_n
                    global cur_hist_text
                    global cur_ax
                    global cur_joint_index

                    joint_index = int(event.inaxes.get_children()[0].get_text().split()[1]) - 1
                    cur_joint_index = joint_index

                    statistic_datas = joints_statistic_datas[joint_index]
                    error_dist = statistic_datas[:, 0]
                    bins = np.arange(0, error_dist.max() + data_step, data_step)

                    ax_hist = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start-1, colspan=fig_grid_col)
                    cur_ax = ax_hist

                    text = ax_hist.text(1, 1, '')

                    plt.title("Error histogram of joint {}".format(joint_index))
                    plt.xlabel("Error(mm)")
                    plt.ylabel("The number of datas")

                    n, bins, patches = plt.hist(error_dist, bins=bins, range=(bins.min(), bins.max()), density=False, lw=1, edgecolor=(0, 0, 0), histtype="bar")
                    cur_hist_n = n
                    cur_hist_text = text

                    fig.canvas.draw()

                def save_callback(event):
                    global cur_ax
                    global cur_joint_index
                    if event.key == "a":
                        # save_current fig
                        extent = cur_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        fig.savefig(os.path.join(save_dir, "joint_{}_error.png".format(cur_joint_index)), bbox_inches=extent.expanded(1.2, 1.3))


                for j_idx in range(nJoints):
                    ax_btns = plt.subplot2grid((fig_grid_row, fig_grid_col), (btn_grid_start + j_idx/fig_grid_col, j_idx%fig_grid_col), rowspan=1, colspan=1)
                    btn_arrs.append(Button(ax_btns, "Joint {}".format(j_idx + 1)))
                    str_to_print = str(j_idx)

                    btn_arrs[j_idx].on_clicked(btn_callback)

                fig.canvas.mpl_connect("button_press_event", lambda event: callback(event, fig, cur_hist_text, data_step, cur_hist_n))
                fig.canvas.mpl_connect("key_press_event", save_callback)

                plt.show()
