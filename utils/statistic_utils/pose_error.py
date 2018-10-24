import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

cur_hist_n = None
cur_hist_text = None
cur_ax = None
cur_joint_index = None

class mResultSaver(object):
    def __init__(self, name=""):
        self.results = None
        self.name = name

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

class mResultVisualizer(object):
    def __init__(self):
        pass
    # data_type, 0 means the error_mean histogram, 1 means the histogram of error_per
    def histdatas(self, savers, save_paths, data_type=0, data_step=5):

        for s_idx, saver in enumerate(savers):
            save_path = save_paths[s_idx]

            if saver.results is not None:
                if data_type == 0:
                    # statistic the error distribution of every frame pose
                    statistic_datas = []

                    for i, cur_data in enumerate(saver.results):
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
                    nJoints = len(saver.results[0]["error_per"])
                    results = {"data_step": data_step, "hist_arr": []}

                    for j_idx in range(nJoints):
                        statistic_datas = []
                        for cur_data in saver.results:
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

    # this one is with mouse callback
    def histogram_one(self, saver, save_dir, data_type=0, data_step=5, callback=None, fig_size=[14, 10]):
        global cur_hist_n
        global cur_hist_text
        global cur_ax

        if saver.results is not None:
            if data_type == 0:
                statistic_datas = []

                for i, cur_data in enumerate(saver.results):
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

                plt.title("Mean error histogram")
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
                nJoints = len(saver.results[0]["error_per"])

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

                    for cur_data in saver.results:
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

    def histogram(self, savers, save_dir, colors, data_type=0, data_step=5, display_step=10, fig_size=[14, 10], data_max=1000, use_3d=True):
        global cur_hist_n
        global cur_hist_text
        global cur_ax

        for tmp in savers:
            assert(tmp.results is not None)

        if data_type == 0:
            statistic_datas = [[] for _ in savers]
            result_names = []

            for s_idx, saver in enumerate(savers):
                result_names.append(saver.name)
                for cur_data in saver.results:
                    cur_index = float(cur_data["index"])
                    cur_mean = cur_data["error_mean"]
                    statistic_datas[s_idx].append([cur_mean, cur_index])

            statistic_datas = np.array(statistic_datas)

            error_dist = statistic_datas[:, :, 0]

            fig = plt.figure()
            fig.set_figwidth(fig_size[0])
            fig.set_figheight(fig_size[1])

            if use_3d:
                ax = fig.add_subplot(1, 1, 1, projection="3d")
                ax.set_zlabel("The number of datas")
                ax.set_xticks(np.arange(0, data_max, display_step))
                ax.set_yticks(np.arange(len(savers)))
                ax.w_yaxis.set_ticklabels(result_names)
            else:
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xticks(np.arange(0, data_max, display_step))
                ax.set_ylabel("The number of datas")
            # text = ax.text(1, 1, 1, '')

            plt.title("Mean error histogram")
            ax.set_xlabel("Error(mm)")

            bins = np.arange(0, data_max, data_step)

            hist_arr = []

            for e in error_dist:
                cur_hist, cur_hist_edge = np.histogram(e, bins=bins, range=(bins.min(), bins.max()), density=False)
                hist_arr.append(cur_hist)

            hist_arr = np.array(hist_arr)
            if use_3d:
                x_data, y_data = np.meshgrid(bins, np.arange(len(hist_arr)) - 0.05)

                x_data = x_data[:, :-1].flatten()
                y_data = y_data[:, :-1].flatten()
                z_data = hist_arr.flatten()

                color_arr = []
                for c_idx in range(hist_arr.shape[0]):
                    color_arr.append(np.ones([hist_arr.shape[1], 4]) * colors[c_idx])

                color_arr = np.reshape(color_arr, [-1, 4])

                ax.bar3d(x_data, y_data, np.zeros(len(z_data)), data_step, 0.1, z_data, zsort='average', color=color_arr, alpha=0.7)
            else:
                for cur_idx, cur_hist in enumerate(hist_arr):
                    ax.plot(bins[:-1], cur_hist, color=colors[cur_idx], label=result_names[cur_idx])
                plt.legend()

            def save_callback(event):
                if event.key == "a":
                    # save_current fig
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(os.path.join(save_dir, "frame_mean_error.png"), bbox_inches=extent.expanded(1.2, 1.3))

            fig.canvas.mpl_connect("key_press_event", save_callback)
            plt.show()

        else:
            # draw the buttons and add the callback
            fig = plt.figure()
            fig.set_figheight(fig_size[1])
            fig.set_figwidth(fig_size[0])
            nJoints = len(savers[0].results[0]["error_per"])

            # fig.set_figheight(9) # 6 for histogram, 3 for buttons
            # fig.set_figwidth(7)
            fig_grid_row = 10
            fig_grid_col = 7
            btn_grid_start = 7

            # ax_hist = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start, colspan=fig_grid_col)
            btn_arrs = []

            statistic_datas_arr = []

            result_names = []
            for s_idx, saver in enumerate(savers):
                result_names.append(saver.name)
                joints_statistic_datas = []
                for j_idx in range(nJoints):
                    statistic_datas = []

                    for cur_data in saver.results:
                        cur_index = float(cur_data["index"])
                        cur_error = cur_data["error_per"][j_idx]
                        statistic_datas.append([cur_error, cur_index])

                    statistic_datas = np.array(statistic_datas)
                    joints_statistic_datas.append(statistic_datas)

                statistic_datas_arr.append(joints_statistic_datas)

            statistic_datas_arr = np.array(statistic_datas_arr)

            def btn_callback(event):
                global cur_ax
                global cur_joint_index

                joint_index = int(event.inaxes.get_children()[0].get_text().split()[1]) - 1
                cur_joint_index = joint_index

                statistic_datas = statistic_datas_arr[:, joint_index]
                error_dist = statistic_datas[:, :, 0]

                if use_3d:
                    ax_hist = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start-1, colspan=fig_grid_col, projection="3d")
                    cur_ax = ax_hist
                    ax_hist.set_zlabel("The number of datas")

                    ax_hist.set_xticks(np.arange(0, data_max, display_step))
                    ax_hist.set_yticks(np.arange(len(error_dist)))
                    ax_hist.w_yaxis.set_ticklabels(result_names)
                else:
                    ax_hist = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start-1, colspan=fig_grid_col)
                    cur_ax = ax_hist

                    ax_hist.set_xticks(np.arange(0, data_max, display_step))
                    ax_hist.set_ylabel("The number of datas")

                # text = ax_hist.text(1, 1, '')
                ax_hist.set_xlabel("Error(mm)")

                bins = np.arange(0, data_max, data_step)
                hist_arr = []

                for e in error_dist:
                    cur_hist, cur_hist_edge = np.histogram(e, bins=bins, range=(bins.min(), bins.max()), density=False)
                    hist_arr.append(cur_hist)

                if use_3d:
                    x_data, y_data = np.meshgrid(bins, np.arange(len(hist_arr)) - 0.05)

                    hist_arr = np.array(hist_arr)
                    x_data = x_data[:, :-1].flatten()
                    y_data = y_data[:, :-1].flatten()
                    z_data = hist_arr.flatten()

                    color_arr = []
                    for c_idx in range(hist_arr.shape[0]):
                        color_arr.append(np.ones([hist_arr.shape[1], 4]) * colors[c_idx])

                    color_arr = np.reshape(color_arr, [-1, 4])

                    ax_hist.bar3d(x_data, y_data, np.zeros(len(z_data)), data_step, 0.1, z_data, zsort='average', color=color_arr, alpha=0.7)
                else:
                    for cur_idx, cur_hist in enumerate(hist_arr):
                        ax_hist.plot(bins[:-1], cur_hist, color=colors[cur_idx], label=result_names[cur_idx])
                    plt.legend()

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

            fig.canvas.mpl_connect("key_press_event", save_callback)
            plt.show()

    def data_error_line_chart(self, savers, save_dir, colors, data_type=0, display_step=10, fig_size=[14, 10]):
        global cur_hist_n
        global cur_hist_text
        global cur_ax

        for tmp in savers:
            assert(tmp.results is not None)

        nJoints = len(savers[0].results[0]["error_per"])
        nDatas = len(savers[0].results)

        if data_type == 0:
            statistic_datas = [[] for _ in savers]
            result_names = []

            for s_idx, saver in enumerate(savers):
                result_names.append(saver.name)
                for cur_data in saver.results:
                    cur_index = float(cur_data["index"])
                    cur_mean = cur_data["error_mean"]
                    statistic_datas[s_idx].append([cur_mean, cur_index])

            statistic_datas = np.array(statistic_datas)

            fig = plt.figure()
            fig.set_figwidth(fig_size[0])
            fig.set_figheight(fig_size[1])

            ax = fig.add_subplot(1, 1, 1)
            ax.set_xticks(np.arange(0, nDatas, display_step))
            ax.set_ylabel("Error(mm)")
            # text = ax.text(1, 1, 1, '')

            plt.title("Error distribution of all datas")
            ax.set_xlabel("data index")

            for cur_idx, cur_data in enumerate(statistic_datas):
                ax.plot(cur_data[:, 1], cur_data[:, 0], color=colors[cur_idx], label=result_names[cur_idx])
            plt.legend()

            def save_callback(event):
                if event.key == "a":
                    # save_current fig
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(os.path.join(save_dir, "data_error_dist.png"), bbox_inches=extent.expanded(1.2, 1.3))

            fig.canvas.mpl_connect("key_press_event", save_callback)
            plt.show()

        else:
            # draw the buttons and add the callback
            fig = plt.figure()
            fig.set_figheight(fig_size[1])
            fig.set_figwidth(fig_size[0])

            # fig.set_figheight(9) # 6 for histogram, 3 for buttons
            # fig.set_figwidth(7)
            fig_grid_row = 10
            fig_grid_col = 7
            btn_grid_start = 7

            # ax_hist = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start, colspan=fig_grid_col)
            btn_arrs = []

            statistic_datas_arr = []

            result_names = []
            for s_idx, saver in enumerate(savers):
                result_names.append(saver.name)
                joints_statistic_datas = []
                for j_idx in range(nJoints):
                    statistic_datas = []

                    for cur_data in saver.results:
                        cur_index = float(cur_data["index"])
                        cur_error = cur_data["error_per"][j_idx]
                        statistic_datas.append([cur_error, cur_index])

                    statistic_datas = np.array(statistic_datas)
                    joints_statistic_datas.append(statistic_datas)

                statistic_datas_arr.append(joints_statistic_datas)

            statistic_datas_arr = np.array(statistic_datas_arr)

            def btn_callback(event):
                global cur_ax
                global cur_joint_index

                joint_index = int(event.inaxes.get_children()[0].get_text().split()[1]) - 1
                cur_joint_index = joint_index

                statistic_datas = statistic_datas_arr[:, joint_index]

                ax = plt.subplot2grid((fig_grid_row, fig_grid_col), (0, 0), rowspan=btn_grid_start-1, colspan=fig_grid_col)
                cur_ax = ax

                ax.set_xticks(np.arange(0, nDatas, display_step))
                ax.set_xlabel("data index")
                ax.set_ylabel("Error(mm)")

                for cur_idx, cur_data in enumerate(statistic_datas):
                    ax.plot(cur_data[:, 1], cur_data[:, 0], color=colors[cur_idx], label=result_names[cur_idx])
                plt.legend()

                fig.canvas.draw()

            def save_callback(event):
                global cur_ax
                global cur_joint_index
                if event.key == "a":
                    # save_current fig
                    extent = cur_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(os.path.join(save_dir, "joint_{}_error_dist.png".format(cur_joint_index)), bbox_inches=extent.expanded(1.2, 1.3))


            for j_idx in range(nJoints):
                ax_btns = plt.subplot2grid((fig_grid_row, fig_grid_col), (btn_grid_start + j_idx/fig_grid_col, j_idx%fig_grid_col), rowspan=1, colspan=1)
                btn_arrs.append(Button(ax_btns, "Joint {}".format(j_idx + 1)))
                str_to_print = str(j_idx)

                btn_arrs[j_idx].on_clicked(btn_callback)

            fig.canvas.mpl_connect("key_press_event", save_callback)
            plt.show()
