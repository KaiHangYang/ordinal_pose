import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append("../")
from utils.statistic_utils import pose_error

def callback(event, fig, text, data_step, datas):
    if event.dblclick:
        x_position = event.xdata
        text.set_position((event.xdata, event.ydata))
        data_index = int(x_position / data_step)
        if data_index >= 0 and data_index < len(datas):
            text.set_text(datas[data_index])
            fig.canvas.draw()

# 0 mean joints errors for every frames, 1 
data_type = 0
use_3d = False

# histogram x steps
display_step = 20
data_step = 10
data_max = 300

# eval_result_datas = [
        # ("order_3_1", "../evaluation/eval_result/10-22/syn_3_1/result_eval_30w.npy", [0.3, 0.73, 0.22, 0.5]),
        # ("noorder_3_1", "../evaluation/eval_result/10-23/syn_3_1/result_eval_28w.npy", [0.9, 0.21, 0.07, 0.5]),
        # ("gt_3_1", "../evaluation/eval_result/10-22/gt_3_1/result_eval_30w.npy", [0.1, 0.21, 0.87, 0.5])
        # ]


eval_result_datas = [
        ("order_3_2", "../evaluation/eval_result/10-22/syn_3_2/result_eval_26w.npy", [0.3, 0.73, 0.22, 0.5]),
        ("noorder_3_2", "../evaluation/eval_result/10-23/syn_3_2/result_eval_30w.npy", [0.9, 0.21, 0.07, 0.5]),
        ("gt_3_2", "../evaluation/eval_result/10-22/gt_3_2/result_eval_30w.npy", [0.1, 0.21, 0.87, 0.5])
        ]

graph_save_dir = "/home/kaihang/Desktop/test_dir"

if __name__ == "__main__":

    savers = []
    save_paths = []
    graph_colors = []

    for e in eval_result_datas:
        assert(os.path.isfile(e[1]))
        saver = pose_error.mResultSaver(name=e[0])
        saver.restore(e[1])
        savers.append(saver)
        # used for save the graph or statistic datas
        save_dir = os.path.splitext(e[1])[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_paths.append(os.path.join(save_dir, "mean_dist.npy" if data_type==0 else "per_dist.npy"))
        graph_colors.append(e[2])

    error_v = pose_error.mResultVisualizer()

    # error_v.histdatas(savers=savers, save_paths=save_paths, data_type=data_type, data_step=data_step)
    error_v.histogram_by_errors(savers=savers, save_dir=graph_save_dir, colors=graph_colors, data_type=data_type, data_step=data_step, display_step=display_step, data_max=data_max, use_3d=use_3d)
    # error_v.data_error_line_chart(savers=savers, save_dir=graph_save_dir, colors=graph_colors, data_type=data_type, display_step=display_step)

    print("test")

