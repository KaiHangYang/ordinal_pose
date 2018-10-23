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

# histogram x steps
data_step = 10

eval_result_data = "../evaluation/eval_result/syn/syn_3_1/result_eval_30w.npy"

if __name__ == "__main__":
    assert(os.path.isfile(eval_result_data))

    error_s = pose_error.mResultSaver()
    error_s.restore(eval_result_data)

    save_dir = os.path.splitext(eval_result_data)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # error_s.histdatas(os.path.join(save_dir, "mean_dist.npy" if data_type==0 else "per_dist.npy"), data_type=data_type, data_step=data_step)
    error_s.histogram(save_dir=save_dir, data_type=data_type, data_step=data_step, callback=callback)

    print("test")

