# -*- coding:utf-8 -*-

"""
@File    : plot_data.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/01/11 22:42
@Function: 
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_(x_train, y_train, plot_row=5):
    counts = dict(Counter(y_train))
    num_classes = len(np.unique(y_train))
    f, axarr = plt.subplots(plot_row, num_classes)
    for c in np.unique(y_train):  # Loops over classes, plot as columns
        c = int(c)
        ind = np.where(y_train == c)
        ind_plot = np.random.choice(ind[0], size=plot_row)
        for n in range(plot_row):  # Loops over rows
            axarr[n, c].plot(x_train[ind_plot[n], :])
            # Only shops axes for bottom row and left column
            if n == 0:
                axarr[n, c].set_title('Class %.0f (%.0f)' % (c, counts[float(c)]))
            if not n == plot_row - 1:
                plt.setp([axarr[n, c].get_xticklabels()], visible=False)
            if not c == 0:
                plt.setp([axarr[n, c].get_yticklabels()], visible=False)
    f.subplots_adjust(hspace=0)  # No horizontal space between subplots
    f.subplots_adjust(wspace=0)  # No vertical space between subplots
    plt.show()
    return
