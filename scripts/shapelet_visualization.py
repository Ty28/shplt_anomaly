# -*- coding:utf-8 -*-

"""
@File    : shapelet_visualization.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/06/10 12:00
@Function:
"""

from data_utils.plot_data import *
from train_utils.eval_ import *
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from shapelets_utils.transform_shapelets import *
from shapelets_utils.learning_shapelets import LearningShapelets
from matplotlib import pyplot
from torch import nn, optim
import numpy
import sys
import os
import warnings

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def vis(dataset_name, dimensions=3, shapelet_samples=3):
    # load original training&test data
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
    X_train_size = X_train.shape[0]
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    # normalize training&test data
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)

    # reverse last 2 dimension: (n_ts, len_ts, n_channels) => (n_ts,
    # n_channels, len_ts)
    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)

    lbs = numpy.unique(y_train)
    y_train_return, y_test_return = numpy.zeros(
        y_train.shape, dtype='int8'), numpy.zeros(
        y_test.shape, dtype='int8')
    val = lbs[0]
    y_train_return[y_train == val] = 1
    y_test_return[y_test == val] = 1
    y_train = y_train_return
    y_test = y_test_return

    # model params
    n_ts, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    # learn X shapelets of length Y
    shapelet_length = int(len_ts * 0.1)
    if shapelet_length < 14:
        shapelet_length = 14
    shapelet_size = 40
    shapelets_size_and_len = {shapelet_length: shapelet_size}
    dist_measure = "euclidean"
    lr = 4e-3
    wd = 1e-4
    epsilon = 1e-7
    batch_size = 128
    shuffle = True
    drop_last = False
    epochs = 200
    n_epoch_steps = 5 * 8
    l1 = 0.2
    l2 = 0.01
    k = int(0.05 * batch_size) if batch_size <= X_train_size else X_train_size

    shapelets_blocks = []
    for i, (shapelets_size, num_shapelets) in enumerate(
            shapelets_size_and_len.items()):
        weights_block = get_weights_via_kmeans(
            X_train, shapelets_size, num_shapelets)
        shapelets_blocks.append(weights_block)
    # init learning shapelets model
    learning_shapelets_Lr = LearningShapelets(
        shapelets_size_and_len=shapelets_size_and_len,
        in_channels=n_channels,
        num_classes=num_classes,
        loss_func=loss_func,
        to_cuda=True,
        verbose=1,
        dist_measure=dist_measure,
        l1=l1,
        l2=l2,
        k=k)
    # set shapelet weights of blocks
    for i, shapelets_block in enumerate(shapelets_blocks):
        learning_shapelets_Lr.set_shapelet_weights_of_block(i, shapelets_block)

    # set optimizer
    optimizer = optim.Adam(
        learning_shapelets_Lr.model.parameters(),
        lr=lr,
        eps=epsilon)
    learning_shapelets_Lr.set_optimizer(optimizer)

    # train step
    losses_acc = []
    losses_dist = []
    losses_sim = []
    train_acc_last_Lr = 0
    for _ in range(n_epoch_steps):
        losses_acc_i, losses_dist_i, losses_sim_i = \
            learning_shapelets_Lr.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=drop_last)
        losses_acc += losses_acc_i
        losses_dist += losses_dist_i
        losses_sim += losses_sim_i
        train_acc_current_Lr = eval_accuracy(
            learning_shapelets_Lr, X_train, y_train)
        if train_acc_current_Lr - train_acc_last_Lr < 1e-3 or train_acc_current_Lr == 1.0:
            break
        train_acc_last_Lr = train_acc_current_Lr
    test_acc_Lr = eval_accuracy(learning_shapelets_Lr, X_test, y_test)

    shapelets_ce = learning_shapelets_Lr.get_shapelets()
    n_shapelets_ce = len(shapelets_ce)

    # fig, axes = pyplot.subplots(dimensions, shapelet_samples)
    # pyplot.rcParams['axes.facecolor'] = 'white'
    # pyplot.rcParams['savefig.facecolor'] = 'white'
    # fig.set_size_inches(4 * 5, 8)
    # for d in range(dimensions):
    #     for i in range(shapelet_samples):
    #         if i < n_shapelets_ce:
    #             shapelet_ce = filterNanFromShapelet(shapelets_ce[i, d])
    #             best_match_ce = dists_to_shapelet(X_test, shapelet_ce, to_cuda=True)[0]
    #             plot_shapelet_on_ts_at_i(shapelet_ce, X_test[best_match_ce[1], d], best_match_ce[0][1], axis=axes[d, i])
    #             axes[d, i].set_title(f'shapelet样例{i}第{d}维')
    # pyplot.tight_layout(rect=[0, 0.03, 1, 0.98])
    # # pyplot.show()
    # fig.savefig(f'{dataset_name}_shplt_on_ts_{shapelet_samples}s_{dimensions}d.png')
    # fig.savefig(f'{dataset_name}_shplt_on_ts_{shapelet_samples}s_{dimensions}d.svg')

    # for s in range(shapelet_samples):
    #     fig1 = pyplot.figure(dpi=500, facecolor='white')
    #     fig1.set_size_inches(15, 10)
    #     gs = fig1.add_gridspec(15, 8)
    #     fig_ax1 = fig1.add_subplot(gs[:8, :])
    #     for i in range(3):
    #         fig_ax1.plot(X_test[s, i])
    #     fig1.savefig(f'{dataset_name}_ts{s}_{dimensions}d.png')
    #     fig1.savefig(f'{dataset_name}_ts{s}_{dimensions}d.svg')

    # for s in range(shapelet_samples):
    #     fig2 = pyplot.figure(dpi=100, facecolor='white')
    #     fig2.set_size_inches(6, 12)
    #     gs = fig2.add_gridspec(6, 4)
    #     fig_ax2 = fig2.add_subplot(gs[:4, :])
    #     # fig_ax2.set_title("shapelet样例0前3维")
    #     for i in range(dimensions):
    #         fig_ax2.plot(shapelets_ce[s, i])
    #     fig2.savefig(f'{dataset_name}_shapelet{s}_{dimensions}d.png')
    #     fig2.savefig(f'{dataset_name}_shapelet{s}_{dimensions}d.svg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker
    pyplot.rcParams['axes.facecolor'] = 'white'
    pyplot.rcParams['savefig.facecolor'] = 'white'

    fig = plt.figure(dpi=100,
                     constrained_layout=True,  # 类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                     )
    fig.set_size_inches(2 * 5, 4)
    # GridSpec将fiure分为3行3列，每行三个axes，gs为一个matplotlib.gridspec.GridSpec对象，可灵活的切片figure
    gs = GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title(f'(a) {dataset_name}时间序列实例')
    for d in range(dimensions):
        plt.plot(X_test[0, d])
    for i in range(shapelet_samples):
        ax = fig.add_subplot(gs[1, i:i + 1])
        for d in range(dimensions):
            if i < n_shapelets_ce:
                shapelet_ce = filterNanFromShapelet(shapelets_ce[i, d])
                if len(shapelet_ce) < 30:
                    x_major_locator = ticker.MultipleLocator(2)  # 以每15显示
                    ax.xaxis.set_major_locator(x_major_locator)
                else:
                    x_major_locator = ticker.MultipleLocator(25)  # 以每15显示
                    ax.xaxis.set_major_locator(x_major_locator)
                plt.plot(shapelet_ce)
                ax.set_title(f'(b) shapelet样例{i}')
    fig.savefig(f'{dataset_name}_ts_shapelet{shapelet_samples}')


# you need to input the dataset name in this array
# dataset_names = ['SelfRegulationSCP2', 'FingerMovements', 'RacketSports', 'Epilepsy']
dataset_names = ['Epilepsy']
if __name__ == '__main__':
    for name in dataset_names:
        vis(name)
