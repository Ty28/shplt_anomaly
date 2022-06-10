# -*- coding:utf-8 -*-

"""
@File    : shapelet_bench.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/06/10 11:42
@Function:
"""

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score, precision_score
from train_utils.eval_ import *
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from shapelets_utils.transform_shapelets import *
from shapelets_utils.learning_shapelets import LearningShapelets
from torch import nn, optim
import sys
import os
import warnings

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())


def eval_f1_score(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"f1_score: {f1_score(Y, predictions)}")
    print(f"recall_score: {recall_score(Y, predictions)}")
    print(f"precision_score: {precision_score(Y, predictions)}")
    return f1_score(
        Y, predictions), recall_score(
        Y, predictions), precision_score(
            Y, predictions)


def eval_roc_auc_score(model, X, Y):
    predictions = model.predict(X)
    b = torch.from_numpy(predictions)
    net_1 = nn.Softmax(dim=1)
    b = net_1(b)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(Y, b)
    roc_auc = auc(fpr, tpr)
    print(f"roc_auc: {roc_auc}")
    return roc_auc


def fit(dataset_name, n, length, save_transformation=True):
    """
        Fit the shapelet classification model.
        Parameters
        ----------
        dataset_name : str
            the name of dataset in UEA dataset bench
        n : int
            the number of shapelets
        length : float
            the length of shapelets, final length is (length * ts_length)
        save_transformation : bool
            if true save the shapelet transformation sequences
                train data in /data/dataset_name/x_train_{n}.npy
                test data in /data/dataset_name/x_test_{n}.npy
    """

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

    # change the label to [0, ..., n]
    lbs = numpy.unique(y_train)
    y_train_return, y_test_return = numpy.zeros(
        y_train.shape, dtype='int8'), numpy.zeros(
        y_test.shape, dtype='int8')
    val = lbs[0]
    y_train_return[y_train == val] = 1
    y_test_return[y_test == val] = 1
    y_train = y_train_return
    y_test = y_test_return

    mts_path = os.path.join('data', dataset_name)
    if not os.path.exists(mts_path):
        os.mkdir(mts_path)
    # print(mts_train.shape)
    numpy.save(os.path.join(mts_path, f'x_train_{n}'), X_train)
    numpy.save(os.path.join(mts_path, f'y_train_{n}'), y_train)
    numpy.save(os.path.join(mts_path, f'x_test_{n}'), X_test)
    numpy.save(os.path.join(mts_path, f'y_test_{n}'), y_test)

    # model params
    n_ts, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    # learn X shapelets of length Y
    shapelet_length = int(len_ts * length)
    shapelet_size = n
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

    # init shapelets blocks by kmeans
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
    test_f1_Lr, test_rec_Lr, test_pre_Lr = eval_f1_score(
        learning_shapelets_Lr, X_test, y_test)
    # test_roc_Lr = eval_roc_auc_score(learning_shapelets_Lr, X_test, y_test)

    if save_transformation:
        mts_path = os.path.join('data_1', dataset_name)
        if not os.path.exists(mts_path):
            os.mkdir(mts_path)
        mts_train = learning_shapelets_Lr.transform(X_train)
        mts_test = learning_shapelets_Lr.transform(X_test)
        # print(mts_train.shape)
        numpy.save(os.path.join(mts_path, f'x_train_{n}'), mts_train)
        numpy.save(os.path.join(mts_path, f'y_train_{n}'), y_train)
        numpy.save(os.path.join(mts_path, f'x_test_{n}'), mts_test)
        numpy.save(os.path.join(mts_path, f'y_test_{n}'), y_test)
    return test_acc_Lr, test_f1_Lr, test_rec_Lr, test_pre_Lr


# you need to input the dataset name in this array
dataset_names = [
    'SelfRegulationSCP2',
    'FingerMovements',
    'RacketSports',
    'Epilepsy']

if __name__ == '__main__':
    for name in dataset_names:
        for num in range(20, 60):
            fit(name, num, 0.05)
