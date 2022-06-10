# -*- coding:utf-8 -*-

"""
@File    : transform_shapelets.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/03/19 16:21
@Function: 
"""
import random
import numpy
import torch
from matplotlib import pyplot
from tslearn.clustering import TimeSeriesKMeans


def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = numpy.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s + shapelets_size]
    return segments


def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters


def filterNanFromShapelet(shapelet):
    """
    Filter NaN values from a shapelet.
    Needed for the output of learning shapelets from tslearn, since smaller size shapelets are padded with NaN values.
    Note: Make sure the NaN values are only leading or trailing.
    """
    return shapelet[~numpy.isnan(shapelet)]


def lead_pad_shapelet(shapelet, pos):
    """
    Adding leading NaN values to shapelet to plot it on a time series at the best matching position.
    """
    pad = numpy.empty(pos)
    pad[:] = numpy.NaN
    padded_shapelet = numpy.concatenate([pad, filterNanFromShapelet(shapelet)])
    return padded_shapelet


def torch_dist_ts_shapelet(ts, shapelet, to_cuda=True):
    """
    Use PyTorch to calculate the distance between a shapelet and a time series.
    Implemented via unfolding.
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if to_cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    if shapelet.dim() == 1:
        shapelet = torch.unsqueeze(shapelet, 0)
    # unfold time series to emulate sliding window
    ts = ts.unfold(1, shapelet.shape[1], 1)
    # calculate euclidean distance over each segment
    dists = torch.sum(torch.cdist(ts, shapelet, p=2), dim=0)
    # filter min dist
    d_min, d_argmin = torch.min(dists, 0)
    return (d_min.item(), d_argmin.item())


def dists_to_shapelet(data, shapelet, to_cuda=True):
    """
    Calculate the distances of a shapelet to a bunch of time series.
    """
    shapelet = filterNanFromShapelet(shapelet)
    dists = []
    for i in range(len(data)):
        dists.append((torch_dist_ts_shapelet(data[i, :], shapelet, to_cuda=to_cuda), i))
    return sorted(dists, key=lambda x: x[0][0])


def plot_shapelet_on_ts_at_i(shapelet, ts, i, title="", axis=None):
    """
    Plot a shapelet on top of a timeseries
    """
    shapelet = filterNanFromShapelet(shapelet)
    padded_shapelet = lead_pad_shapelet(shapelet, i)
    if axis is None:
        pyplot.clf()
        pyplot.rcParams["figure.figsize"] = (23, 6)
        pyplot.plot(ts)
        pyplot.plot(padded_shapelet)
    else:
        axis.plot(ts)
        axis.plot(padded_shapelet)


def dists_ts_2_shapelet(ts, shapelet, to_cuda=True):
    """
    Calculate the distances of a time series to a shapelet.
    @param ts: single time series data
    @type ts: array-like(float) of shape (in_channels, len_ts)
    @param shapelet: single shapelet data
    @type shapelet: array-like(float) of shape (len_shplt)
    @param to_cuda: whether to use GPU
    @type to_cuda: bool
    @return: a list of the distances of each sliding windows to the shapelet
    @rtype: array-like(float) of shape (len_ts - len_shplt + 1)
    """
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if to_cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    if shapelet.dim() == 1:
        shapelet = torch.unsqueeze(shapelet, 0)
    # unfold time series to emulate sliding window
    ts = ts.unfold(1, shapelet.shape[1], 1)
    # calculate euclidean distance over each segment
    dists = torch.sum(torch.cdist(ts, shapelet, p=2), dim=0)
    #     # filter min dist
    #     d_min, d_argmin = torch.min(dists, 0)
    return dists.squeeze().cpu().detach().numpy()


def transform_single_ts_2_shpltseq(ts, shapelets, to_cuda=True):
    """
    Calculate the distances of a time series to a shapelet.
    @param ts: single time series data
    @type ts: array-like(float) of shape (in_channels, len_ts)
    @param shapelets: all shapelets data
    @type shapelet: array-like(float) of shape (n_shplt, len_shplt)
    @param to_cuda: whether to use GPU
    @type to_cuda: bool
    @return: a list of the distances of each sliding windows to the shapelet
    @rtype: array-like(float) of shape (len_ts - len_shplt + 1)
    """
    n_shapelets = len(shapelets)
    dists_all = []
    for i in range(n_shapelets):
        shapelet = filterNanFromShapelet(shapelets[i, 0])
        dists = dists_ts_2_shapelet(ts, shapelet, to_cuda)
        dists_all.append(dists)
    dists_all = numpy.array(dists_all)
    dists_all = torch.tensor(dists_all, dtype=torch.float)
    if to_cuda:
        dists_all = dists_all.cuda()
    d_min, d_argmin = torch.min(dists_all, 0)
    return d_min, d_argmin


def transform_data_2_shpltseq(data, shapelets, to_cuda=True):
    """
    Calculate the distances of all time series to a shapelet.
    @param ts: all time series data
    @type ts: array-like(float) of shape (n_ts, in_channels, len_ts)
    @param shapelets: all shapelets data
    @type shapelet: array-like(float) of shape (n_shplt, len_shplt)
    @param to_cuda: whether to use GPU
    @type to_cuda: bool
    @return: lists of the distances of each sliding windows to the shapelet
    @rtype: array-like(float) of shape (n_ts, len_ts - len_shplt + 1)
    """
    n_ts = len(data)
    shpltseq_min = []
    shpltseq_argmin = []
    for i in range(n_ts):
        d_min, d_argmin = transform_single_ts_2_shpltseq(data[i, :], shapelets, to_cuda)
        shpltseq_min.append(d_min.cpu().detach().numpy())
        shpltseq_argmin.append(d_argmin.cpu().detach().numpy())
    shpltseq_min = numpy.array(shpltseq_min)
    shpltseq_argmin = numpy.array(shpltseq_argmin)
    return shpltseq_min, shpltseq_argmin


def transform_shpltseq_2_onehot(shpltseq, n_shplt):
    x_train_data = []
    n_shpltseq = len(shpltseq)
    for i in range(n_shpltseq):
        one_hot = numpy.zeros((n_shplt), dtype=float)
        one_hot[shpltseq[i]] = 1.0
        x_train_data.append(one_hot)
    x_train_data = numpy.array(x_train_data)
    return x_train_data


def transform_shpltseqs_2_train(shpletseqs, n_shplt):
    train_data = []
    n_shpltseqs = len(shpletseqs)
    for i in range(n_shpltseqs):
        onehot_matrix = transform_shpltseq_2_onehot(shpletseqs[i], n_shplt)
        train_data.append(onehot_matrix)
    train_data = numpy.array(train_data)
    return train_data


