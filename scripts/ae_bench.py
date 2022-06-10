# -*- coding:utf-8 -*-

"""
@File    : ae_bench.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/06/10 11:50
@Function:
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.lstm import RecurrentAutoencoder
import torch.nn as nn
import torch
import copy
from torch.utils.data import DataLoader
import random
import torch.backends.cudnn as cudnn
import os
from torch.utils.data import Dataset
from sklearn.metrics import auc


# EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self,
            patience=7,
            verbose=False,
            delta=0,
            path='checkpoint.pt',
            trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# DataLoader
class AutoencoderDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx, :, :])
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_name = 'Epilepsy'
numas = 40  # number of shapelets
mts_path = os.path.join('data', dataset_name)
train_dataset_x = np.load(os.path.join(mts_path, f'x_train_{numas}.npy'))
train_dataset_y = np.load(os.path.join(mts_path, f'y_train_{numas}.npy'))
normal_df = []
anomaly_df = []
normal_label = 0
for i in range(len(train_dataset_y)):
    if train_dataset_y[i] == normal_label:
        normal_df.append(train_dataset_x[i])
    else:
        anomaly_df.append(train_dataset_x[i])
normal_df = np.array(normal_df)
anomaly_df = np.array(anomaly_df)
RANDOM_SEED = 42
train_df, val_df = train_test_split(
    normal_df,
    test_size=0.05,
    random_state=RANDOM_SEED
)

train_dataset = torch.tensor(train_df)
train_dataset = train_dataset.unsqueeze(dim=2)
train_dataset = train_dataset.float()
print(train_dataset.shape)

val_dataset = torch.tensor(val_df)
val_dataset = val_dataset.unsqueeze(dim=2)
val_dataset = val_dataset.float()
print(val_dataset.shape)

# LSTM AutoEncoder
timesteps = train_dataset.shape[1]
epochs = 50
batch = 128
lr = 0.0001
seq_len = timesteps
n_features = train_dataset.shape[2]
print(timesteps, n_features)
patience = 7

# training


def train_model(model, train_dataset, val_dataset, n_epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    print("start!")
    train_dataset_ae = AutoencoderDataset(train_dataset)
    tr_dataloader = DataLoader(train_dataset_ae, batch_size=batch_size,
                               shuffle=False, num_workers=0)
    val_dataset_ae = AutoencoderDataset(val_dataset)
    va_dataloader = DataLoader(val_dataset_ae, batch_size=len(val_dataset),
                               shuffle=False, num_workers=0)
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for batch_idx, batch_x in enumerate(tr_dataloader):
            optimizer.zero_grad()
            batch_x_tensor = batch_x.to(device)
            seq_pred = model(batch_x_tensor)
            # print(seq_pred.shape)
            loss = criterion(seq_pred, batch_x_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            va_x = next(va_dataloader.__iter__())
            va_x_tensor = va_x.to(device)
            seq_pred = model(va_x_tensor)
            loss = criterion(seq_pred, va_x_tensor)
            val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)

    return model, model.eval()


area_list = []

for j in range(3, 4):

    seed = 10 ** j
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    print('seed', seed)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model = model.to(device)

    model, history = train_model(
        model, train_dataset, val_dataset, n_epochs=epochs, batch_size=batch)

    # save model
    MODEL_PATH = os.path.join('model_path', dataset_name)
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model, os.path.join(MODEL_PATH, 'ae_model.pth'))
    torch.save(model.encoder, os.path.join(MODEL_PATH, 'enc_model.pth'))
    torch.save(model.decoder, os.path.join(MODEL_PATH, 'dec_model.pth'))

    # bring test data
    # test_dataset_x = pd.read_csv('dataA_test_x.csv', index_col=0)
    # test_dataset = torch.tensor(test_dataset_x.values)
    # test_dataset = test_dataset.unsqueeze(dim=2)
    # test_dataset = test_dataset.float()

    test_dataset_x = np.load(os.path.join(mts_path, f'x_test_{numas}.npy'))
    # test_dataset_x = test_dataset_x.transpose(0, 2, 1)
    # test_dataset_x = TimeSeriesScalerMeanVariance().fit_transform(test_dataset_x)
    y_test = np.load(os.path.join(mts_path, f'y_test_{numas}.npy'))
    test_dataset = torch.tensor(test_dataset_x)
    test_dataset = test_dataset.unsqueeze(dim=2)
    test_dataset = test_dataset.float()

    test_dataset_ae = AutoencoderDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset_ae, batch_size=len(test_dataset),
                                 shuffle=False, num_workers=0)

    print(np.sum(y_test) / len(y_test))
    test_losses = []
    model = model.eval()
    with torch.no_grad():
        test_x = next(test_dataloader.__iter__())
        test_x_tensor = test_x.to(device)
        test_seq_pred = model(test_x_tensor)

    # reconstruction error
    predictions = np.array(test_seq_pred.cpu())
    # predictions = predictions.reshape(test_seq_pred.shape[0], test_seq_pred.shape[1])
    predictions = predictions.reshape(test_seq_pred.shape[0], -1)
    # print(predictions.shape)
    # mse = np.mean(np.power(np.array(test_dataset.squeeze()) - predictions, 2), axis=1)
    mse = np.mean(
        np.power(
            np.array(
                test_dataset.reshape(
                    test_dataset.shape[0], -1)) - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': y_test[:test_seq_pred.shape[0]].squeeze()})
    groups = error_df.groupby('True_class')
    loss_list = mse.tolist()

    def roc(loss_list, threshold):
        # print(threshold)
        test_score_df = pd.DataFrame(index=range(len(loss_list)))
        test_score_df['loss'] = loss_list
        test_score_df['y'] = y_test[:test_seq_pred.shape[0]].squeeze()
        test_score_df['threshold'] = threshold
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        # test_score_df['t'] = [x[47] for x in test_dataset.x]  # x[59]

        start_end = []
        state = 0
        for idx in test_score_df.index:
            if state == 0 and test_score_df.loc[idx, 'y'] == 1:
                state = 1
                start = idx
            if state == 1 and test_score_df.loc[idx, 'y'] == 0:
                state = 0
                end = idx
                start_end.append((start, end))

        for s_e in start_end:
            if sum(test_score_df[s_e[0]:s_e[1] + 1]['anomaly']) > 0:
                for i in range(s_e[0], s_e[1] + 1):
                    test_score_df.loc[i, 'anomaly'] = 1

        actual = np.array(test_score_df['y'])
        predicted = np.array([int(a) for a in test_score_df['anomaly']])

        return actual, predicted

    # AUROC

    # threshold = 1,2,3....100
    # final_loss = [(loss / timesteps) for loss in loss_list]
    final_loss = loss_list
    final_loss = pd.array(final_loss)
    min_value = np.min(final_loss)
    max_value = np.max(final_loss)
    intervals = (max_value - min_value) / 100
    # [loss / timesteps for loss in loss_list]
    threshold_list = []
    best_f1 = 0
    best_pre = 0
    best_re = 0
    best_thresh = 0
    for i in range(101):
        threshold = min_value + i * intervals
        threshold_list.append(threshold)

    final_actual11 = []
    final_predicted11 = []

    TPR = []
    FPR = []
    PRECISION = []
    F1 = []
    ACC = []
    for i in range(len(threshold_list)):
        ac, pr = roc(error_df.Reconstruction_error, threshold_list[i])
        final_actual11.append(ac)
        final_predicted11.append(pr)

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # compare final_actual11[i] and final_predicted11[i]
        for j in range(len(final_actual11[i])):
            if final_actual11[i][j] == 1 and final_predicted11[i][j] == 1:
                TP += 1
            elif final_actual11[i][j] == 1 and final_predicted11[i][j] == 0:
                FN += 1
            elif final_actual11[i][j] == 0 and final_predicted11[i][j] == 1:
                FP += 1
            elif final_actual11[i][j] == 0 and final_predicted11[i][j] == 0:
                TN += 1
        if TP + TN + FP + FN == 0:
            acc = 0.0
        else:
            acc = (TP + TN) / (TP + TN + FP + FN)
        if TP + FN == 0:
            tpr = 0.0
        else:
            tpr = TP / (TP + FN)
        if FP + TN == 0:
            fpr = 0.0
        else:
            fpr = FP / (FP + TN)
        if TP + FP == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        if tpr + precision == 0:
            f1_score = 0.0
        else:
            f1_score = 2.0 * tpr * precision / (tpr + precision)

        if best_f1 < f1_score:
            best_f1 = f1_score
            best_re = tpr
            best_pre = precision
            best_thresh = threshold_list[i]
        ACC.append(acc)
        TPR.append(tpr)
        FPR.append(fpr)
        PRECISION.append(precision)
        F1.append(f1_score)

    area = auc(FPR, TPR)
    print('area under curve:', area)
    area_list.append(area)
    print(max(ACC))
    print(best_f1, best_re, best_pre)
print(area_list)
