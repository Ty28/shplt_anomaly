# -*- coding:utf-8 -*-

"""
@File    : eval_.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/01/23 17:04
@Function: 
"""
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_model(criterion, model, model_type, val_iter, mode='Validation'):
    """
    Function to run validation on given model
    :param criterion: loss function
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param val_iter: validation dataloader
    :param mode: mode: 'Validation' or 'Test' - depends on the dataloader given.Used for logging
    :return mean validation loss (and accuracy if in clf mode)
    """
    # Validation loop
    model.eval()
    loss_sum = 0
    correct_sum = 0
    predictions, losses = [], []
    with torch.no_grad():
        for data in val_iter:
            if len(data) == 2:
                data, labels = data[0].to(device), data[1].to(device)
            else:
                data = data.to(device)

            model_out = model(data)
            if model_type == 'LSTMAE_CLF':
                model_out, out_labels = model_out
                pred = out_labels.max(1, keepdim=True)[1]
                correct_sum += pred.eq(labels.view_as(pred)).sum().item()
                # Calculate loss
                mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
                loss = mse_loss + ce_loss
            elif model_type == 'LSTMAE_PRED':
                # For S&P prediction
                model_out, preds = model_out
                labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
                preds = preds[:, :-1]  # Take preds up to T-1
                mse_rec, mse_pred = criterion(model_out, data, preds, labels)
                loss = mse_rec + mse_pred
            else:
                # Calculate loss for none clf models
                loss = criterion(model_out, data)
            predictions.append(model_out)
            losses.append(loss.item())
            loss_sum += loss.item()
    val_loss = loss_sum / len(val_iter.dataset)
    val_acc = round(correct_sum / len(val_iter.dataset) * 100, 2)
    acc_out_str = f'; Average Accuracy: {val_acc}' if model_type == 'LSTMAECLF' else ''
    print(f' {mode}: Average Loss: {val_loss}{acc_out_str}')
    return val_loss, predictions, losses


def eval_accuracy(model, X, Y):
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    return (predictions == Y).sum() / Y.size


def grid_search_f1_score(loss_normal, loss_anomaly):
    # find the best threshold of model
    # 遍历所有的可选阈值，取最高的F1score的阈值作为最终阈值
    len_normal = len(loss_normal)
    len_anomaly = len(loss_anomaly)
    best_f1_score = -1.
    best_threshold = 0
    for threshold in loss_normal:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = f1_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    for threshold in loss_anomaly:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = f1_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    return best_f1_score, best_threshold


def grid_search_accuracy_score(loss_normal, loss_anomaly):
    # find the best threshold of model
    # 遍历所有的可选阈值，取最高的F1score的阈值作为最终阈值
    len_normal = len(loss_normal)
    len_anomaly = len(loss_anomaly)
    best_f1_score = -1.
    best_threshold = 0
    for threshold in loss_normal:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = accuracy_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    for threshold in loss_anomaly:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = accuracy_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    return best_f1_score, best_threshold


def grid_search_precision_score(loss_normal, loss_anomaly):
    # find the best threshold of model
    # 遍历所有的可选阈值，取最高的F1score的阈值作为最终阈值
    len_normal = len(loss_normal)
    len_anomaly = len(loss_anomaly)
    best_f1_score = -1.
    best_threshold = 0
    for threshold in loss_normal:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = precision_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    for threshold in loss_anomaly:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = precision_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    return best_f1_score, best_threshold


def grid_search_recall_score(loss_normal, loss_anomaly):
    # find the best threshold of model
    # 遍历所有的可选阈值，取最高的F1score的阈值作为最终阈值
    len_normal = len(loss_normal)
    len_anomaly = len(loss_anomaly)
    best_f1_score = -1.
    best_threshold = 0
    for threshold in loss_normal:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = recall_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    for threshold in loss_anomaly:
        y_true = []
        y_pred = []
        #         print(threshold)
        for i in range(len_normal):
            y_true.append(1)
            if loss_normal[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        for i in range(len_anomaly):
            y_true.append(0)
            if loss_anomaly[i] <= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        #         print(y_true)
        #         print(y_pred)
        f1_score_current = recall_score(y_true, y_pred)
        if best_f1_score <= f1_score_current:
            best_f1_score = f1_score_current
            best_threshold = threshold
    return best_f1_score, best_threshold
