import torch

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd



def loss_batch(model, loss_func, xb, yb, opt=None):

    out = model(xb)
    # print('aaaaa', yb.size(), yb, yb.squeeze(1))
    # print('aaaaa', yb.size(), yb)
    loss = loss_func(out, yb)
    # loss = loss_func(out+1e-10, yb)

    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False):

    # Initialize dic to store metrics for each epoch.
    metrics_dic = {}
    metrics_dic['epoch'] = []
    metrics_dic['train_loss'] = []
    metrics_dic['train_accuracy'] = []
    metrics_dic['val_loss'] = []
    metrics_dic['val_accuracy'] = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_examples = 0
        for xb, yb in train_dl:
            # xb大小[row, col], row是train的行数，col是(features个数)105
            xb, yb = xb.to(device), yb.to(device)
            loss, batch_size, pred = loss_batch(model, loss_func, xb, yb, opt)
            if train_metric == False:
                train_loss += loss
                num_examples += batch_size

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, _ = validate(model, valid_dl, loss_func)
            if train_metric:
                train_loss, train_accuracy, _ = validate(model, train_dl, loss_func)
            else:
                train_loss = train_loss / num_examples

        metrics_dic['epoch'].append(epoch)
        metrics_dic['val_loss'].append(val_loss)
        metrics_dic['val_accuracy'].append(val_accuracy)
        metrics_dic['train_loss'].append(train_loss)
        metrics_dic['train_accuracy'].append(train_accuracy)

    metrics = pd.DataFrame.from_dict(metrics_dic)

    return model, metrics


def validate(model, dl, loss_func):
    total_loss = 0.0
    total_size = 0
    predictions = []
    y_true = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss, batch_size, pred = loss_batch(model, loss_func, xb, yb)
        total_loss += loss * batch_size
        total_size += batch_size

        predictions.append(pred)
        y_true.append(yb.cpu().numpy())
    mean_loss = total_loss / total_size
    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean((predictions == y_true))
    return mean_loss, accuracy, (y_true, predictions)

def test_pred_json(model, test_dl):

    predictions = np.array([])

    for xb in test_dl:
        out = model(xb)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        predictions = np.append(predictions, pred)

    return predictions

