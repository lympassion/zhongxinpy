import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Pytorch
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

from pathlib import Path

from data_preprocess import *
from model_about import *
import nn_model

#%%

working_dir = Path('.')
DATA_PATH = Path("./zx_data/train.csv")
save_model_path = working_dir / 'zx_model'



if not save_model_path.exists():
    save_model_path.mkdir(parents=True)



#### HYPERPARAMETERS ####
epochs = 50
bs = 64
lr = 0.001
wd = 1e-5
betas=(0.99, 0.999)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random_seed = 42

df = pd.read_csv(DATA_PATH)

# print(set(df['label']))
# print(df.info())
# print(df.isna().sum())




if __name__ == '__main__':

    df, feature_useful = process_feature(df)
    train_valid_test_ls = train_valid_test(df)
    print('feature_useful:\n', len(feature_useful), feature_useful)


    print('before get loader')
    train_dl, valid_dl = get_loader(train_valid_test_ls, feature_useful, bs)

    # Training with Adams Optimizer
    ## Instantiate model, optimizer and loss function
    CNN = 3  # 2: CNN_1D_2L else: CNN_1D_3L
    if CNN==2:
        model = nn_model.CNN_1D_2L(len(feature_useful))
    else:
        model = nn_model.CNN_1D_3L(len(feature_useful))

    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    loss_func = CrossEntropyLoss()
    model, metrics = fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=True)
    print(metrics)


    # Save trained model
    if CNN==2:
        torch.save(model.state_dict(), save_model_path / 'model2.pth')
        model2 = nn_model.CNN_1D_2L(len(feature_useful))
        model2.load_state_dict(torch.load(save_model_path / 'model2.pth'))
    else:
        torch.save(model.state_dict(), save_model_path / 'model3.pth')
        model2 = nn_model.CNN_1D_3L(len(feature_useful))
        model2.load_state_dict(torch.load(save_model_path / 'model3.pth'))

    model2.eval()
    print('开始验证')
    mean_loss, accuracy, _ = validate(model, valid_dl, loss_func)
    print(mean_loss, accuracy)
    metrics.plot(y = ['val_loss','val_accuracy','train_loss', 'train_accuracy'])
    # title = '3'+'layer  '+'9 9 9'+'kernel' + 'without dropout norm'
    title = str(CNN)+'layer  '+'5 5 5'+'kernel' + 'have dropout' + ' without normal'

    plt.title(title)
    plt.xlabel('accuracy'+('%.5f'%accuracy))
    plt.savefig('./images/' + title)
    plt.show()

