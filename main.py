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
import json

from pathlib import Path

from data_preprocess import *
from model_about import *
import nn_model

#%%

working_dir = Path('.')
DATA_PATH = Path("./zx_data/train.csv")
DATA_TEST_PATH = Path("./zx_data/pre_contest_test1.csv")

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
df_test = pd.read_csv(DATA_TEST_PATH)

# print(set(df['label']))
# print(df.info())
# print(df.isna().sum())




if __name__ == '__main__':

    df, feature_useful = process_feature(df)
    df_test, _ = process_feature(df_test)

    train_valid_ls = train_valid(df)
    print('feature_useful:\n', len(feature_useful), feature_useful)


    print('before get loader')
    train_dl, valid_dl, test_dl = get_loader(train_valid_ls, df_test, feature_useful, bs)


    # Training with Adams Optimizer
    CNN = 2  # 2: CNN_1D_2L else: CNN_1D_3L
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

    title = str(CNN) + 'layer  ' + '5 5' + 'kernel' + 'have dropout' + ' without normal'
    metrics.plot(y = ['val_loss','val_accuracy','train_loss', 'train_accuracy'])
    plt.title(title)
    plt.xlabel('accuracy' + ('%.5f' % metrics.at[epochs-1, 'val_accuracy']))
    plt.savefig('./images/' + title)
    plt.show()
    print('准确率', metrics.at[epochs-1, 'val_accuracy'])


    print('测试')
    model2.eval()  # 评估模式
    pred_np = test_pred_json(model2, test_dl)
    pred_dict = {}
    for i in range(df_test.shape[0]):
        pred_dict[str(i)] = int(pred_np[i])
    pred_json = json.dumps(pred_dict)
    print(type(pred_json))
    print(set(pred_np.tolist()), '\n', pred_json)
    with open("submit.json", "w") as f:
        f.write(pred_json)
    f.close()



