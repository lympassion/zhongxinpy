import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm





def normalize(column):
    mean = column.mean()
    std = column.std()
    return (column - mean) / std

    # max_v = column.max()
    # min_v = column.min()
    # return column.apply(lambda x: (x - min_v) / (max_v -min_v))

def process_feature(df):
    """
    原始数据中最多某一列的缺失值为11%
    对缺失值少于20%的列，用列均值进行填充
    :param df:
    :return:
    """
    # miss_num = df.isna().sum()
    # miss_num_over = {}
    # for i in range(105):
    #     i = str(i)
    #     if miss_num['feature'+i] >= df.shape[0] * 0.2:  # 缺失值大于20%
    #         df.drop('feature'+i, axis=1, inplace=True)
    #         miss_num_over['feature'+i] = miss_num['feature'+i]
    #     else:
    #         df.fillna(df.mean()['feature'+i], inplace=True)
    # miss_num_over['df'] = df

    feature_useful = []
    for i in range(105):
        index = 'feature' + str(i)
        df.fillna(df.mean()[index], inplace=True)
        if len(df[index].unique()) > df.shape[0] * 0.1:
            # feature不是重复出现才可以反向传播
            feature_useful.append(index)

    df[feature_useful] = df[feature_useful].apply(normalize)
    return df, feature_useful


def train_valid(df):
    """
    train:valid:test =
    :param df:
    :return:
    """

    train_threshold = int(df.shape[0] * 0.9)
    train = df[0:train_threshold]
    valid  = df[train_threshold:]

    return [train, valid]


def get_loader(train_valid_test_ls, df_test, feature_useful, bs):

    print('train head:')
    print(train_valid_test_ls[0][feature_useful].head())
    train_features_tensor = torch.tensor(train_valid_test_ls[0][feature_useful].values, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_valid_test_ls[0]['label'].values, dtype=torch.long)
    valid_features_tensor = torch.tensor(train_valid_test_ls[1][feature_useful].values, dtype=torch.float32)
    valid_label_tensor = torch.tensor(train_valid_test_ls[1]['label'].values, dtype=torch.long)
    print('train', type(train_features_tensor))

    print('test head:')
    print(df_test[feature_useful].head())
    test_features_tensor = torch.tensor(df_test[feature_useful].values, dtype=torch.float32)
    print('test', type(test_features_tensor))



    train_ds = TensorDataset(train_features_tensor, train_label_tensor)
    valid_ds = TensorDataset(valid_features_tensor, valid_label_tensor)
    # test_ds = TensorDataset(test_features_tensor)
    test_ds = test_features_tensor

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
        DataLoader(test_ds, batch_size=bs)
    )