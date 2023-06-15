import torch
from torch import nn
from torch.nn import functional as F


# 2 Layers CNN
class CNN_1D_2L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.classes = 6
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (5,), stride=1, padding=2),  # [N, C_out, L_out]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)  # [64, 64, 250]
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),  # [64, 128, 250]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(2, stride=2)  # [64, 128, 125]
        )

        self.layer3 = nn.Sequential(
            nn.Linear((self.n_in // 4) * 128, 128),
            nn.ReLU(),
            nn.Linear(128, self.classes),
        )

        # 最后线性层的输出应该是分的类别
        # self.linear1 = nn.Linear((self.n_in // 4) * 128, self.classes)

    def forward(self, x):
        # print('x', x.size())
        x = x.view(-1, 1, self.n_in)  # [N, C_in, L_in]
        # print('x = x.view(-1, 1, self.n_in)', x.size())
        x = self.layer1(x)
        # print('x = self.layer1(x)', x.size()) # [64, 64, 250]
        x = self.layer2(x)  # [64, 128, 125]
        # print('x = self.layer2(x)', x.size())
        # x = x.view(-1, self.n_in*128//4)
        # 这里应该先整除，因为输入数据是[sample, 128, features//4]
        x = x.view(-1, (self.n_in // 4) * 128)
        # print('x = x.view(-1, self.n_in*128//4)', x.size())
        return self.layer3(x)


# 2 Layers CNN
class CNN_3D_2L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.classes = 6
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 64, (5, 5, 5), padding=(2,2,2)),  # [N, C_out, L_out]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, (5, 5, 5), padding=(2,2,2)),  # [64, 128, 250]
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(self.n_in//2 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, self.classes),
        )

        # 最后线性层的输出应该是分的类别
        # self.linear1 = nn.Linear(self.n_in * 128, self.classes)




    def forward(self, x):
        # print('x', x.size())
        # self.n_in = 91，可以分为13*7
        x = x.view(-1, 1, self.n_in, 1, 1)  # [N, C_in, L_in]
        # print('x = x.view(-1, 1, self.n_in)', x.size())
        x = self.layer1(x)
        # print('x = self.layer1(x)', x.size()) # [64, 64, 250]
        x = self.layer2(x)  # [64, 128, 125]
        # print('x = self.layer2(x)', x.size())
        # x = x.view(-1, self.n_in*128//4)
        # 这里应该先整除，因为输入数据是[sample, 128, features//4]
        x = x.view(-1, self.n_in//2 * 128)
        # print('x = x.view(-1, self.n_in*128//4)', x.size())

        # return self.linear1(x)
        return self.layer3(x)



# 3 Layers CNN
class CNN_1D_3L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (5,), stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, (5,), stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
        )

        self.linear1 = nn.Linear((self.n_in // 8) * 128, 6)

    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, (self.n_in // 8) * 128)
        return self.linear1(x)