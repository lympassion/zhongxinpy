import torch.nn as nn
import torch

# 假设原始数据集大小为(100, 1, 1024)
batch_size = 100
n_channels = 1
n_features = 1024

# 将数据集变形为(100, 1, 1024, 1, 1)，其中n_timesteps=128，n_features=8

data = torch.randn(batch_size, n_channels, n_features)
data = data.view(batch_size, n_channels, n_features, 1, 1)


# 定义3D卷积网络
class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.fc1 = nn.Linear(128 * n_features, 256)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        # x = nn.functional.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        x = self.conv2(x)
        x = nn.functional.relu(x)
        # x = nn.functional.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        x = x.view(-1, 128 * n_features)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


# 创建模型实例并进行前向传播
net = Conv3DNet()
output = net(data)