import torch
import torch.nn as nn

# 定义模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, 9)
        self.conv2 = nn.Conv1d(64, 32, 5)
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.conv4 = nn.Conv1d(32, 16, 5)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(2, 2, padding=1)
        self.fc1 = nn.Linear(256, 256)  # why 256? 
        self.fc2 = nn.Linear(256, 108)  # 新增一个全连接层
        self.dropout = nn.Dropout(0.5)  # Dropout 层，概率为0.5
        self.fc3 = nn.Linear(108, 4)  # 这是原来的第二个全连接层，现在变成第三个

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x