import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class lstm_cnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(lstm_cnn, self).__init__()
        self.conv1 = nn.Conv1d(12, 32, 9)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.conv3 = nn.Conv1d(64, 32, 5)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(2, 2, padding=1)  # 添加池化层

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)  # 修改 input_size 为最后一个 Conv1d 层的输出通道数
        self.output_layer = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将形状调整为 (batch, feature=12, seq=301) 
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = x.permute(0, 2, 1)  # 将形状调整为 (batch, seq, feature) 以输入 LSTM

        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(x, (hidden_state, cell_state))
        out = self.output_layer(out[:, -1, :])

        return out