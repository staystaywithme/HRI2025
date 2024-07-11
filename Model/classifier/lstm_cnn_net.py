import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        self.conv1 = nn.Conv1d(18, 64, 9)
        self.conv2 = nn.Conv1d(64, 32, 5)
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.conv4 = nn.Conv1d(32, 16, 5)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(2, 2, padding=1)
        self.fc1 = nn.Linear(256, 256)  # 修改这一层的输出特征数
        self.fc2 = nn.Linear(256, 108)  # 新增一个全连接层
        self.dropout = nn.Dropout(0.5)  # Dropout 层，概率为0.5
        self.fc3 = nn.Linear(108, 4)  # 这是原来的第二个全连接层，现在变成第三个

    def forward(self, x):
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (hidden_state, cell_state))
        out = self.output_layer(out[:, -1, :])
        return out