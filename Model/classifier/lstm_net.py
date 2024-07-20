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
        self.relu = nn.ReLU()

    def forward(self, x, hidden_state, cell_state):
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (h, c) = self.lstm(x, (hidden_state, cell_state))
        out = self.output_layer(out[:, -1, :])
        return out, h, c