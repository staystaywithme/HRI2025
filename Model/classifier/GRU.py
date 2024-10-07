import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, hidden_state):
        # GRU不需要cell state，所以我们只初始化hidden state
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, h = self.gru(x, hidden_state)
        out = self.output_layer(out[:, -1, :])
        return out, h