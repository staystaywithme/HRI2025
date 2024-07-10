import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from net import LSTMNetwork
from torch.utils.data import DataLoader, TensorDataset
#load data
AC = np.load('C:\Github_LIU\SII2025\process_data\classifier\AC.npy')
AD = np.load('C:\Github_LIU\SII2025\process_data\classifier\AD.npy')
BC = np.load('C:\Github_LIU\SII2025\process_data\classifier\BC.npy')
BD = np.load('C:\Github_LIU\SII2025\process_data\classifier\BD.npy')
#split data into training and testing and validation
X = np.concatenate((AC, AD, BC, BD), axis=0)
AC_labels = [0] * len(AC)
AD_labels = [1] * len(AD)
BC_labels = [2] * len(BC)
BD_labels = [3] * len(BD)
y = np.concatenate((AC_labels, AD_labels, BC_labels, BD_labels), axis=0)
#归一化
def min_max_normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
X = min_max_normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, X_val.shape,y_train.shape, y_test.shape, y_val.shape)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LSTM network definition
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = 12
hidden_size = 64
output_size = 4

model = LSTMNetwork(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Convert data to PyTorch tensors (no need to unsqueeze since data is already in (batch_size, sequence_length, input_size) format)
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
y_val = torch.LongTensor(y_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train model
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        _, predicted = torch.max(val_outputs.data, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy.item():.4f}')

# Test
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (predicted == y_test).float().mean()
print(f'Test Accuracy: {test_accuracy.item():.4f}')