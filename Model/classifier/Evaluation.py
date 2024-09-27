import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset
import matplotlib.pyplot as plt
from lstm_cnn_net import lstm_cnn
from lstm_net import LSTM
from cnn_net import MyNet
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

model=torch.load('89lstm.pth')

