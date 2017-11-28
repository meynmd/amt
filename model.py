import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np

class ConvNet(nn.Module):
    # num_features corresponds to the number of frequency bins
    def __init__(self, window_size, num_features):
        super(ConvNet, self).__init__()

        # Model parameters.
        self.window_size = window_size
        self.num_features = num_features

        # Conv layers.
        self.conv1 = nn.Conv2d(1,50,(5,25),padding=(2,12))
        self.conv2 = nn.Conv2d(50,50,(3,5),padding=(1,2))

        # FC layers.
        self.fc1 = nn.Linear(7*30*50,1000)
        self.fc2 = nn.Linear(1000,200)

        # Output layer.
        self.fc3 = nn.Linear(200,88)