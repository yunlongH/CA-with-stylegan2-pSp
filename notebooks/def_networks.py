
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


