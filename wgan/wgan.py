import torch
import torch.nn as nn
import torch.functional as F

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.l1 = nn.Linear(784, 800)
        self.l2 = nn.Linear(800, 1)

    def forward(self, x):
        x = nn.ReLU(self.l1(x))
        x = self.l2(x)
        return x


