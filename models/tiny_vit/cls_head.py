import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, in_dim, num_classes=1000):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)