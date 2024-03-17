import torch
from torch import nn

class model(nn.Module):
   def __init__(self, hid_dim):
      super().__init__()
      self.linear = nn.Linear(hid_dim, hid_dim)
      self.act = nn.ReLU()
  def forward(self, x):
     for i in range(num_layers):
        x = self.linear(x)
        x = self.act(x)
     return x
