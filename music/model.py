# it is not final model, it is only to fill this class

import torch
from torch import nn

class model(nn.Module):
   def __init__(self, num_layers, hid_dim):
      super().__init__()
      self.linear = nn.Linear(hid_dim, hid_dim)
      self.act = nn.ReLU()
  def forward(self, x):
     for i in range(num_layers):
        x = self.linear(x)
        x = self.act(x)
     return x
