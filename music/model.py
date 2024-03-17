import torch
from torch import nn

class model(nn.Module):
   def __init__(self, hid_dim):
      super().__init__()
      self.linear = nn.Linear(hid_dim)   
  def forward():
        return self.linear
