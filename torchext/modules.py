import torch
import math
import numpy as np

from .functions import *

class CoordConv2d(torch.nn.Module):
  def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
    super().__init__()

    self.conv = torch.nn.Conv2d(channels_in+2, channels_out, kernel_size=kernel_size, padding=padding, stride=stride)

    self.uv = None

  def forward(self, x):
    if self.uv is None:
      height, width = x.shape[2], x.shape[3]
      u, v = np.meshgrid(range(width), range(height))
      u = 2 * u / (width - 1) - 1
      v = 2 * v / (height - 1) - 1
      uv = np.stack((u, v)).reshape(1, 2, height, width)
      self.uv = torch.from_numpy( uv.astype(np.float32) )
    self.uv = self.uv.to(x.device)
    uv = self.uv.expand(x.shape[0], *self.uv.shape[1:])
    xuv = torch.cat((x, uv), dim=1)
    y = self.conv(xuv)
    return y
