import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function


class SkipConnection(nn.Module):

    def __init__(self, scale=1):
        super(SkipConnection, self).__init__()
        self.scale = scale
    def _shortcut(self, input):
        #needs to be implemented

        return input

    def forward(self, x):
        # with torch.no_grad():
        identity = self._shortcut(x)
        return identity * self.scale

class DeterChannelDropoutSkip(SkipConnection):

    def __init__(self, num_remain_channels, scale=1):
        super(DeterChannelDropoutSkip, self).__init__(scale)
        self.num_remain_channels = num_remain_channels
    
    def _shortcut(self, input):
        # input is (N, C, H, M)
        # and return is (N, C-num_reduce_channels, H, M)
        
        return input[:,0:self.num_remain_channels,:,:]        

class ChannelPaddingSkip(SkipConnection):

    def __init__(self, num_expand_channels_left, num_expand_channels_right, scale=1):
        super(ChannelPaddingSkip, self).__init__(scale)
        self.num_expand_channels_left = num_expand_channels_left
        self.num_expand_channels_right = num_expand_channels_right
    
    def _shortcut(self, input):
        # input is (N, C, H, M)
        # and return is (N, C + num_left + num_right, H, M)
        
        return F.pad(input, (0, 0, 0, 0, self.num_expand_channels_left, self.num_expand_channels_right) , "constant", 0) 
