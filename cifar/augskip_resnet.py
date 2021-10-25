'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

import torch
from torch._C import memory_format
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from .skip_connection import *
from scipy.linalg import hadamard

class Zero_Relu(Function):
        
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clamp(min=0)
        return output    
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    
zero_relu = Zero_Relu.apply

def partial_identity_conv1x1(weight):
    tensor = weight.data
    out_dim = tensor.size()[0]
    in_dim = tensor.size()[1]
    ori_dim = tensor.size()
    assert tensor.size()[2] == 1 and tensor.size()[3] == 1
    if out_dim<in_dim:
        i = torch.eye(out_dim).type_as(tensor)
        j = torch.zeros(out_dim,(in_dim-out_dim)).type_as(tensor)
        k = torch.cat((i,j),1)
    elif out_dim>in_dim:
        i = torch.eye(in_dim).type_as(tensor)
        j = torch.zeros((out_dim-in_dim),in_dim).type_as(tensor)
        k = torch.cat((i,j),0)
    else:
        k = torch.eye(out_dim).type_as(tensor)
    k.unsqueeze_(2)
    k.unsqueeze_(3)
    assert k.size() == ori_dim
    
    weight.data = k

class Hadamard_Transform(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Hadamard_Transform, self).__init__()
        if dim_in != dim_out:
            raise RuntimeError('orthogonal transform not supports dim_in != dim_out currently')
        hadamard_matrix = hadamard(dim_in)
        hadamard_matrix = torch.Tensor(hadamard_matrix)

        n = int(np.log2(dim_in))
        normalized_hadamard_matrix = hadamard_matrix / (2**(n / 2))

        self.hadamard_matrix = nn.Parameter(normalized_hadamard_matrix, requires_grad=False)


    def forward(self, x):
        # input is a B x C x N x M
        
        return torch.matmul(x.permute(0,2,3,1), self.hadamard_matrix).permute(0,3,1,2)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = zero_relu

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                Hadamard_Transform(self.expansion*planes,self.expansion*planes))
            self.shortcut[0].type_name = 'conv1x1'

    def forward(self, x):
        out = self.bn1(self.relu(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out += shortcut

        identity = out
        out = self.conv2(self.bn2(self.relu(out)))
        out += identity

        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init='ZerO'):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(512*block.expansion)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.first_transform = nn.Sequential(ChannelPaddingSkip(0,61, scale=1), Hadamard_Transform(64,64))
        
        self.relu = zero_relu
        
        if init == 'ZerO':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, PreActBlock):
                    # initialize every conv layer as zero
                    nn.init.constant_(m.conv1.weight, 0)
                    nn.init.constant_(m.conv2.weight, 0)
                if isinstance(m, nn.Conv2d):
                    # initialize first conv layer as zero
                    if hasattr(m,'type_name'):
                        if 'conv1x1' in m.type_name:
                            # nn.init.constant_(m.weight, 0)
                            partial_identity_conv1x1(m.weight)
                    else:
                        nn.init.constant_(m.weight, 0)

        # check initialization status
        for name, param in self.named_parameters():
            unique_values = torch.unique(param.data)
            if len(unique_values) > 2 and param.requires_grad:
                print('!!!the following is not initialized as zero or one!!')
                print(name)
                print(unique_values)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x) + self.first_transform(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.relu(out)
        out = self.bn2(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def augskip_resnet18():
    return PreActResNet(PreActBlock, [2,2,2,2])