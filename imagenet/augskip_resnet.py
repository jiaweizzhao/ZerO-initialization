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


# __all__ = ['zero_resnet50_v2']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
            raise RuntimeError('hadamard transform not supports dim_in != dim_out currently')
        hadamard_matrix = hadamard(dim_in)
        hadamard_matrix = torch.Tensor(hadamard_matrix)

        n = int(np.log2(dim_in))
        normalized_hadamard_matrix = hadamard_matrix / (2**(n / 2))

        self.hadamard_matrix = nn.Parameter(normalized_hadamard_matrix, requires_grad=False)


    def forward(self, x):
        # input is a B x C x N x M
        
        return torch.matmul(x.permute(0,2,3,1), self.hadamard_matrix).permute(0,3,1,2)

class AugSkipBottleneck_no_bn(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AugSkipBottleneck_no_bn, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = zero_relu
        self.downsample = downsample
        self.stride = stride

        if stride != 1:
            self.shortcut = nn.AvgPool2d(1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out + self.bias1a)
        out = out + self.bias1b

        identity2 = out
        out = self.relu(out)
        out = self.conv2(out + self.bias2a)
        out = out + self.bias2b

        if self.shortcut is not None:
            identity2 = self.shortcut(identity2)
        out += identity2

        out = self.relu(out)
        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b

        if self.downsample is not None:
            identity = self.relu(identity)
            identity = self.downsample(identity + self.bias1a)

        out += identity

        return out

class AugSkipBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AugSkipBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = zero_relu
        self.downsample = downsample
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)


        if stride != 1:
            self.shortcut = nn.AvgPool2d(1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        if self.downsample is not None:
            out = self.relu(x)
            out = self.bn1(out)
            identity = out
        else:
            identity = x
            out = self.relu(x)
            out = self.bn1(out)

        out = self.conv1(out)

        identity2 = out
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            identity2 = self.shortcut(identity2)
        out += identity2

        out = self.relu(out)
        out = self.bn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        return out

class AugSkipResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, pre_ac=False, bn=False):
        super(AugSkipResNet, self).__init__()

        self.pre_ac = pre_ac
        self.bn = bn

        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1.type_name = 'conv1'
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = zero_relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.first_transform = nn.Sequential(ChannelPaddingSkip(0,61, scale=1), Hadamard_Transform(64,64))

        if self.bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, AugSkipBottleneck_no_bn) or isinstance(m, AugSkipBottleneck):
                # print('size of conv1', m.conv1.weight.data.size())
                partial_identity_conv1x1(m.conv1.weight)
                nn.init.constant_(m.conv2.weight, 0)
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    partial_identity_conv1x1(m.downsample[0].weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # initialize first conv layer as zero
                if hasattr(m,'type_name'):
                    if 'conv1' in m.type_name:
                        nn.init.constant_(m.weight, 0)

        # double check initialization status
        for name, param in self.named_parameters():
            unique_values = torch.unique(param.data)
            if len(unique_values) > 2 and param.requires_grad:
                print('!!the following is not initialized as zero or one!!')
                print(name)
                print(unique_values)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample = nn.Sequential(downsample, Hadamard_Transform(planes * block.expansion, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # identity = x
        x = self.conv1(x) + self.first_transform(self.maxpool(x))
        x = self.relu(x)
        if self.bn:
            x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.pre_ac:
            x = self.relu(x)
            if self.bn:
                x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def augskip_resnet50_no_bn(**kwargs):
    """Constructs a Fixup-ResNet-50 model.
    """
    model = AugSkipResNet(AugSkipBottleneck_no_bn, [3, 4, 6, 3], pre_ac=True, **kwargs)
    return model

def augskip_resnet50(**kwargs):
    """Constructs a Fixup-ResNet-50 model.
    """
    model = AugSkipResNet(AugSkipBottleneck, [3, 4, 6, 3], pre_ac=True, bn=True, **kwargs)
    return model
