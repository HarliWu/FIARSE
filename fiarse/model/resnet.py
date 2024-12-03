# -*- coding: utf-8 -*-

'''
This is full set for cifar-10 datasets 
Models: ResNet
'''

import torch.nn as nn
import torch.nn.functional as F

############# ResNet #############

class BatchNorm_no_tracking(nn.BatchNorm2d):
    def __init__(self, num_features: int):
        super().__init__(num_features, momentum=None, track_running_stats=False)

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_groups=2, num_channels=num_channels)

class InstanceNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_groups=num_channels, num_channels=num_channels)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d, sequential=nn.Sequential):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(planes)
        
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes)

        self.shortcut = sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            self.shortcut.append(norm_layer(self.expansion*planes))
            
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=BatchNorm_no_tracking, conv_layer=nn.Conv2d, sequential=nn.Sequential):
        super(PreActBlock, self).__init__()
        
        self.norm1 = norm_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.norm2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d, sequential=nn.Sequential):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = norm_layer(planes)

        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        
        self.conv3 = conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.norm3 = norm_layer(self.expansion*planes)

        self.shortcut = sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            self.shortcut.append(norm_layer(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=BatchNorm_no_tracking, conv_layer=nn.Conv2d, sequential=nn.Sequential):
        super(PreActBottleneck, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        
        self.norm2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.norm3 = norm_layer(planes)
        self.conv3 = conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.conv3(F.relu(self.norm3(out)))
        out += shortcut
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_class=10, norm_layer=nn.BatchNorm2d, \
                    conv_layer=nn.Conv2d, linear_layer=nn.Linear, sequential=nn.Sequential):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm_layer(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.linear = linear_layer(512*block.expansion, n_class)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer, sequential):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # layers.append(block(self.in_planes, planes, stride, norm_layer, conv_layer, sequential))
            layers.append(block(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes * block.expansion
        return sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_classes=10, norm_layer=BatchNorm_no_tracking, \
                    conv_layer=nn.Conv2d, linear_layer=nn.Linear, sequential=nn.Sequential):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential)
        self.linear = linear_layer(512*block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer, sequential):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # layers.append(block(self.in_planes, planes, stride, norm_layer, conv_layer, sequential))
            layers.append(block(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
