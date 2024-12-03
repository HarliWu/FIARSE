import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

from typing import Any, Literal

import sys
sys.path.append('./')
from model_utils import *
from resnet import *


class MaskedBasicBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1, 
                 norm_layer=MaskedBatchNorm):
        super().__init__(in_planes, planes, stride, norm_layer, \
                         conv_layer=MaskedConv2d, 
                         sequential=MaskedSequential)
    
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def set_threshold(self, value, bern=False):
        self.conv1.set_threshold(value, bern)
        self.conv2.set_threshold(value, bern)
        self.shortcut.set_threshold(value, bern)

    @property
    def size(self):
        return self.conv1.size + self.conv2.size + self.shortcut.size
    
    def to_vector(self, attr:Literal['param', 'score']):
        vectors = []

        for v in [self.conv1.to_vector(attr), 
                  self.conv2.to_vector(attr), 
                  self.shortcut.to_vector(attr)]:
            if v is not None:
                vectors += v

        return vectors


class BetaResNet(ResNet):
    '''
        We generate the mask here 
        element-wise
    '''
    
    def __init__(self, block, num_blocks, n_class=10, 
                 norm_layer=MaskedBatchNorm):
        super().__init__(block, num_blocks, n_class, norm_layer, 
                         conv_layer=MaskedConv2d, 
                         linear_layer=MaskedLinear, 
                         sequential=MaskedSequential)
        
        self.layer_wise_score = self.to_vector('score', layerwise=True)

    def del_mask(self):
        if hasattr(self, 'mask'):
            delattr(self, 'mask')

    def copy_from(self, model):
        if hasattr(model, 'mask'):
            self.mask  = model.mask.clone().detach()

    def generate_mask(self, model_size=1.0, topk=False, bern=False):
        scores = torch.abs(self.to_vector('param', layerwise=False))

        if topk:
            # Find the threshold via TopK
            self.mask = torch.zeros_like(scores)
            numel = len(self.mask)
            
            topk_result = torch.topk(scores, k=int(numel*model_size))
            self.mask[topk_result.indices] = 1
            self.threshold = topk_result.values[-1]
            self.set_threshold(self.threshold, bern=bern)
        
        else:
            self.threshold = torch.tensor(0.)
            self.mask = torch.ones_like(scores)
            self.set_threshold(torch.tensor(0.), bern=bern)
        
        # return model size 
        size = torch.sum((scores >= self.threshold)) / len(self.mask)
        print(self.threshold, size)
        return torch.sum((scores >= self.threshold)) / len(self.mask)
        
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

    def set_threshold(self, value, bern=False):
        self.conv1.set_threshold(value, bern)
        self.layer1.set_threshold(value, bern)
        self.layer2.set_threshold(value, bern)
        self.layer3.set_threshold(value, bern)
        self.layer4.set_threshold(value, bern)
        self.linear.set_threshold(value, bern)

    def to_vector(self, attr:Literal['param', 'score', 'grad'], layerwise=False):
        vectors = []
        for layer in [self.conv1, self.layer1, self.layer2, 
                      self.layer3, self.layer4, self.linear]:
            vectors = vectors + layer.to_vector(attr)
        
        if layerwise:
            return vectors
        
        return torch.concat(vectors)
        