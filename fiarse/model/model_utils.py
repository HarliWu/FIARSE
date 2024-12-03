import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from typing import Any, Literal


class BatchNorm_no_tracking(nn.BatchNorm2d):
    def __init__(self, num_features: int):
        super().__init__(num_features, momentum=None, track_running_stats=False)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_groups=2, num_channels=num_channels)


class InstanceNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_groups=num_channels, num_channels=num_channels)


class Bern(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        # mask on the parameters greater than threshold
        scalar = 1
        ctx.save_for_backward(scores, scalar*threshold)
        return (scores>=threshold)

    @staticmethod
    def backward(ctx, grad_output):
        scores, threshold = ctx.saved_tensors
        grad = 2 * threshold / torch.pow(scores+threshold, 2)
        grad = torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
        mask = (scores>=threshold)
        return grad_output*grad*mask, None


class MaskedLinear(nn.Linear):
    """
        Implementation of masked linear layer, with training strategy in
        https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
    """
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

        self.threshold = torch.tensor(0.)
        self.bern = True        # Apply Bern() function

    def forward(self, x):    
        if self.bern:    
            weight_mask = Bern.apply(torch.abs(self.weight), self.threshold)
            bias_mask   = Bern.apply(torch.abs(self.bias), self.threshold)
        else:
            weight_mask = (torch.abs(self.weight) >= self.threshold)
            bias_mask = (torch.abs(self.bias) >= self.threshold)
        
        effective_weight = self.weight * weight_mask
        effective_bias   = self.bias * bias_mask
        
        return F.linear(x, effective_weight, effective_bias)

    def set_threshold(self, value, bern=True):
        self.threshold, self.bern = value, bern

    @property
    def size(self):
        return (np.prod(self.weight.shape) if self.weight is not None else 0) \
                + (np.prod(self.bias.shape) if self.bias is not None else 0)
    
    def to_vector(self, attr:Literal['param', 'score', 'grad']):
        # weight is not None 
        if attr == 'param':
            vector = self.weight.view(-1)
        elif attr == 'score':
            vector = torch.randn_like(self.weight).view(-1)
        elif attr == 'grad':
            vector = self.weight.grad.view(-1)
        else:
            raise ValueError
            
        if self.bias is not None:
            if attr == 'param':
                vector = torch.concat([vector, self.bias.view(-1)])
            elif attr == 'score':
                vector = torch.concat([vector, torch.randn_like(self.bias).view(-1)])
            elif attr == 'grad':
                vector = torch.concat([vector, self.bias.grad.view(-1)])

        return [vector]


class MaskedConv2d(nn.Conv2d):
    """
        Implementation of masked convolutional layer, with training strategy in
        https://proceedings.neurips.cc/paper/2019/file/
        1113d7a76ffceca1bb350bfe145467c6-Paper.pdf
    """
    def __init__(self, in_features: int, out_features: int, kernel_size, **kwargs):
        super().__init__(in_features, out_features, kernel_size=kernel_size, **kwargs)

        self.threshold = torch.tensor(0.)
        self.bern = True        # Apply Bern() function

    def forward(self, x):        
        if self.bern:
            weight_mask = Bern.apply(torch.abs(self.weight), self.threshold)
        else:
            weight_mask = (torch.abs(self.weight) >= self.threshold)

        effective_weight = self.weight * weight_mask
        
        if self.bias is not None:
            if self.bern:
                bias_mask = Bern.apply(torch.abs(self.bias), self.threshold)
            else:
                bias_mask = (torch.abs(self.bias) >= self.threshold)
                
            effective_bias = self.bias * bias_mask
        else:
            effective_bias = None
        
        return self._conv_forward(x, effective_weight, effective_bias)

    def set_threshold(self, value, bern=True):
        self.threshold, self.bern = value, bern

    @property
    def size(self):
        return (np.prod(self.weight.shape) if self.weight is not None else 0) \
                + (np.prod(self.bias.shape) if self.bias is not None else 0)
    
    def to_vector(self, attr:Literal['param', 'score', 'grad']):
        # weight is not None 
        if attr == 'param':
            vector = self.weight.view(-1)
        elif attr == 'score':
            vector = torch.randn_like(self.weight).view(-1)
        elif attr == 'grad':
            vector = self.weight.grad.view(-1)
        else:
            raise ValueError
            
        if self.bias is not None:
            if attr == 'param':
                vector = torch.concat([vector, self.bias.view(-1)])
            elif attr == 'score':
                vector = torch.concat([vector, torch.randn_like(self.bias).view(-1)])
            elif attr == 'grad':
                vector = torch.concat([vector, self.bias.grad.view(-1)])

        return [vector]


class MaskedEmbedding(nn.Embedding):
    def forward(self, input):
        if self.bern:
            weight_mask = Bern.apply(torch.abs(self.weight), self.threshold)
        else:
            weight_mask = (torch.abs(self.bias) >= self.threshold)

        effective_weight = self.weight * weight_mask

        return F.embedding(
            input, effective_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def set_threshold(self, value, bern=True):
        self.threshold, self.bern = value, bern

    @property
    def size(self):
        return self.weight.numel()
    
    def to_vector(self, attr:Literal['param', 'score', 'grad']):
        # weight is not None 
        if attr == 'param':
            vector = self.weight.view(-1)
        elif attr == 'score':
            vector = torch.randn_like(self.weight).view(-1)
        elif attr == 'grad':
            vector = self.weight.grad.view(-1)
        else:
            raise ValueError
        
        return [vector]


class MaskedGroupNorm(GroupNorm):
    size = 0
    
    def forward(self, x):
        return super().forward(x)


class MaskedInstanceNorm(InstanceNorm):
    size = 0
    
    def forward(self, x):
        return super().forward(x)


class MaskedBatchNorm(nn.BatchNorm2d):
    size = 0
    
    def forward(self, x):
        return super().forward(x)


class MaskedBatchNorm_no_tracking(MaskedBatchNorm):
    size = 0
    
    def __init__(self, num_features: int):
        super().__init__(num_features, momentum=None, 
                         track_running_stats=False)


class MaskedSequential(nn.Sequential):
    def set_threshold(self, value, bern=True):
        for module in self:
            if module.size != 0:
                module.set_threshold(value, bern=bern)
    
    @property
    def size(self):
        ans = 0
        for module in self:
            ans = ans + module.size
        return ans
    
    def to_vector(self, attr:Literal['param', 'score', 'grad']):
        temp = []

        for module in self:
            if callable(getattr(module, 'to_vector', None)):
                tensor = module.to_vector(attr=attr)
                temp = temp + tensor

        return temp
