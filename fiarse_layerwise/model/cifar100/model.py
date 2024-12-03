import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, '../'))

from resnet import *
from resnet_masked import *

######## ResNet without any pruning 

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2], n_class=100)

def ResNet18_gn():
    return ResNet(BasicBlock, [2,2,2,2], n_class=100, norm_layer=GroupNorm)

def ResNet18_in():
    return ResNet(BasicBlock, [2,2,2,2], n_class=100, norm_layer=InstanceNorm)


######## ResNet with element-wise pruning 

def BetaResNet18():
    return BetaResNet(MaskedBasicBlock, [2,2,2,2], n_class=100)

def BetaResNet18_sbn():
    return BetaResNet(MaskedBasicBlock, [2,2,2,2], n_class=100, norm_layer=MaskedBatchNorm_no_tracking)

def BetaResNet18_gn():
    return BetaResNet(MaskedBasicBlock, [2,2,2,2], n_class=100, norm_layer=MaskedGroupNorm)

def BetaResNet18_in():
    return BetaResNet(MaskedBasicBlock, [2,2,2,2], n_class=100, norm_layer=MaskedInstanceNorm)
