from __future__ import absolute_import
from .inception import *
from .resnet import *
from .PCB import PCB, PCBTrain
from .baseline import ft_net

from .AGW import embed_net
from .DDAG import embed_net2


























__factory = {
    'inception': inception,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'pcb': PCB,
    'pcbt': PCBTrain,
    'baseline':ft_net,
    'AGW': embed_net,
    'DDAG': embed_net2
}













def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    





























    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
