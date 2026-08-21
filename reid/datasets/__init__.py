from __future__ import absolute_import
import warnings


from .CnMix import CnMix
from .Sketch import Sketch
from .regdb import Regdb
from .sysu import Sysu
from .llcm import Llcm


__factory = {

    'CnMix': CnMix,
    'regdb_v2': Regdb,
    'Sketch': Sketch,
    'sysu_v2': Sysu,
    'llcm_v2': Llcm
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    

















    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
