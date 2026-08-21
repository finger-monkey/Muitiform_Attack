from __future__ import absolute_import
import os.path as osp
import numpy as np
from ..utils.data import Dataset
from ..utils.serialization import read_json

def _pluck(identities, indices, relabel=False):
    ret=[]
    for label,pid in enumerate(indices):
        for camid, images in enumerate(identities[pid]):
            for fname in images:
                ret.append((fname, label if relabel else pid, camid))
    return ret

class Llcm(Dataset):
    
    def __init__(self, root, split_id=0, num_val=0.1, download=True):
        super(Llcm, self).__init__(root, split_id)
        required=('images','meta.json','splits.json')
        if not all(osp.exists(osp.join(root,x)) for x in required):
            raise RuntimeError('LLCM is not converted; run convert_llcm_to_market.py')
        self.load(num_val)

    def load(self, num_val=0.1, verbose=True):
        splits=read_json(osp.join(self.root,'splits.json'))
        if self.split_id >= len(splits): raise ValueError('split_id exceeds total splits')
        self.split=splits[self.split_id]
        trainval=np.asarray(self.split['trainval']); np.random.shuffle(trainval)
        nval=int(round(len(trainval)*num_val)) if isinstance(num_val,float) else num_val
        train=sorted(trainval[:-nval]) if nval else sorted(trainval)
        val=sorted(trainval[-nval:]) if nval else []
        self.meta=read_json(osp.join(self.root,'meta.json')); ids=self.meta['identities']
        self.train=_pluck(ids,train,True); self.val=_pluck(ids,val,True)
        self.trainval=_pluck(ids,trainval,True)
        self.num_train_ids=len(train); self.num_val_ids=len(val); self.num_trainval_ids=len(trainval)
        def parse(names):
            return [(f, int(osp.splitext(f)[0].split('_')[0]), int(osp.splitext(f)[0].split('_')[1])) for f in names]
        self.query=parse(self.meta['query_fnames']); self.gallery=parse(self.meta['gallery_fnames'])
        if verbose:
            print('Llcm dataset loaded')
            print('  trainval | {:5d} | {:8d}'.format(self.num_trainval_ids,len(self.trainval)))
            print('  query    | {:5d} | {:8d}'.format(len(self.split['query']),len(self.query)))
            print('  gallery  | {:5d} | {:8d}'.format(len(self.split['gallery']),len(self.gallery)))
