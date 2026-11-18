"""Canonical, leakage-free LandslideCast 2-D HDF5 dataset."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

TIMES=torch.tensor([0,10,20,30,40,50,60,70,80],dtype=torch.float32)
SPLITS={"train":(1,10000),"val":(10001,11000),"test":(11001,12000)}
CHANNELS=("fai","z","h","u")

def scenario_names(split):
    if split not in SPLITS: raise ValueError(split)
    lo,hi=SPLITS[split]
    return tuple(f"x{i}" for i in range(lo,hi+1))

def validate_splits():
    groups={key:set(scenario_names(key)) for key in SPLITS}
    assert {key:len(value) for key,value in groups.items()}=={"train":10000,"val":1000,"test":1000}
    assert all(groups[a].isdisjoint(groups[b]) for a,b in (("train","val"),("train","test"),("val","test")))

class LandslideCast2DDataset(Dataset):
    def __init__(self,h5_path,split):
        self.h5_path=str(Path(h5_path).expanduser().resolve())
        self.split=split
        self.names=scenario_names(split)
        self._h5=None
        validate_splits()
        if not Path(self.h5_path).is_file(): raise FileNotFoundError(self.h5_path)
        import h5py
        with h5py.File(self.h5_path,"r") as handle:
            missing=[name for name in self.names if name not in handle]
            if missing: raise KeyError(f"missing {len(missing)} {split} scenarios; first={missing[:3]}")
    def __len__(self): return len(self.names)
    def _open(self):
        if self._h5 is None:
            import h5py
            self._h5=h5py.File(self.h5_path,"r")
        return self._h5
    @staticmethod
    def canonicalize(raw,name):
        value=torch.as_tensor(np.asarray(raw),dtype=torch.float32)
        if value.ndim==5 and value.shape[0]==1: value=value.squeeze(0)
        if tuple(value.shape)!=(200,200,9,4): raise ValueError(f"{name}: raw shape {tuple(value.shape)} != (200,200,9,4) after optional leading singleton squeeze")
        if not torch.isfinite(value).all(): raise FloatingPointError(f"{name}: NaN/Inf in raw sample")
        return value.permute(1,0,2,3).contiguous() # official x/y transpose
    def __getitem__(self,index):
        name=self.names[index]
        raw=self.canonicalize(self._open()[name][...],name)
        phi=raw[:,:,0,0].clone(); z=raw[:,:,0,1].clone(); h0=raw[:,:,0,2].clone(); u0=raw[:,:,0,3].clone()
        axis=torch.arange(200,dtype=torch.float32)/200.0
        target=raw[:,:,1:,2:4].permute(2,3,0,1).contiguous()
        if tuple(target.shape)!=(8,2,200,200): raise AssertionError(tuple(target.shape))
        return {"scenario":name,"phi":phi,"z":z,"h0":h0,"u0":u0,
                "z_norm":(z-z.min())/2133.0,"h0_norm":h0/141.0,
                "x_grid":axis[:,None].expand(200,200).clone(),
                "y_grid":axis[None,:].expand(200,200).clone(),
                "t_grid":torch.linspace(0,1,9),"target":target}
    def __getstate__(self):
        state=self.__dict__.copy(); state["_h5"]=None; return state
    def close(self):
        if self._h5 is not None: self._h5.close(); self._h5=None
