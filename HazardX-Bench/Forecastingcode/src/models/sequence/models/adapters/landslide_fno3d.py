import importlib.util,sys,types
from pathlib import Path
import torch.nn as nn
from .landslide_common import explicit_sequence,assert_output
def _official_fnn3d():
    root=Path(__file__).resolve().parents[3]/"Landslidecast-main"/"2D_landslide"/"models"
    package=types.ModuleType("landslidecast_official_models"); package.__path__=[str(root)]; sys.modules.setdefault(package.__name__,package)
    name=package.__name__+".fourier3d"; spec=importlib.util.spec_from_file_location(name,root/"fourier3d.py"); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module.FNN3d
class LandslideFNO3D(nn.Module):
    def __init__(self,**_):
        super().__init__(); FNN3d=_official_fnn3d(); self.backbone=FNN3d([4,8,8,16],[4,8,8,16],[4,4,4,4],layers=[64]*5,fc_dim=128,in_dim=6,out_dim=2,activation="gelu",pad_z=1)
    def forward(self,batch):
        x=explicit_sequence(batch)[:,[1,0,2,3,4,5]].permute(0,3,4,2,1).contiguous(); y=self.backbone(x)[:,:,:,1:,:]
        return assert_output(y.permute(0,3,4,1,2).contiguous(),batch)
