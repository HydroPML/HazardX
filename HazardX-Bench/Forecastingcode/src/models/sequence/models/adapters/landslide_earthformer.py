import torch.nn as nn
import torch.nn.functional as F
from gphyt.model.earthformer_official import EarthFormerScratch
from .landslide_common import physical,assert_output
class LandslideEarthFormer(nn.Module):
    """Official CuboidTransformerModel with native THW position encoding."""
    def __init__(self,official_root="",**_):
        super().__init__(); template=EarthFormerScratch(200,200,official_root=official_root)
        cls=template.backbone.__class__; cfg=dict(template.official_config); cfg["input_shape"]=(9,216,216,3); cfg["target_shape"]=(9,216,216,2)
        self.backbone=cls(**cfg); self.spatial_multiple=24
    def forward(self,batch):
        x=physical(batch).permute(0,2,3,1).unsqueeze(1).expand(-1,9,-1,-1,-1); x=F.pad(x,(0,0,0,16,0,16))
        y=self.backbone(x)[:,:,:200,:200,:]
        return assert_output(y[:,1:].permute(0,1,4,2,3).contiguous(),batch)

