"""DPOT-S using its official [physical,x,y,t] grid and temporal aggregator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gphyt.model.dpot_s import DPOTSPTFT
from .landslide_common import physical,assert_output

class LandslideDPOT(nn.Module):
    def __init__(self,official_root="",checkpoint_path="",report_path="",**_):
        super().__init__()
        self.core=DPOTSPTFT(200,200,official_root=official_root,checkpoint_path=checkpoint_path,report_path=report_path)
        backbone=self.core.backbone
        source=backbone.time_agg_layer.w.detach()
        backbone.time_agg_layer.w=nn.Parameter(source.repeat(9,1,1)/9.0)
        backbone.time_agg_layer.n_timesteps=9
        old=backbone.out_layer[-1]
        backbone.out_layer[-1]=nn.Conv2d(old.in_channels,18,1)
        backbone.out_channels=2; backbone.out_timesteps=9
        self.loading_report=self.core.loading_report
        for name in ("time_agg_layer.w","out_layer.4.weight","out_layer.4.bias"):
            if name not in self.loading_report.newly_initialized: self.loading_report.newly_initialized.append(name)
        self.loading_report.newly_initialized.sort(); self.loading_report.write(report_path)
    def forward(self,batch):
        fields=physical(batch,("phi","z_norm","h0_norm"))
        x=fields.permute(0,2,3,1).unsqueeze(3).repeat(1,1,1,9,1)
        backbone=self.core.backbone; batch_size,height,width,timesteps,_=x.shape
        hidden=torch.cat((x,backbone.get_grid_3d(x)),dim=-1).contiguous()
        hidden=rearrange(hidden,"b x y t c -> (b t) c x y")
        hidden=backbone.patch_embed.proj(hidden)
        hidden=hidden+F.interpolate(backbone.pos_embed,size=hidden.shape[-2:],mode="bicubic",align_corners=False)
        hidden=rearrange(hidden,"(b t) c x y -> b x y t c",b=batch_size,t=timesteps)
        hidden=rearrange(backbone.time_agg_layer(hidden),"b x y c -> b c x y")
        for block in backbone.blocks: hidden=block(hidden)
        prediction=backbone.out_layer(hidden).permute(0,2,3,1).reshape(batch_size,height,width,9,2)
        return assert_output(prediction[:,:,:,1:,:].permute(0,3,4,1,2).contiguous(),batch)
