"""Derivative-free GPhyT-S THW backbone with a trajectory decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gphyt.model.gphyt_s import GPhyTSPTFT
from .landslide_common import physical,assert_output

class LandslideGPhyT(nn.Module):
    spatial_padding=(8,8)
    use_derivatives=False
    def __init__(self,official_root="",checkpoint_path="",report_path="",**_):
        super().__init__()
        self.core=GPhyTSPTFT(200,200,official_root=official_root,checkpoint_path=checkpoint_path,report_path=report_path,use_derivatives=False)
        old=self.core.backbone.detokenizer
        self.trajectory_decoder=type(old)(patch_size=(1,16,16),dim_embed=old.dim_embed,out_channels=2,mode="linear",overlap=0,img_size=(9,208,208))
        self.core.backbone.detokenizer=nn.Identity()
        self.loading_report=self.core.loading_report
        for name,_ in self.trajectory_decoder.named_parameters(): self.loading_report.newly_initialized.append("trajectory_decoder."+name)
        self.loading_report.newly_initialized=sorted(set(self.loading_report.newly_initialized)); self.loading_report.write(report_path)
    def forward(self,batch):
        fields=F.pad(physical(batch,("phi","z_norm","h0_norm")),(0,8,0,8))
        x=fields.permute(0,2,3,1).unsqueeze(1).repeat(1,9,1,1,1).contiguous()
        backbone=self.core.backbone
        if backbone.use_derivatives: raise AssertionError("GPhyT-S derivatives unexpectedly enabled")
        tokens=backbone.tokenizer(x)
        if backbone.init_pos_encodings is not None: tokens=backbone.init_pos_encodings(tokens)
        tokens=backbone.attention_blocks(tokens)
        prediction=self.trajectory_decoder(tokens)[:,1:,:200,:200,:]
        return assert_output(prediction.permute(0,1,4,2,3).contiguous(),batch)

class LandslideGPhyTEuler(LandslideGPhyT):
    """Decode state rates and integrate eight explicit-Euler steps with dt=1."""
    dt=1.0
    def forward(self,batch):
        fields=F.pad(physical(batch,("phi","z_norm","h0_norm")),(0,8,0,8))
        x=fields.permute(0,2,3,1).unsqueeze(1).repeat(1,9,1,1,1).contiguous()
        backbone=self.core.backbone
        if backbone.use_derivatives: raise AssertionError("GPhyT-S derivatives unexpectedly enabled")
        tokens=backbone.tokenizer(x)
        if backbone.init_pos_encodings is not None: tokens=backbone.init_pos_encodings(tokens)
        tokens=backbone.attention_blocks(tokens)
        rates=self.trajectory_decoder(tokens)[:,:8,:200,:200,:]
        initial=torch.stack((batch["h0"],batch["u0"]),dim=-1).unsqueeze(1)
        prediction=initial+torch.cumsum(self.dt*rates,dim=1)
        return assert_output(prediction.permute(0,1,4,2,3).contiguous(),batch)
