"""Prithvi-EO2-300M on the native nine-frame 3-D token grid."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gphyt.model.prithvi_eo2_300m import PrithviEO2300MPTFT,_normalize_prithvi_checkpoint
from gphyt.model.prithvi_eo2_registry_adapter import load_official_prithvi_module
from .landslide_common import physical,assert_output

class LandslidePrithvi(nn.Module):
    spatial_padding=(8,8)
    def __init__(self,checkpoint_path="",report_path="",**_):
        super().__init__()
        self.core=PrithviEO2300MPTFT(200,200,checkpoint_path=checkpoint_path,report_path=report_path)
        # The generic single-frame wrapper owns a 2-D recovery head, while
        # this nine-frame adapter decodes exclusively with the 3-D head below.
        # Remove the disconnected head from parameters and optimizer state.
        self.core.recovery=nn.Identity()
        checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        source=_normalize_prithvi_checkpoint(checkpoint)["pos_embed"]
        official=load_official_prithvi_module(checkpoint_path)
        position=official._interpolate_pos_encoding(pos_embed=source,grid_size=(4,14,14),patch_size=(1,16,16),shape=(9,208,208),embed_dim=1024)
        self.core.backbone.pos_embed=position
        from types import MethodType
        self.core.backbone.interpolate_pos_encoding=MethodType(lambda module, sample_shape: module.pos_embed,self.core.backbone)
        self.recovery=nn.ConvTranspose3d(1024,2,kernel_size=(1,16,16),stride=(1,16,16))
        self.loading_report=self.core.loading_report
        stale_recovery={"recovery.0.weight","recovery.0.bias"}
        self.loading_report.newly_initialized=[name for name in self.loading_report.newly_initialized if name not in stale_recovery]
        for name,_ in self.recovery.named_parameters(): self.loading_report.newly_initialized.append("recovery."+name)
        self.loading_report.newly_initialized=sorted(set(self.loading_report.newly_initialized)); self.loading_report.write(report_path)
    def enable_activation_checkpointing(self):
        if hasattr(self.core.backbone,"set_grad_checkpointing"): self.core.backbone.set_grad_checkpointing(True)
    def disable_activation_checkpointing(self):
        if hasattr(self.core.backbone,"set_grad_checkpointing"): self.core.backbone.set_grad_checkpointing(False)
    def forward(self,batch):
        fields=F.pad(physical(batch,("z_norm","phi","h0_norm")),(0,8,0,8))
        official_input=fields.unsqueeze(2).repeat(1,1,9,1,1)
        tokens=self.core.backbone.forward_features(official_input,temporal_coords=None,location_coords=None)[-1][:,1:]
        expected=9*13*13
        if tokens.shape[1]!=expected: raise AssertionError(f"Prithvi tokens {tokens.shape[1]} != {expected}")
        feature=tokens.transpose(1,2).reshape(fields.shape[0],1024,9,13,13)
        prediction=self.recovery(feature)[:,:,1:,:200,:200]
        return assert_output(prediction.permute(0,2,1,3,4).contiguous(),batch)
