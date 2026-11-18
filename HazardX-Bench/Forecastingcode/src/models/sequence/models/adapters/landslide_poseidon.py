"""Poseidon-T with native continuous lead-time conditioning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from gphyt.model.poseidon_t import PoseidonTPTFT
from .landslide_common import physical,assert_output

def _conditional_norm_fp32(module,x,time):
    dtype=x.dtype
    with torch.autocast(device_type=x.device.type,enabled=False):
        value=x.float(); mean=value.mean(dim=-1,keepdim=True); var=(value.square().mean(dim=-1,keepdim=True)-mean.square()).clamp_min(0.)
        value=(value-mean)/(var+module.eps).sqrt(); condition=time.reshape(-1,1).float(); weight=module.weight(condition).unsqueeze(1); bias=module.bias(condition).unsqueeze(1)
        if value.dim()==4: weight=weight.unsqueeze(1); bias=bias.unsqueeze(1)
        return (weight*value+bias).to(dtype=dtype)

def _attention_fp32(module,hidden_states,attention_mask=None,head_mask=None,output_attentions=False):
    dtype=hidden_states.dtype
    with torch.autocast(device_type=hidden_states.device.type,enabled=False):
        mask=None if attention_mask is None else attention_mask.float(); head=None if head_mask is None else head_mask.float()
        outputs=module._landslide_original_forward(hidden_states.float(),mask,head,output_attentions)
    return tuple(value.to(dtype=dtype) if torch.is_tensor(value) else value for value in outputs)

class LandslidePoseidon(nn.Module):
    lead_times=(.125,.25,.375,.5,.625,.75,.875,1.)
    spatial_padding=(24,24)
    def __init__(self,official_root="",checkpoint_path="",report_path="",lead_chunk=8,**_):
        super().__init__()
        self.lead_chunk=max(1,min(len(self.lead_times),int(lead_chunk)))
        self.core=PoseidonTPTFT(200,200,official_root=official_root,checkpoint_path=checkpoint_path,report_path=report_path)
        patched=0
        for module in self.core.backbone.modules():
            if module.__class__.__name__=="ConditionalLayerNorm": module.forward=MethodType(_conditional_norm_fp32,module); patched+=1
        if patched==0: raise RuntimeError("Poseidon contains no ConditionalLayerNorm modules")
        self.fp32_conditional_norm_count=patched
        attention_patched=0
        for module in self.core.backbone.modules():
            if module.__class__.__name__=="Swinv2SelfAttention":
                module._landslide_original_forward=module.forward; module.forward=MethodType(_attention_fp32,module); attention_patched+=1
        if attention_patched==0: raise RuntimeError("Poseidon contains no Swinv2SelfAttention modules")
        self.fp32_attention_count=attention_patched
        recovery=self.core.backbone.patch_recovery
        old=recovery.projection
        recovery.projection=nn.ConvTranspose2d(old.in_channels,2,old.kernel_size,old.stride)
        recovery.mixup=nn.Conv2d(2,2,5,padding=2,bias=False)
        self.loading_report=self.core.loading_report
        for name in ("patch_recovery.projection.weight","patch_recovery.projection.bias","patch_recovery.mixup.weight"):
            if name not in self.loading_report.newly_initialized: self.loading_report.newly_initialized.append(name)
        self.loading_report.newly_initialized.sort(); self.loading_report.write(report_path)
    def parameter_groups(self,backbone_lr=5e-5,adapter_lr=5e-4):
        return self.core.parameter_groups(backbone_lr,adapter_lr)
    def forward(self,batch):
        x=physical(batch,("z_norm","phi","h0_norm")); padded=F.pad(x,(0,24,0,24)); predictions=[]
        batch_size=x.shape[0]; leads=padded.new_tensor(self.lead_times)
        for start in range(0,len(self.lead_times),self.lead_chunk):
            chunk=leads[start:start+self.lead_chunk]; chunk_size=chunk.numel()
            expanded=padded[:,None].expand(batch_size,chunk_size,-1,-1,-1).reshape(batch_size*chunk_size,padded.shape[1],padded.shape[2],padded.shape[3])
            time=chunk[None].expand(batch_size,chunk_size).reshape(-1)
            prediction=self.core._forward_rectangular(expanded,time)[:,:,:200,:200].reshape(batch_size,chunk_size,2,200,200)
            predictions.append(prediction)
        return assert_output(torch.cat(predictions,1),batch)


