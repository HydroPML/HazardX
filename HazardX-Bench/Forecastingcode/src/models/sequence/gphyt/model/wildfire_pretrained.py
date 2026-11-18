"""Wildfire PT-FT adapters retaining the validated Flood backbones."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import torch
import torch.nn as nn
from gphyt.model.pde_transformer_sc_s import PDETransformerSCSPTFT
from gphyt.model.dpot_s import DPOTSPTFT
from gphyt.model.prithvi_eo2_300m import PrithviEO2300MPTFT, _normalize_prithvi_checkpoint
from gphyt.model.prithvi_eo2_registry_adapter import load_official_prithvi_module
from gphyt.model.wildfire_common import write_loading_report
from gphyt.model.wildfire_scratch import validate_wildfire_input,validate_logits

class TemporalPhysicalAdapter(nn.Module):
 def __init__(self):
  super().__init__();self.adapter=nn.Conv3d(40,3,1);self.temporal_logits=nn.Parameter(torch.zeros(5))
 def forward(self,x):
  z=self.adapter(x.permute(0,2,1,3,4));return (z*torch.softmax(self.temporal_logits,0)[None,None,:,None,None]).sum(2)

class FloodBackbonePTFT(nn.Module):
 def __init__(self,backbone):super().__init__();self.input_adapter=TemporalPhysicalAdapter();self.backbone=backbone;self.loading_report=getattr(backbone,'loading_report',None)
 def forward(self,x):validate_wildfire_input(x);return validate_logits(self.backbone(self.input_adapter(x)),x)

class NativePDETWildfire(nn.Module):
 new_parameter_prefixes=("input_adapter","head")
 def __init__(self,root,checkpoint,report):
  super().__init__();self.core=PDETransformerSCSPTFT(128,128,official_root=root,checkpoint_path=checkpoint,report_path=report);self.input_adapter=nn.Conv3d(40,3,1);self.head=nn.Conv2d(15,1,1);self.loading_report=self.core.loading_report
  # These official SC-S branches are multiplied by literal zero. Preserve the
  # published computation, but do not present disconnected weights as FT.
  backbone=self.core.backbone;impl=backbone.model
  inactive_scalings={"simulation_time":impl.SIMULATION_TIME_SCALING,"simulation_dt":impl.SIMULATION_DT_SCALING,"pde_parameter":impl.PDE_PARAMETER_SCALING}
  if any(float(value)!=0.0 for value in inactive_scalings.values()):raise RuntimeError(f"unexpected PDE-T conditioning scaling: {inactive_scalings}")
  inactive_prefixes=("simulation_time_embedder_","simulation_dt_embedder_","pde_parameter_embedder_")
  self.structurally_inactive_parameters=[]
  for name,param in impl.named_parameters():
   if name.startswith(inactive_prefixes):param.requires_grad_(False);self.structurally_inactive_parameters.append("core.backbone.model."+name)
  if not self.structurally_inactive_parameters:raise RuntimeError("PDE-T zero-scaled conditioning branches were not found")
 def forward(self,x):
  validate_wildfire_input(x);z=self.input_adapter(x.permute(0,2,1,3,4));b=z.shape[0];fields=[];times=[];types=[]
  for day in range(5):
   for channel in range(3):
    value=z[:,channel:channel+1,day];fields.append(torch.cat((value,torch.zeros_like(value)),1));times.append(torch.full((b,),day/4,device=x.device,dtype=x.dtype));types.append(torch.full((b,),channel,device=x.device,dtype=torch.long))
  labels=[torch.zeros(b,device=x.device,dtype=torch.long) for _ in fields];pars=[torch.zeros(b,1,device=x.device,dtype=x.dtype) for _ in fields];pclasses=[torch.zeros(b,1,device=x.device,dtype=torch.long) for _ in fields];tasks=[torch.zeros(b,device=x.device,dtype=torch.long) for _ in fields]
  out=self.core.backbone(x=fields,simulation_time=times,channel_type=types,pde_type=labels,pde_parameters=pars,pde_parameters_class=pclasses,simulation_dt=[torch.ones_like(t) for t in times],task=tasks,t=times,return_dict=True).sample
  return validate_logits(self.head(torch.cat([v[:,1:2] for v in out],1)),x)

class NativeDPOTWildfire(nn.Module):
 new_parameter_prefixes=("input_adapter","core.backbone.out_layer.4")
 def __init__(self,root,checkpoint,report):
  super().__init__();self.core=DPOTSPTFT(128,128,official_root=root,checkpoint_path=checkpoint,report_path=report);self.input_adapter=nn.Conv3d(40,3,1);b=self.core.backbone
  if not hasattr(b.time_agg_layer,"w") or b.time_agg_layer.w.shape[0]!=1:raise RuntimeError("unexpected DPOT time aggregator")
  b.time_agg_layer.w=nn.Parameter(b.time_agg_layer.w.detach().repeat(5,1,1)/5);b.in_timesteps=5;self.loading_report=self.core.loading_report
 def forward(self,x):
  validate_wildfire_input(x);z=self.input_adapter(x.permute(0,2,1,3,4)).permute(0,3,4,2,1).contiguous();out=self.core.backbone(z)[0]
  if out.ndim==5:out=out[...,0,0].unsqueeze(1)
  return validate_logits(out,x)

class NativePrithviWildfire(nn.Module):
 new_parameter_prefixes=("input_adapter","core.recovery")
 def __init__(self,checkpoint,report):
  super().__init__();self.core=PrithviEO2300MPTFT(128,128,checkpoint_path=checkpoint,report_path=report);self.input_adapter=nn.Conv3d(40,6,1);b=self.core.backbone;state=_normalize_prithvi_checkpoint(torch.load(checkpoint,map_location="cpu",weights_only=True));old=b.patch_embed.proj;b.patch_embed.proj=nn.Conv3d(6,1024,kernel_size=(1,16,16),stride=(1,16,16));b.patch_embed.proj.weight.data.copy_(state["patch_embed.proj.weight"]);b.patch_embed.proj.bias.data.copy_(state["patch_embed.proj.bias"]);b.num_frames=5;b.patch_embed.input_size=(5,128,128);b.patch_embed.grid_size=[5,8,8];b.patch_embed.num_patches=320
  official=load_official_prithvi_module(checkpoint);pos=official._interpolate_pos_encoding(pos_embed=state["pos_embed"],grid_size=(4,14,14),patch_size=(1,16,16),shape=(5,128,128),embed_dim=1024);b.pos_embed=pos;self.loading_report=self.core.loading_report
 def forward(self,x):
  validate_wildfire_input(x);z=self.input_adapter(x.permute(0,2,1,3,4));tokens=self.core.backbone.forward_features(z,temporal_coords=None,location_coords=None)[-1][:,1:];tokens=tokens.reshape(x.shape[0],5,64,1024).mean(1).transpose(1,2).reshape(x.shape[0],1024,8,8);return validate_logits(self.core.recovery(tokens),x)

class VideoMAEWildfire(nn.Module):
 new_parameter_prefixes=("head",)
 architecture_source = "OpenGVLab VideoMAEv2 ViT-B + published wildfire progressive head"
 def __init__(self,root,checkpoint,report):
  super().__init__();src=Path(root)/"models/modeling_finetune.py";spec=importlib.util.spec_from_file_location("_w_vm2",src);m=importlib.util.module_from_spec(spec);sys.modules["_w_vm2"]=m;spec.loader.exec_module(m)
  # Native pretrained tubelet=2 is retained. Five scientific frames are
  # followed by one all-zero compatibility frame, so no observed day is
  # discarded or repeated and no future information is introduced.
  self.backbone=m.vit_base_patch16_224(img_size=128,in_chans=40,num_classes=1,all_frames=6,tubelet_size=2,use_mean_pooling=False)
  self.backbone.head=nn.Identity()
  head_src=Path(root).parent/"wildfire-forecasting-main/models/segmentation_head.py"
  hs=importlib.util.spec_from_file_location("_published_wildfire_head",head_src);hm=importlib.util.module_from_spec(hs);sys.modules["_published_wildfire_head"]=hm;hs.loader.exec_module(hm)
  self.head=hm.ProgressiveUpsampling(in_dim=768,num_classes=1,hidden_dim=256,img_size=128,patch_size=16)
  raw=torch.load(checkpoint,map_location="cpu",weights_only=True);state=raw.get("module",raw.get("model",raw.get("state_dict",raw)));target=self.backbone.state_dict();exact=[];converted=[];missing=[];mismatch=[];load={}
  for k,v in target.items():
   if k in state and state[k].shape==v.shape:load[k]=state[k];exact.append(k)
   elif k=="patch_embed.proj.weight" and k in state:
    source=state[k]
    if tuple(source.shape[2:])!=tuple(v.shape[2:]):mismatch.append({"key":k,"checkpoint_shape":list(source.shape),"model_shape":list(v.shape)})
    else:load[k]=source.mean(1,keepdim=True).repeat(1,40,1,1,1)*(source.shape[1]/40);converted.append(k)
   elif k in ("norm.weight","norm.bias") and "fc_"+k in state and state["fc_"+k].shape==v.shape:
    load[k]=state["fc_"+k];converted.append(k)
   elif k!="pos_embed":missing.append(k)
  if missing or mismatch:raise RuntimeError(f"VideoMAEv2 retained backbone load incomplete missing={missing} mismatch={mismatch}")
  self.backbone.load_state_dict(load,strict=False)
  retained=[k for k in target if k!="pos_embed"];loaded_keys=set(exact)|set(converted);total=sum(target[k].numel() for k in retained);loaded=sum(target[k].numel() for k in retained if k in loaded_keys)
  unexpected=sorted(set(state)-set(load)-{"fc_norm.weight","fc_norm.bias"})
  self.loading_report=write_loading_report(self,checkpoint,report,{"model":"VideoMAEv2-PT-FT","exact_loaded":exact,"converted":converted,"newly_initialized":["head"],"missing":missing,"unexpected":unexpected,"shape_mismatch":mismatch,"backbone_parameter_count":total,"loaded_backbone_parameter_count":loaded,"backbone_loading_ratio":loaded/total,"temporal_adaptation":"five observed frames + one zero frame; native tubelet_size=2"})
 def forward(self,x):
  validate_wildfire_input(x);z=x.permute(0,2,1,3,4);z=torch.nn.functional.pad(z,(0,0,0,0,0,1));tokens=self.backbone.patch_embed(z);tokens=tokens+self.backbone.pos_embed.to(tokens.device,dtype=tokens.dtype);tokens=self.backbone.pos_drop(tokens)
  for block in self.backbone.blocks:tokens=block(tokens)
  tokens=self.backbone.norm(tokens);tokens=tokens.reshape(x.shape[0],3,64,768).mean(1);return validate_logits(self.head(tokens),x)

def build_wildfire_pretrained(a):
 report=getattr(a,"_pretrained_report_path",str(Path(a.output_dir)/a.model_type/'pretrained_loading_report.json'))
 if a.model_type=='PDE-T-SC-S-PT-FT':return NativePDETWildfire(a.pdet_official_root,a.pdet_checkpoint,report)
 if a.model_type=='DPOT-S-PT-FT':return NativeDPOTWildfire(a.dpot_official_root,a.dpot_checkpoint,report)
 if a.model_type=='Prithvi-EO2-300M-PT-FT':return NativePrithviWildfire(a.prithvi_checkpoint,report)
 if a.model_type=='VideoMAEv2-PT-FT':return VideoMAEWildfire(a.videomae_official_root,a.videomae_checkpoint,report)
 raise KeyError(a.model_type)
