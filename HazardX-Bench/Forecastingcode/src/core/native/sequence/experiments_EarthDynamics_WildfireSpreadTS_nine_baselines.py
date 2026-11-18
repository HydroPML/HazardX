#!/usr/bin/env python3
"""Unified WildfireSpreadTS 5-day -> next-day benchmark.

No full run is started implicitly. Use --mode audit, smoke, overfit, or train.
Test evaluation is reachable only after loading best_validation.pt.
"""
from __future__ import annotations

import argparse, json, math, os, pickle, random, time
from datetime import date
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, default_collate
from torch.utils.data.distributed import DistributedSampler

from data.wildfire_spread_ts_dataset import (WildfireSpreadTS5DayDataset,
    assert_event_disjoint, fold_years)
from gphyt.model.wildfire_common import (OfficialBinaryDiceLoss, assert_optimizer_complete,
    binary_metrics, warmup_cosine_lambda)
from gphyt.model.wildfire_scratch import (WildfireConvLSTM, WildfireEarthFormer,
    WildfireFNO2D, WildfireUNet2D, WildfireUTAE)

MODELS=("UNet2D-Scratch","FNO2D-Scratch","ConvLSTM-Scratch","EarthFormer-Scratch",
 "UTAE-Scratch","PDE-T-SC-S-PT-FT","DPOT-S-PT-FT","VideoMAEv2-PT-FT",
 "Prithvi-EO2-300M-PT-FT")
SCRATCH=set(MODELS[:5]); PT=set(MODELS[5:])

def parser():
 p=argparse.ArgumentParser()
 p.add_argument("--model_type",required=True,choices=MODELS);p.add_argument("--fold_id",type=int,default=0,choices=range(12))
 p.add_argument("--data_dir",default="/home/yuezeng/EarthDynamics/Wildfire/Data")
 p.add_argument("--data_backend",choices=("tiff","hdf5"),default="tiff")
 p.add_argument("--hdf5_dir",default="/home/yuezeng/EarthDynamics/Wildfire/HDF5")
 p.add_argument("--wildfire_official_root",default="official_sources/WildfireSpreadTS/WildfireSpreadTS-main")
 p.add_argument("--earthformer_official_root",default="official_sources/LandslideCast/earthformer/earth-forecasting-transformer-7732b03bdb366110563516c3502315deab4c2026")
 p.add_argument("--pdet_official_root",default="official_sources/LandslideCast/pdet/pde-transformer-850e09d352487922e4a853ee90a2624717769cc7")
 p.add_argument("--dpot_official_root",default="official_sources/LandslideCast/dpot/DPOT-dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17")
 p.add_argument("--pdet_checkpoint",default="checkpoints/LandslideCast/PDE-T/sc-s/diffusion_pytorch_model.safetensors")
 p.add_argument("--dpot_checkpoint",default="checkpoints/LandslideCast/DPOT-S/model_S.pth")
 p.add_argument("--videomae_checkpoint",default="checkpoints/WildfireSpreadTS/VideoMAEv2/vit_b_k710_dl_from_giant.pth");p.add_argument("--videomae_official_root",default="official_sources/WildfireSpreadTS/VideoMAEv2-master")
 p.add_argument("--prithvi_checkpoint",default="checkpoints/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt")
 p.add_argument("--output_dir",default="results/WildfireSpreadTS_5day");p.add_argument("--mode",choices=("audit","smoke","memory","overfit","train"),default="audit")
 p.add_argument("--epochs",type=int,default=100);p.add_argument("--validation_every",type=int,default=5)
 p.add_argument("--batch_size",type=int,default=8);p.add_argument("--accum_steps",type=int,default=1);p.add_argument("--num_workers",type=int,default=2)
 p.add_argument("--overfit_steps",type=int,default=75);p.add_argument("--seed",type=int,default=0);p.add_argument("--resume",default="")
 p.add_argument("--max_train_batches",type=int,default=0,help=argparse.SUPPRESS);p.add_argument("--max_eval_batches",type=int,default=0,help=argparse.SUPPRESS)
 return p

def datasets(a):
 tr,va,te=fold_years(a.fold_id); kw=dict(data_root=a.data_dir,stats_years=tr,official_root=a.wildfire_official_root,seed=a.seed,backend=a.data_backend,hdf5_root=a.hdf5_dir)
 ds=[WildfireSpreadTS5DayDataset(years=y,split=s,**kw) for y,s in ((tr,"train"),(va,"val"),(te,"test"))]
 assert_event_disjoint(*ds);return tr,va,te,*ds

def build_model(a):
 if a.model_type=="UNet2D-Scratch":return WildfireUNet2D()
 if a.model_type=="FNO2D-Scratch":return WildfireFNO2D()
 if a.model_type=="ConvLSTM-Scratch":return WildfireConvLSTM()
 if a.model_type=="EarthFormer-Scratch":return WildfireEarthFormer(a.earthformer_official_root)
 if a.model_type=="UTAE-Scratch":return WildfireUTAE(a.wildfire_official_root)
 try: from gphyt.model.wildfire_pretrained import build_wildfire_pretrained
 except ImportError as e: raise RuntimeError("PT-FT adapters are unavailable; refusing to silently run a random/simplified backbone") from e
 return build_wildfire_pretrained(a)

OPTIMIZATION = {
 "UNet2D-Scratch":{"optimizer":"Adam","lr":1e-4,"weight_decay":1e-4,"betas":(0.9,0.999),"scheduler":"cosine","grad_clip":1.0},
 "FNO2D-Scratch":{"optimizer":"Adam","lr":1e-4,"weight_decay":1e-4,"betas":(0.9,0.999),"scheduler":"cosine","grad_clip":1.0},
 "ConvLSTM-Scratch":{"optimizer":"Adam","lr":1e-4,"weight_decay":0.0,"betas":(0.9,0.999),"scheduler":"onecycle","grad_clip":0.0},
 "EarthFormer-Scratch":{"optimizer":"AdamW","lr":3e-4,"weight_decay":0.0,"betas":(0.9,0.999),"scheduler":"warmup_cosine_20","grad_clip":1.0},
 "UTAE-Scratch":{"optimizer":"Adam","lr":1e-4,"weight_decay":1e-4,"betas":(0.9,0.999),"scheduler":"cosine","grad_clip":1.0},
 "PDE-T-SC-S-PT-FT":{"optimizer":"AdamW","lr":4e-5,"weight_decay":0.0,"betas":(0.9,0.999),"scheduler":"constant","grad_clip":0.0},
 "DPOT-S-PT-FT":{"optimizer":"Adam","lr":3e-5,"weight_decay":0.0,"betas":(0.9,0.9),"scheduler":"dpot_onecycle","grad_clip":1.0},
 "VideoMAEv2-PT-FT":{"optimizer":"AdamW","lr":1e-4,"head_lr":1e-3,"weight_decay":1e-2,"betas":(0.9,0.999),"scheduler":"warmup_cosine_5","grad_clip":1.0},
 "Prithvi-EO2-300M-PT-FT":{"optimizer":"AdamW","lr":5e-5,"weight_decay":0.05,"betas":(0.9,0.999),"scheduler":"cosine","grad_clip":1.0},
}

def optimizer(model,a):
 config=OPTIMIZATION[a.model_type];base=model.module if isinstance(model,DDP) else model
 trainable=[(n.removeprefix("module."),p) for n,p in model.named_parameters() if p.requires_grad]
 if a.model_type=="VideoMAEv2-PT-FT":
  prefixes=tuple(getattr(base,"new_parameter_prefixes",("head",)));backbone=[p for n,p in trainable if not n.startswith(prefixes)];head=[p for n,p in trainable if n.startswith(prefixes)]
  if not backbone or not head:raise RuntimeError(f"VideoMAEv2 parameter groups invalid backbone={len(backbone)} head={len(head)}")
  groups=[{"params":backbone,"lr":config["lr"],"name":"pretrained_backbone"},{"params":head,"lr":config["head_lr"],"name":"new_head"}]
 else:groups=[{"params":[p for _,p in trainable],"lr":config["lr"],"name":"all_trainable"}]
 cls=torch.optim.AdamW if config["optimizer"]=="AdamW" else torch.optim.Adam
 kwargs=dict(lr=config["lr"],weight_decay=config["weight_decay"],betas=config["betas"])
 if cls is torch.optim.Adam:kwargs["foreach"]=False
 opt=cls(groups,**kwargs);assert_optimizer_complete(model,opt);return opt,config["weight_decay"]

def _warmup_cosine(opt,total,warmup):
 total=max(1,int(total));warmup=max(1,min(int(warmup),total));linear=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1e-3,end_factor=1.0,total_iters=warmup)
 if warmup>=total:return linear
 cosine=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total-warmup,eta_min=0.0)
 return torch.optim.lr_scheduler.SequentialLR(opt,[linear,cosine],milestones=[warmup])

def make_scheduler(opt,a,total_updates,diagnostic=False):
 config=OPTIMIZATION[a.model_type];kind=config["scheduler"];total=max(1,int(total_updates))
 if kind=="onecycle":return torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=config["lr"],total_steps=total)
 if kind=="dpot_onecycle":return torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=config["lr"],total_steps=total,pct_start=0.2,div_factor=1e4,final_div_factor=1e4)
 if kind=="warmup_cosine_20":return _warmup_cosine(opt,total,int(0.2*total))
 if kind=="warmup_cosine_5":
  warm=min(5,total) if diagnostic else max(1,round(total*5/max(1,a.epochs)))
  return _warmup_cosine(opt,total,warm)
 if kind=="cosine":return torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total,eta_min=0.0)
 if kind=="constant":return torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
 raise ValueError(kind)

def maybe_clip_gradients(model,a):
 limit=float(OPTIMIZATION[a.model_type]["grad_clip"])
 if limit>0:return float(torch.nn.utils.clip_grad_norm_(model.parameters(),limit,error_if_nonfinite=True))
 for parameter in model.parameters():
  if parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise FloatingPointError("non-finite gradient")
 return None

def dataloader_io_kwargs(num_workers):
 workers=int(num_workers)
 if workers<0:raise ValueError(f"num_workers must be non-negative, got {workers}")
 kwargs={"num_workers":workers,"pin_memory":True}
 if workers>0:kwargs.update(persistent_workers=True,prefetch_factor=2)
 return kwargs

def audit(a,tr,va,te,train,val,test):
 events=lambda d:len({(w.year,w.event_id) for w in d.windows})
 sample=train[0]
 payload={"fold_id":a.fold_id,"data_backend":a.data_backend,
  "data_root":a.hdf5_dir if a.data_backend=="hdf5" else a.data_dir,
  "train_years":tr,"val_years":va,"test_years":te,
  "events":{"train":events(train),"val":events(val),"test":events(test)},
  "samples":{"train":len(train),"val":len(val),"test":len(test)},
  "raw_channels":len(train.raw_channel_names),"processed_channels":len(train.processed_channel_names),
  "raw_channel_names":train.raw_channel_names,"processed_channel_names":train.processed_channel_names,
  "example":{k:(list(v.shape) if isinstance(v,torch.Tensor) else v) for k,v in sample.items()},
  "input_min":float(sample["input"].min()),"input_max":float(sample["input"].max()),
  "input_mean":float(sample["input"].mean()),"input_std":float(sample["input"].std()),
  "target_positive_ratio":float(sample["target"].mean()),"event_leakage":False,
  "dataloader":{"num_workers":a.num_workers,"persistent_workers":a.num_workers>0,
                "pin_memory":True,"prefetch_factor":2 if a.num_workers>0 else None}}
 print(json.dumps(payload,indent=2,default=list));return payload

def model_forward(model,batch,device):
 x=batch["input"].to(device,non_blocking=True)
 base=model.module if isinstance(model,DDP) else model
 if isinstance(base,WildfireUTAE):return model(x,batch["day_of_year"].to(device,non_blocking=True))
 return model(x)

def gradient_depth_audit(model):
 base=model.module if isinstance(model,DDP) else model
 named=[(n,p) for n,p in base.named_parameters() if p.requires_grad]
 result={"trainable_parameter_tensors":len(named)}
 unused=[n for n,p in named if p.grad is None];zero=[n for n,p in named if p.grad is not None and not bool(torch.count_nonzero(p.grad))]
 result.update(gradient_parameter_tensors=len(named)-len(unused),unused_parameter_tensors=unused,zero_gradient_parameter_tensors=zero)
 if unused:raise RuntimeError(f"trainable parameters disconnected from loss: {unused}")
 for label,part in zip(("shallow","middle","deep"),torch.tensor_split(torch.arange(len(named)),3)):
  candidates=[]
  for i in part.tolist():
   n,p=named[i];g=p.grad
   if g is not None and bool(torch.isfinite(g).all()) and bool(torch.count_nonzero(g)):candidates.append(n)
  if not candidates:raise RuntimeError(f"{label} parameter third has no finite non-zero gradient")
  result[label]=candidates[0]
 return result

def one_step(model,batch,opt,loss_fn,device,a):
 x=batch["input"].to(device,non_blocking=True);y=batch["target"].to(device,non_blocking=True);torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None
 t=time.perf_counter();out=model_forward(model,batch,device);torch.cuda.synchronize() if device.type=="cuda" else None;tf=time.perf_counter()-t
 loss=loss_fn(out,y);t=time.perf_counter();loss.backward();torch.cuda.synchronize() if device.type=="cuda" else None;tb=time.perf_counter()-t
 grad_audit=gradient_depth_audit(model);grad_norm=maybe_clip_gradients(model,a);t=time.perf_counter();opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize() if device.type=="cuda" else None;to=time.perf_counter()-t
 if tuple(out.shape)!=(x.shape[0],1,128,128) or not torch.isfinite(loss):raise RuntimeError("smoke output/loss invalid")
 return {"loss":float(loss),"forward_s":tf,"backward_s":tb,"optimizer_s":to,
  "peak_allocated":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
  "peak_reserved":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,"gradient_depth_audit":grad_audit,"grad_clip":OPTIMIZATION[a.model_type]["grad_clip"],"pre_clip_grad_norm":grad_norm}

def positive_batch(dataset,batch_size,num_workers):
 samples=[]
 for index in range(len(dataset)):
  sample=dataset[index]
  if float(sample["target"].sum())>0:samples.append(sample)
  if len(samples)==batch_size:break
 if len(samples)<batch_size:raise RuntimeError("could not find enough fire-positive training crops")
 return default_collate(samples)

def overfit_check(model,dataset,a,loss_fn,device,steps):
 candidates=[]
 for i in range(len(dataset)):
  sample=dataset[i];positive_pixels=float(sample["target"].sum())
  if positive_pixels>0:candidates.append((positive_pixels,sample))
  if len(candidates)==32:break
 if len(candidates)<4:raise RuntimeError("could not find four fire-positive crops for overfit audit")
 candidates.sort(key=lambda item:item[0],reverse=True);samples=[sample for _,sample in candidates[:4]]
 batches=[default_collate(samples)]
 # This diagnostic must not reuse the formal five-epoch warmup: at 50--100
 # steps that schedule leaves the learning rate effectively zero.
 opt,_=optimizer(model,a)
 sched=make_scheduler(opt,a,steps,diagnostic=True)
 def measure():
  model.eval();ls=[];outs=[];ys=[]
  with torch.no_grad():
   for batch in batches:
    o=model_forward(model,batch,device);y=batch["target"].to(device,non_blocking=True);ls.append(float(loss_fn(o,y)));outs.append(o.float().cpu());ys.append(y.float().cpu())
  logits=torch.cat(outs);truth=torch.cat(ys);return sum(ls)/len(ls),logits,truth
 initial_loss,initial_logits,truth=measure()
 trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 probe_indices=sorted(set((0,len(trainable)//2,len(trainable)-1)))
 before={trainable[i][0]:trainable[i][1].detach().cpu().clone() for i in probe_indices};model.train()
 for step in range(steps):
  batch=batches[step%len(batches)];x=batch["input"].to(device,non_blocking=True);y=batch["target"].to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):out=model_forward(model,batch,device);loss=loss_fn(out,y)
  loss.backward();maybe_clip_gradients(model,a);opt.step();sched.step()
 final_loss,logits,truth=measure();prob=logits.sigmoid();initial_ap=binary_metrics(initial_logits,truth,.5)["AP"]
 diagnostic_threshold=max((i/100 for i in range(1,100)),key=lambda q:binary_metrics(logits,truth,q)["F1"]);s=binary_metrics(logits,truth,diagnostic_threshold);binary=prob>=diagnostic_threshold
 updated=any(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters() if n in before)
 result={"steps":steps,"initial_loss":initial_loss,"final_loss":final_loss,"loss_decreased":final_loss<initial_loss,"initial_AP":initial_ap,"final_AP":s["AP"],"AP_improved":s["AP"]>initial_ap,"parameters_updated":updated,"prediction_min":float(prob.min()),"prediction_max":float(prob.max()),"prediction_std":float(prob.std()),"diagnostic_threshold":diagnostic_threshold,"predicted_positive_fraction":float(binary.float().mean()),"not_all_zero":bool(binary.any()),"not_all_one":bool((~binary).any())}
 if not all((result["loss_decreased"],result["AP_improved"],result["parameters_updated"],result["prediction_std"]>0,result["not_all_zero"],result["not_all_one"])):raise RuntimeError(f"overfit audit failed: {result}")
 return result

def evaluate(model,loader,loss_fn,device,threshold=.5,max_batches=0):
 model.eval();logits=[];targets=[];loss_sum=0.;count=0
 with torch.no_grad():
  for batch_index,batch in enumerate(loader):
   if max_batches and batch_index>=max_batches:break
   x=batch["input"].to(device,non_blocking=True);y=batch["target"].to(device,non_blocking=True);o=model_forward(model,batch,device);loss_sum+=float(loss_fn(o,y))*x.shape[0];count+=x.shape[0];logits.append(o.float().cpu());targets.append(y.float().cpu())
 l=torch.cat(logits);t=torch.cat(targets);return {"loss":loss_sum/count,**binary_metrics(l,t,threshold)},l,t

def rng_checkpoint_state():
 def pack(value):return torch.tensor(list(pickle.dumps(value,protocol=4)),dtype=torch.uint8)
 return {"python":pack(random.getstate()),"numpy":pack(np.random.get_state()),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng_checkpoint_state(state):
 if not state:return
 def unpack(value):return pickle.loads(bytes(value.tolist()))
 random.setstate(unpack(state["python"].cpu()));np.random.set_state(unpack(state["numpy"].cpu()));torch.set_rng_state(state["torch"].cpu())
 if torch.cuda.is_available() and state.get("cuda"):torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])

def train_lifecycle(a,model,opt,sched,loss_fn,device,train,val,test,outdir):
 distributed=dist.is_available() and dist.is_initialized();rank=dist.get_rank() if distributed else 0
 sampler=DistributedSampler(train,shuffle=True,seed=a.seed) if distributed else None
 io_kwargs=dataloader_io_kwargs(a.num_workers)
 train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,**io_kwargs)
 val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,**io_kwargs)
 test_loader=DataLoader(test,batch_size=1,shuffle=False,**io_kwargs)
 best=-1.;best_path=outdir/"best_validation.pt";latest_path=outdir/"latest.pt";start_epoch=0;global_step=0
 if a.resume:
  ck=torch.load(a.resume,map_location=device,weights_only=True)
  saved_args=ck.get("args",{})
  for key in ("model_type","fold_id"):
   if key in saved_args and saved_args[key]!=getattr(a,key):raise RuntimeError(f"resume {key} mismatch: checkpoint={saved_args[key]} current={getattr(a,key)}")
  (model.module if isinstance(model,DDP) else model).load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);sched.load_state_dict(ck["scheduler"]);start_epoch=int(ck["epoch"]);best=float(ck.get("best_AP",-1));global_step=int(ck.get("global_step",0));restore_rng_checkpoint_state(ck.get("rng_state"))
 opt.zero_grad(set_to_none=True)
 for epoch in range(start_epoch,a.epochs):
  if sampler is not None:sampler.set_epoch(epoch)
  model.train()
  effective_batches=min(len(train_loader),a.max_train_batches or len(train_loader))
  for n,batch in enumerate(train_loader):
   if n>=effective_batches:break
   y=batch["target"].to(device,non_blocking=True)
   sync_now=(n+1)%a.accum_steps==0 or n+1==effective_batches
   sync_context=model.no_sync() if isinstance(model,DDP) and not sync_now else __import__('contextlib').nullcontext()
   group_start=(n//a.accum_steps)*a.accum_steps;group_size=min(a.accum_steps,effective_batches-group_start)
   with sync_context:
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):loss=loss_fn(model_forward(model,batch,device),y)/group_size
    loss.backward()
   if sync_now:
    maybe_clip_gradients(model,a);opt.step();opt.zero_grad(set_to_none=True);sched.step();global_step+=1
  if (epoch+1)%a.validation_every==0:
   # Each rank evaluates the identical deterministic validation set; rank zero owns selection.
   metrics,_,_=evaluate(model,val_loader,loss_fn,device,max_batches=a.max_eval_batches)
   if rank==0:
    print("VALIDATION",epoch+1,json.dumps(metrics),flush=True)
    base=model.module if isinstance(model,DDP) else model
    checkpoint={"model":base.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch+1,"global_step":global_step,"best_AP":max(best,metrics["AP"]),"args":vars(a),"rng_state":rng_checkpoint_state()}
    latest_tmp=latest_path.with_suffix(".pt.tmp")
    torch.save(checkpoint,latest_tmp)
    latest_tmp.replace(latest_path)
    if metrics["AP"]>best:
     best=metrics["AP"]
     best_tmp=best_path.with_suffix(".pt.tmp")
     torch.save(checkpoint,best_tmp)
     best_tmp.replace(best_path)
   if distributed:dist.barrier()
 selected_path=best_path if best_path.is_file() else (Path(a.resume) if a.resume else best_path)
 if not selected_path.is_file():raise RuntimeError("best_validation.pt was not produced; validation schedule is invalid")
 ck=torch.load(selected_path,map_location=device,weights_only=True);(model.module if isinstance(model,DDP) else model).load_state_dict(ck["model"])
 _,vl,vt=evaluate(model,val_loader,loss_fn,device,max_batches=a.max_eval_batches);threshold=max((i/100 for i in range(1,100)),key=lambda q:binary_metrics(vl,vt,q)["F1"]);metrics,_,_=evaluate(model,test_loader,loss_fn,device,threshold,max_batches=a.max_eval_batches)
 if rank==0:(outdir/"test_metrics.json").write_text(json.dumps({"validation_threshold":threshold,**metrics},indent=2))
 return metrics

def destroy_distributed():
 if dist.is_available() and dist.is_initialized():dist.barrier();dist.destroy_process_group()

def main():
 a=parser().parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 world=int(os.environ.get("WORLD_SIZE","1"));local_rank=int(os.environ.get("LOCAL_RANK","0"))
 if world>1:dist.init_process_group("nccl")
 rank=dist.get_rank() if dist.is_initialized() else 0
 tr,va,te,train,val,test=datasets(a);info=audit(a,tr,va,te,train,val,test)
 if a.mode=="audit":destroy_distributed();return
 device=torch.device("cuda",local_rank) if torch.cuda.is_available() else torch.device("cpu")
 if device.type=="cuda":torch.cuda.set_device(device)
 if rank>0:a._pretrained_report_path=f"/tmp/wildfire_pt_rank{rank}_{os.getpid()}_{a.model_type}.json"
 model=build_model(a).to(device)
 pristine={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if a.mode in ("overfit","train") and not a.resume else None
 if world>1:model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
 base=model.module if isinstance(model,DDP) else model
 architecture_parameters=getattr(base,"architecture",getattr(getattr(base,"core",None),"official_config",getattr(getattr(getattr(base,"core",None),"backbone",None),"official_config",None)))
 print("ARCHITECTURE",json.dumps({"model":a.model_type,"optimization":OPTIMIZATION[a.model_type],"adapter_class":base.__class__.__name__,"architecture_source":getattr(base,"architecture_source","validated Flood PT-FT wrapper + official backbone"),"architecture_parameters":architecture_parameters,"canonical_input":["B",5,40,128,128],"output":["B",1,128,128],"structurally_inactive_parameters":getattr(base,"structurally_inactive_parameters",getattr(base,"structurally_inactive_parameter_names",[]))},indent=2,default=str),flush=True)
 opt,wd=optimizer(model,a);samples_per_rank=math.ceil(len(train)/world);full_epoch_batches=math.ceil(samples_per_rank/a.batch_size);epoch_batches=min(full_epoch_batches,a.max_train_batches or full_epoch_batches);updates=math.ceil(epoch_batches/a.accum_steps)*a.epochs
 sched=make_scheduler(opt,a,updates);loss_fn=OfficialBinaryDiceLoss()
 out=Path(a.output_dir)/a.model_type
 if rank==0:out.mkdir(parents=True,exist_ok=True)
 if dist.is_initialized():dist.barrier()
 if a.resume:
  metrics=train_lifecycle(a,model,opt,sched,loss_fn,device,train,val,test,out)
  if rank==0:print("TEST",json.dumps(metrics,indent=2))
  destroy_distributed();return
 if a.mode=="memory":
  sweep=[]
  for size in (1,2,4,8):
   try:
    batch=positive_batch(train,size,a.num_workers);entry=one_step(model,batch,opt,loss_fn,device,a);entry.update(batch_size=size,status="pass");sweep.append(entry)
   except torch.cuda.OutOfMemoryError as error:
    opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();sweep.append({"batch_size":size,"status":"oom","error":str(error)});break
  if rank==0:(out/"memory_sweep_report.json").write_text(json.dumps(sweep,indent=2));print("MEMORY_SWEEP",json.dumps(sweep,indent=2))
  destroy_distributed();return
 batch=positive_batch(train,a.batch_size,a.num_workers)
 result=one_step(model,batch,opt,loss_fn,device,a);sched.step();result.update(parameters_total=sum(p.numel() for p in model.parameters()),parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),optimizer=OPTIMIZATION[a.model_type]["optimizer"],configured_lr=OPTIMIZATION[a.model_type]["lr"],configured_head_lr=OPTIMIZATION[a.model_type].get("head_lr"),current_learning_rates=[g["lr"] for g in opt.param_groups],betas=OPTIMIZATION[a.model_type]["betas"],weight_decay=wd,scheduler=OPTIMIZATION[a.model_type]["scheduler"],total_steps=updates,warmup_steps=(int(.2*updates) if OPTIMIZATION[a.model_type]["scheduler"]=="warmup_cosine_20" else (round(updates*5/max(1,a.epochs)) if OPTIMIZATION[a.model_type]["scheduler"]=="warmup_cosine_5" else 0)))
 state={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"args":vars(a),"rng_state":rng_checkpoint_state()}
 if rank==0:torch.save(state,out/"smoke.pt")
 if dist.is_initialized():dist.barrier()
 model.eval();x=batch["input"].to(device,non_blocking=True)
 with torch.no_grad():before_reload=model_forward(model,batch,device).detach().clone()
 model.load_state_dict(torch.load(out/"smoke.pt",map_location=device,weights_only=True)["model"]);model.eval()
 with torch.no_grad():same=torch.equal(before_reload,model_forward(model,batch,device))
 result["reload_bitwise_equal"]=same
 if a.model_type in PT and rank==0:
  report_path=out/"pretrained_loading_report.json";report=json.loads(report_path.read_text());exact=report.get("exact_loaded",[])
  report.update(model=a.model_type,checkpoint_path=str(Path(getattr(a,{"PDE-T-SC-S-PT-FT":"pdet_checkpoint","DPOT-S-PT-FT":"dpot_checkpoint","VideoMAEv2-PT-FT":"videomae_checkpoint","Prithvi-EO2-300M-PT-FT":"prithvi_checkpoint"}[a.model_type])).resolve()),parameters_total=result["parameters_total"],parameters_trainable=result["parameters_trainable"],loaded_backbone_parameter_count=report.get("loaded_backbone_parameter_count",report.get("loaded_parameter_count")),representative_loaded_layers=([exact[0],exact[len(exact)//2],exact[-1]] if exact else []),gradient_depth_audit=result["gradient_depth_audit"],structurally_inactive_parameters=list(getattr(base,"structurally_inactive_parameters",getattr(base,"structurally_inactive_parameter_names",[]))),optimizer_parameter_groups=[{"name":g.get("name","scratch"),"lr":g["lr"],"parameter_count":sum(p.numel() for p in g["params"])} for g in opt.param_groups])
  report_path.write_text(json.dumps(report,indent=2,sort_keys=True))
 if rank==0:print("SMOKE",json.dumps(result,indent=2));(out/"smoke_report.json").write_text(json.dumps(result,indent=2))
 if a.mode=="smoke":destroy_distributed();return
 over=overfit_check(model,train,a,loss_fn,device,a.overfit_steps)
 if rank==0:print("OVERFIT",json.dumps(over,indent=2));(out/"overfit_report.json").write_text(json.dumps(over,indent=2))
 (model.module if isinstance(model,DDP) else model).load_state_dict(pristine);opt,wd=optimizer(model,a);sched=make_scheduler(opt,a,updates)
 if a.mode=="overfit":destroy_distributed();return
 metrics=train_lifecycle(a,model,opt,sched,loss_fn,device,train,val,test,out)
 if rank==0:print("TEST",json.dumps(metrics,indent=2))
 destroy_distributed()
if __name__=="__main__":main()
