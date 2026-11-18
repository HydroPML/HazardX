"""Unified leakage-free LandslideCast 2-D benchmark for nine baselines."""
from __future__ import annotations
import argparse,json,math,os,time,random
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.landslidecast_2d_dataset import LandslideCast2DDataset,TIMES,validate_splits
from models.adapters import *
from models.adapters.landslide_common import benchmark_loss

ROOT=Path(__file__).resolve().parent
MODELS={"UNet3D-Scratch":LandslideUNet3D,"FNO3D-Scratch":LandslideFNO3D,"ConvLSTM-Scratch":LandslideConvLSTM,"EarthFormer-Scratch":LandslideEarthFormer,"PDE-T-SC-S-PT-FT":LandslidePDET,"DPOT-S-PT-FT":LandslideDPOT,"Poseidon-T-PT-FT":LandslidePoseidon,"GPhyT-S-PT-FT":LandslideGPhyT,"Prithvi-EO2-300M-PT-FT":LandslidePrithvi,"GPhyT-S-PT-FT-Euler":LandslideGPhyTEuler}
PT=set(list(MODELS)[4:])
AMP_MODELS={"UNet3D-Scratch","ConvLSTM-Scratch","PDE-T-SC-S-PT-FT","Poseidon-T-PT-FT","GPhyT-S-PT-FT","Prithvi-EO2-300M-PT-FT","GPhyT-S-PT-FT-Euler"}
# Exact priority recipes: FloodSim seven-baseline script; UNet/FNO from FloodCast two-baseline.
OPT={
 "UNet3D-Scratch":("Adam",1e-4,1e-4,(.9,.999),"cosine",1.),"FNO3D-Scratch":("Adam",1e-4,1e-4,(.9,.999),"cosine",1.),
 "ConvLSTM-Scratch":("Adam",1e-4,0.,(.9,.999),"onecycle",0.),"EarthFormer-Scratch":("AdamW",3e-4,0.,(.9,.999),"warmup_cosine_20",1.),
 "PDE-T-SC-S-PT-FT":("AdamW",4e-5,0.,(.9,.999),"none",0.),"DPOT-S-PT-FT":("Adam",3e-5,0.,(.9,.9),"dpot_cycle",1.),
 "Poseidon-T-PT-FT":("AdamW",5e-5,1e-6,(.9,.999),"cosine",5.),"GPhyT-S-PT-FT":("AdamW",1e-4,1e-4,(.9,.999),"gphyt_warmup_cosine",5.),"GPhyT-S-PT-FT-Euler":("AdamW",1e-4,1e-4,(.9,.999),"gphyt_warmup_cosine",5.),
 "Prithvi-EO2-300M-PT-FT":("AdamW",5e-5,.05,(.9,.999),"cosine",1.)}
PATHS={
 "EarthFormer-Scratch":("earthformer/earth-forecasting-transformer-7732b03bdb366110563516c3502315deab4c2026",None),
 "PDE-T-SC-S-PT-FT":("pdet/pde-transformer-850e09d352487922e4a853ee90a2624717769cc7","PDE-T/sc-s/diffusion_pytorch_model.safetensors"),
 "DPOT-S-PT-FT":("dpot/DPOT-dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17","DPOT-S/model_S.pth"),
 "Poseidon-T-PT-FT":("poseidon/poseidon-b8fa28f59bd7f7673323f28d11a12c6f3a215c61","Poseidon-T/pytorch_model.bin"),
 "GPhyT-S-PT-FT":("gphyt/General-Physics-Transformer-df1527c4773ababc6f8af0d117fb5803fa3bf74b","GPhyT-S/gphyt-S.pth"),
 "GPhyT-S-PT-FT-Euler":("gphyt/General-Physics-Transformer-df1527c4773ababc6f8af0d117fb5803fa3bf74b","GPhyT-S/gphyt-S.pth"),
 "Prithvi-EO2-300M-PT-FT":(None,"../Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt")}

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument("--model-type",required=True,choices=MODELS); p.add_argument("--data-path",default=str(ROOT.parent/"Data2/2d_t10.h5")); p.add_argument("--output-dir",default=str(ROOT/"results/LandslideCast")); p.add_argument("--mode",choices=("smoke","train"),default="smoke"); p.add_argument("--epochs",type=int,default=100); p.add_argument("--validate-every",type=int,default=10); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--accum-steps",type=int,default=8); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--seed",type=int,default=0); p.add_argument("--smoke-steps",type=int,default=10); p.add_argument("--eval-max-batches",type=int,default=1); p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=None); p.add_argument("--resume",default=""); p.add_argument("--official-root",default=""); p.add_argument("--pretrained-checkpoint",default=""); return p.parse_args()
def resolve(a):
 if a.mode=="train" and (a.epochs!=100 or a.validate_every!=10): raise ValueError("benchmark fixes epochs=100 and validation_every=10")
 if min(a.batch_size,a.accum_steps,a.smoke_steps)<1: raise ValueError("batch/accum/smoke values must be positive")
 validate_splits(); root,checkpoint=PATHS.get(a.model_type,(None,None))
 if root and not a.official_root: a.official_root=str(ROOT/"official_sources/LandslideCast"/root)
 if checkpoint and not a.pretrained_checkpoint: a.pretrained_checkpoint=str(ROOT/"checkpoints/LandslideCast"/checkpoint)
 required_amp=a.model_type in AMP_MODELS
 if a.amp is not None and bool(a.amp)!=required_amp: raise ValueError("model precision is fixed by the benchmark table")
 a.amp=required_amp
 return a
def distributed():
 world=int(os.environ.get("WORLD_SIZE","1")); local=int(os.environ.get("LOCAL_RANK","0")); rank=int(os.environ.get("RANK","0"))
 if world>1: torch.cuda.set_device(local); dist.init_process_group("nccl")
 return rank,world,local
def move(batch,device): return {key:(value.to(device,non_blocking=True) if torch.is_tensor(value) else value) for key,value in batch.items()}
def synchronize(device):
 if device.type=="cuda": torch.cuda.synchronize(device)
def loader(path,split,a,rank,world,shuffle):
 dataset=LandslideCast2DDataset(path,split); sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=shuffle) if world>1 else None
 return dataset,DataLoader(dataset,batch_size=a.batch_size,shuffle=shuffle and sampler is None,sampler=sampler,num_workers=a.num_workers,pin_memory=True,drop_last=False,persistent_workers=a.num_workers>0)
def build_model(a,out,device):
 model=MODELS[a.model_type](official_root=a.official_root,checkpoint_path=a.pretrained_checkpoint,report_path=str(out/"pretrained_loading_report.json"))
 if a.model_type=="Prithvi-EO2-300M-PT-FT": model.enable_activation_checkpointing()
 return model.to(device)
def build_optimizer(model,a):
 name,lr,wd,betas,_,_=OPT[a.model_type]; parameters=model.parameter_groups(lr,5e-4) if a.model_type=="Poseidon-T-PT-FT" else (p for p in model.parameters() if p.requires_grad)
 cls=torch.optim.AdamW if name=="AdamW" else torch.optim.Adam
 return cls(parameters,lr=lr,weight_decay=wd,betas=betas,foreach=False)
def build_scheduler(optimizer,a,updates):
 _,lr,_,_,kind,_=OPT[a.model_type]; total=max(1,a.epochs*updates)
 if kind=="none": return None
 if kind=="onecycle": return torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=total)
 if kind=="dpot_cycle": return torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=3e-5,total_steps=total,pct_start=.2,div_factor=1e4,final_div_factor=1e4)
 if kind in ("warmup_cosine_20","gphyt_warmup_cosine"):
  warm=min(total-1,max(1,int(total*.2) if kind=="warmup_cosine_20" else min(5000,int(total*.2))))
  def factor(step):
   if step<warm: return .001+.999*step/max(1,warm)
   progress=(step-warm)/max(1,total-warm); return .5*(1+math.cos(math.pi*progress))
  return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
 return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total)
def parameters(model):
 total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 return total,trainable
def audit_optimizer_coverage(model,optimizer):
 expected={id(p) for p in model.parameters() if p.requires_grad}; actual={id(p) for group in optimizer.param_groups for p in group["params"]}
 if expected!=actual: raise AssertionError(f"optimizer coverage mismatch: missing={len(expected-actual)}, unexpected={len(actual-expected)}")
def finalize_loading_report(model,a,out,total,trainable):
 if a.model_type not in PT: return
 path=out/"pretrained_loading_report.json"; payload=json.loads(path.read_text())
 loaded=payload.get("exact_loaded",[])+payload.get("converted",[])
 if not loaded: raise RuntimeError(f"{a.model_type}: no pretrained tensors recorded")
 payload.update(model=a.model_type,checkpoint_path=str(Path(a.pretrained_checkpoint).resolve()),loaded_backbone_parameter_count=payload.get("loaded_parameter_count",0),parameters_total=total,parameters_trainable=trainable,loaded_depth_representatives={"shallow":loaded[0],"middle":loaded[len(loaded)//2],"deep":loaded[-1]})
 path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
def metrics(pred,target):
 result={}
 for index,name in enumerate(("h","u")):
  error=pred[:,:,index]-target[:,:,index]; result[f"RMSE_{name}"]=error.square().mean().sqrt(); result[f"MAE_{name}"]=error.abs().mean(); result[f"RelL2_{name}"]=torch.linalg.vector_norm(error.flatten(1),dim=1).div(torch.linalg.vector_norm(target[:,:,index].flatten(1),dim=1)+1e-12).mean(); last=error[:,-1]; result[f"RMSE_{name}@t80"]=last.square().mean().sqrt(); result[f"MAE_{name}@t80"]=last.abs().mean(); result[f"RelL2_{name}@t80"]=torch.linalg.vector_norm(last.flatten(1),dim=1).div(torch.linalg.vector_norm(target[:,-1,index].flatten(1),dim=1)+1e-12).mean()
 return result
@torch.no_grad()
def evaluate(model,data,device,max_batches,amp):
 model.eval(); sums={}; count=0
 for index,batch in enumerate(data):
  if max_batches and index>=max_batches: break
  batch=move(batch,device)
  with torch.autocast("cuda",enabled=amp): pred=model(batch)
  loss,h,u=benchmark_loss(pred.float(),batch["target"].float()); values={"objective":loss,"official_RelL2_h":h,"official_RelL2_u":u,**metrics(pred.float(),batch["target"].float())}
  for key,value in values.items(): sums[key]=sums.get(key,0.)+float(value)
  count+=1
 if count==0: raise RuntimeError("evaluation processed zero batches")
 return {key:value/count for key,value in sums.items()}
def gradient_audit(model):
 module=model.module if isinstance(model,DDP) else model; candidates=[(name,p) for name,p in module.named_parameters() if p.requires_grad and ("backbone" in name or a.model_type not in PT)]
 probes={label:candidates[index] for label,index in (("shallow",0),("middle",len(candidates)//2),("deep",len(candidates)-1))}
 report={}
 for label,(name,p) in probes.items(): report[label]={"parameter":name,"finite":p.grad is not None and bool(torch.isfinite(p.grad).all()),"nonzero":p.grad is not None and bool(torch.count_nonzero(p.grad))}
 return report
def save(path,model,optimizer,scheduler,epoch,best):
 module=model.module if isinstance(model,DDP) else model; torch.save({"model":module.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":None if scheduler is None else scheduler.state_dict(),"epoch":epoch-1,"next_epoch":epoch,"best_validation":best},path)
def main():
 global a; a=resolve(parse_args()); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); rank,world,local=distributed(); device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu"); out=Path(a.output_dir)/a.model_type; out.mkdir(parents=True,exist_ok=True)
 train_set,train_data=loader(a.data_path,"train",a,rank,world,True); _,val_data=loader(a.data_path,"val",a,rank,world,False); _,test_data=loader(a.data_path,"test",a,rank,world,False)
 model=build_model(a,out,device); raw=model; total,trainable=parameters(model); optimizer=build_optimizer(model,a); audit_optimizer_coverage(model,optimizer); finalize_loading_report(model,a,out,total,trainable); updates=max(1,math.ceil(len(train_data)/a.accum_steps)); scheduler=build_scheduler(optimizer,a,updates)
 if world>1: model=DDP(model,device_ids=[local],find_unused_parameters=False)
 if rank==0:
  first=train_set[0]; print("DATASET_AUDIT",json.dumps({"path":a.data_path,"scenario_count":12000,"first_train":train_set.names[0],"last_train":train_set.names[-1],"raw_shape":[1,200,200,9,4],"channels":["fai","z","h","u"],"times":TIMES.tolist(),"h0_shape":list(first["h0"].shape),"target_shape":list(first["target"].shape),"split_counts":[10000,1000,1000],"z_norm_range":[float(first["z_norm"].min()),float(first["z_norm"].max())],"h0_norm_range":[float(first["h0_norm"].min()),float(first["h0_norm"].max())]})); train_set.close()
  print("OPTIMIZATION",json.dumps({"model":a.model_type,"optimizer":OPT[a.model_type][0],"lr":OPT[a.model_type][1],"weight_decay":OPT[a.model_type][2],"betas":OPT[a.model_type][3],"scheduler":OPT[a.model_type][4],"grad_clip":OPT[a.model_type][5],"precision":"AMP FP16" if a.amp else "FP32","epochs":100,"validation_every":10,"effective_batch":a.batch_size*a.accum_steps*world,"num_workers":a.num_workers,"seed":a.seed,"parameters_total":total,"parameters_trainable":trainable}))
 start_epoch=0; best=float("inf")
 if a.resume:
  state=torch.load(a.resume,map_location="cpu"); raw.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start_epoch=state.get("next_epoch",state["epoch"]); best=state["best_validation"]
  if scheduler is not None and state.get("scheduler") is not None: scheduler.load_state_dict(state["scheduler"])
 max_epochs=1 if a.mode=="smoke" else a.epochs; max_steps=a.smoke_steps if a.mode=="smoke" else 0; scaler=torch.amp.GradScaler("cuda",enabled=a.amp,init_scale=1.); optimizer.zero_grad(set_to_none=True); step_count=0; amp_skipped=0
 for epoch in range(start_epoch,max_epochs):
  if hasattr(train_data.sampler,"set_epoch"): train_data.sampler.set_epoch(epoch)
  model.train(); torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None
  for step,batch in enumerate(train_data):
   if max_steps and step>=max_steps: break
   batch=move(batch,device); synchronize(device); begin=time.perf_counter()
   with torch.autocast("cuda",enabled=a.amp): pred=model(batch); loss,h,u=benchmark_loss(pred.float(),batch["target"].float()); scaled=loss/a.accum_steps
   synchronize(device); forward_time=time.perf_counter()-begin; begin=time.perf_counter(); scaler.scale(scaled).backward(); synchronize(device); backward_time=time.perf_counter()-begin
   if (step+1)%a.accum_steps==0 or (max_steps and step+1==max_steps):
    scaler.unscale_(optimizer); clip=OPT[a.model_type][5]
    if clip: torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
    audit=gradient_audit(model) if a.model_type in PT else {}; old_scale=scaler.get_scale(); synchronize(device); begin=time.perf_counter(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); synchronize(device); optimizer_time=time.perf_counter()-begin
    succeeded=(not a.amp) or scaler.get_scale()>=old_scale
    if succeeded:
     if scheduler is not None: scheduler.step()
     step_count+=1
    else: amp_skipped+=1
    if rank==0: print("TRAIN",json.dumps({"epoch":epoch,"update":step_count,"optimizer_step_succeeded":succeeded,"amp_skipped_updates":amp_skipped,"shape":list(pred.shape),"loss":float(loss),"RelL2_h":float(h),"RelL2_u":float(u),"forward_seconds":forward_time,"backward_seconds":backward_time,"optimizer_seconds":optimizer_time,"peak_allocated":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,"peak_reserved":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,"gradient_audit":audit}))
  validate_now=a.mode=="smoke" or (epoch+1)%a.validate_every==0
  if validate_now:
   validation=evaluate(model,val_data,device,a.eval_max_batches if a.mode=="smoke" else 0,a.amp)
   if rank==0:
    print("VALIDATION",json.dumps(validation))
    if validation["objective"]<best: best=validation["objective"]; save(out/"best_validation.pt",model,optimizer,scheduler,epoch+1,best)
  if rank==0:
   save(out/"last.pt",model,optimizer,scheduler,epoch+1,best)
   print("CHECKPOINT",json.dumps({"epoch":epoch,"next_epoch":epoch+1,"path":str(out/"last.pt"),"rolling_overwrite":True}))
 if rank==0 and not (out/"best_validation.pt").exists(): save(out/"best_validation.pt",model,optimizer,scheduler,max_epochs,best)
 if world>1: dist.barrier()
 best_state=torch.load(out/"best_validation.pt",map_location="cpu"); raw.load_state_dict(best_state["model"]); test=evaluate(model,test_data,device,a.eval_max_batches if a.mode=="smoke" else 0,a.amp)
 if rank==0:
  print("TEST",json.dumps(test)); probe=move(next(iter(val_data)),device); raw.eval()
  with torch.no_grad(),torch.autocast("cuda",enabled=a.amp): before=raw(probe).float()
  reload_model=build_model(a,out,device); finalize_loading_report(reload_model,a,out,total,trainable); reload_model.load_state_dict(best_state["model"]); reload_model.eval()
  with torch.no_grad(),torch.autocast("cuda",enabled=a.amp): after=reload_model(probe).float()
  identical=torch.equal(before,after) if not a.amp else torch.allclose(before,after,rtol=1e-5,atol=1e-6)
  if not identical: raise AssertionError("checkpoint reload output mismatch")
  summary={"model":a.model_type,"mode":a.mode,"train_updates":step_count,"amp_skipped_updates":amp_skipped,"validation":validation,"test":test,"reload_consistent":identical,"parameters_total":total,"parameters_trainable":trainable}; (out/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print("COMPLETE",json.dumps(summary))
 if world>1: dist.destroy_process_group()
if __name__=="__main__":
 from landslidecast_accelerated_runner import main as accelerated_main
 accelerated_main()

