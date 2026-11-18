#!/usr/bin/env python3
"""Measure original diasterother models; preserve their environment and checkpoints."""
import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_s2wcd_efficiency import (measure_forward, measure_flops, gpu_snapshot,
    selected_gpu_state, save, IGNORED_OPS)


def run(args, config):
    import numpy as np
    import torch
    import yaml
    desc=yaml.safe_load(config.read_text())
    cfg=yaml.safe_load(Path(desc['source_config']).read_text())
    checkpoint=Path(desc['checkpoint'])
    # Exact architecture/settings used for the saved downstream model.
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    trained_cfg=payload.get('config',cfg)
    cfg=trained_cfg
    sys.path.insert(0,desc['repository'])
    module,name=desc['class_path'].rsplit('.',1)
    factory=getattr(importlib.import_module(module),name)
    kwargs=dict(cfg['model']['kwargs'])
    kwargs['pretrained']=False
    kwargs['pretrained_checkpoint_path']=None
    torch.manual_seed(42);np.random.seed(42)
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    before=gpu_snapshot();state=selected_gpu_state(args)
    if state and state['busy']:
        raise RuntimeError('Selected GPU already busy before measurement')
    model=factory(**kwargs)
    model.load_state_dict(payload['model'],strict=True)
    model.to(args.device).eval()
    del payload
    base=Path(cfg['dataset']['root'])
    split=cfg['dataset']['splits']['test']['data_list_path']
    ids=[x.strip() for x in (base/split).read_text().splitlines() if x.strip()][:args.samples]
    pool=[];shapes=[]
    for sid in ids:
        pre=np.load(base/'T1_sen2'/f'{sid}.npy').astype('float32')
        post=np.load(base/'T2_sen2'/f'{sid}.npy').astype('float32')
        h,w=pre.shape[-2:];s=args.patch_size;top,left=(h-s)//2,(w-s)//2
        if min(h,w)<s or pre.shape!=post.shape:raise ValueError(f'Invalid input shape: {sid}')
        shapes.append(list(pre.shape))
        # Exact early-fusion training preprocessing (outside timing).
        pre=np.nan_to_num(np.clip(pre[:,top:top+s,left:left+s],0,10000)/10000)
        post=np.nan_to_num(np.clip(post[:,top:top+s,left:left+s],0,10000)/10000)
        pool.append((torch.from_numpy(pre.copy()).unsqueeze(0).to(args.device),
                     torch.from_numpy(post.copy()).unsqueeze(0).to(args.device)))
    if len(pool)!=args.samples:raise ValueError('Insufficient samples')
    class Forward(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,pre,post):return self.model(torch.cat((pre,post),dim=1))
    wrapped=Forward(model).eval()
    result={'status':'ok','config':str(config),'config_stem':config.stem,
        'source_config':desc['source_config'],'source_repository':desc['repository'],
        'model':desc['model'],'variant':desc['variant'],'settings':vars(args),
        'weights':str(checkpoint),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'python':platform.python_version(),
        'device':args.device,'gpu_name':torch.cuda.get_device_name(), 'gpu_before':before,
        'gpu_state_before':state,'patch_size':args.patch_size,'batch_size':1,'precision':'fp32',
        'input_shape_per_time':list(pool[0][0].shape),'sample_ids':ids,'original_shapes_per_time':shapes,
        'input_pool_batches':len(pool),'total_params_m':sum(p.numel() for p in model.parameters())/1e6,
        'trainable_params_m':sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6,
        'operator_policy':'dominant_ops_v2','explicitly_ignored_ops':IGNORED_OPS,
        'unit':'temporal pairs/s','fma_convention':'fvcore: one fused multiply-add = one operation',
        'tf32':False,'scope':'real-data center-crop model forward; excludes IO/preprocessing/H2D/sliding merge/argmax',
        'preprocessing':'source training early-fusion: clip DN [0,10000]/10000, learned native-band projection',
        'framework_note':'Measured using the original model environment; different PyTorch from primary 27 models.'}
    sync=lambda:torch.cuda.synchronize(args.device)
    measure_forward(args,wrapped,pool,torch.device(args.device),sync,nullcontext,result)
    measure_flops(args,wrapped,pool,nullcontext,result)
    result['gpu_after']=gpu_snapshot()
    result['metrics_complete']=result.get('flops_status')=='supported_trace'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-dir',default=str(ROOT/'configs/wildfire/s2wcd_external'))
    p.add_argument('--output',default=str(ROOT/'output/efficiency/s2wcd_256_fp32_external'))
    p.add_argument('--models',nargs='+')
    p.add_argument('--device',default='cuda:0');p.add_argument('--patch-size',type=int,default=256)
    p.add_argument('--samples',type=int,default=3);p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--worker')
    args=p.parse_args();args.batch_size=1;args.precision='fp32';args.skip_flops=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.worker:
        config=Path(args.worker)
        try:result=run(args,config)
        except Exception:result={'status':'failed','config_stem':config.stem,'error':traceback.format_exc()};print(result['error'])
        save(out/(config.stem+'.json'),result)
        return
    files=sorted(Path(args.config_dir).glob('*.yaml'))
    if args.models:files=[f for f in files if f.stem in args.models]
    for config in files:
        print('[run]',config.stem,flush=True)
        with (out/(config.stem+'.log')).open('w') as handle:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker',str(config)],stdout=handle,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/(config.stem+'.json')).read_text())
        print(config.stem,result.get('status'),result.get('flops_status'),result.get('unsupported_ops'),flush=True)
    if any(not json.loads((out/(f.stem+'.json')).read_text()).get('metrics_complete') for f in files):raise SystemExit('Some metrics incomplete; inspect logs/JSON')

if __name__=='__main__':main()
