#!/usr/bin/env python3
"""Collect 27 primary + 2 external results without hiding environment differences."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from benchmark_s2wcd_efficiency import summarize

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',default=str(ROOT/'output/efficiency/s2wcd_256_fp32_seed43_idle'))
    p.add_argument('--external',default=str(ROOT/'output/efficiency/s2wcd_256_fp32_external'))
    p.add_argument('--output',default=str(ROOT/'output/efficiency/s2wcd_256_fp32_all29'))
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    configs=sorted((ROOT/'configs/wildfire/s2wcd_sen2_norm').glob('*.yaml'))+sorted((ROOT/'configs/wildfire/s2wcd_external').glob('*.yaml'))
    manifest=[]
    for config in configs:
        parent=Path(args.external if config.parent.name=='s2wcd_external' else args.primary)
        source=parent/(config.stem+'.json')
        row=json.loads(source.read_text())
        if not row.get('metrics_complete'):raise RuntimeError(f'Incomplete: {source}')
        if source.resolve()==(out/source.name).resolve():raise ValueError('Use a separate collection directory')
        shutil.copy2(source,out/source.name)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,out/log.name)
        manifest.append({'config':str(config),'source':str(source.resolve()),'torch':row['torch'],
                         'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    (out/'collection_manifest.json').write_text(json.dumps(manifest,indent=2))
    summarize(out,configs)
    report=json.loads((out/'completeness_report.json').read_text())
    assert report['complete_models']==29 and report['numerically_complete']
    assert not report['timing_requires_idle_gpu_rerun']
    print('29 complete models; uniform protocol:',report['uniform_protocol'])
    print(out/'summary.xlsx')

if __name__=='__main__':main()
