"""CPU smoke checks for FLOGA stream configurations, routing and gradients.

Run from the project root: python scripts/check_unet_stream_ablation.py
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.augmentation import build_transforms
from src.core.registry import model_libs
from src.core.sliding_window import sliding_window_inference_cd
from src.datasets.wildfire_floga import FLOGADataset
from src.tasks.cd import TaskCD


def main():
    torch.set_num_threads(2)
    base = ROOT / 'configs/wildfire/floga_sen2_norm/stream_ablation'
    configs = [yaml.safe_load((base / f'cd_unet_{v}.yaml').read_text())
               for v in ['single_stream', 'dual_stream', 'dual_stream_shared']]
    for cfg in configs[1:]:
        assert {k: v for k, v in cfg.items() if k not in {'model', 'output_dir'}} == {
            k: v for k, v in configs[0].items() if k not in {'model', 'output_dir'}}
    report = []
    for cfg in configs:
        torch.manual_seed(cfg['seed'])
        name = cfg['model']['name']
        model = model_libs[name](**cfg['model']['kwargs'])
        # Exercise real test data, including the configured preprocessing.
        ds = FLOGADataset(**cfg['dataset']['test'], transforms=build_transforms(
            cfg['augmentation']['val'], task='cd'))
        a, b, label, sid = ds[0]
        assert a.shape == b.shape == (9, 256, 256)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        model.eval()
        with torch.no_grad():
            pred = TaskCD().run_model(model, name, (torch.from_numpy(a[None]), torch.from_numpy(b[None])))
        assert pred.shape == (1, 2, 256, 256) and torch.isfinite(pred).all()
        # Non-square input + backward checks both dates contribute to the loss.
        model.train()
        pre = torch.randn(2, 9, 32, 48, requires_grad=True)
        post = torch.randn(2, 9, 32, 48, requires_grad=True)
        logits = TaskCD().run_model(model, name, (pre, post))
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(2, (2, 32, 48)))
        loss.backward()
        assert pre.grad.abs().sum() > 0 and post.grad.abs().sum() > 0
        if name == 'unet_dualstream':
            assert all(layer.weight.grad is not None for layer in model.fusion)
            if not model.share_encoder:
                assert model.encoder1.block[0].weight is not model.post_blocks[0].block[0].weight
                assert model.post_blocks[0].block[0].weight.grad.abs().sum() > 0
        # Exercise the sliding evaluator's separate-input dispatch.
        model.eval()
        with torch.no_grad():
            result = sliding_window_inference_cd(
                model, pre.detach()[0], post.detach()[0], 32, 16, 2, 'cpu',
                forward_fn=(lambda m, x, y: m(pre=x, post=y)) if name == 'unet_dualstream' else None,
                resolve_fn=lambda output: output, batch_size=1)
        assert result.shape == (32, 48)
        report.append(dict(model=name, kwargs=cfg['model']['kwargs'],
                           params=sum(p.numel() for p in model.parameters()),
                           sample_id=sid, loss=float(loss.detach()), status='passed'))
        del model, logits, pred, loss, pre, post
        gc.collect()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
