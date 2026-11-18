"""Real-data A1--A3 checks plus regression checks against the existing A4 path."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.registry import model_libs
from src.core.sliding_window import sliding_window_inference
from src.datasets.landslide_l4s import Landslide4SenseDataset


def main():
    torch.set_num_threads(2)
    root = ROOT / 'configs/landslide/l4s/terrain_ablation'
    manifest = json.loads((root / 'manifest.json').read_text())
    results = []
    for item in manifest:
        cfg = yaml.safe_load((ROOT / item['config']).read_text())
        base = yaml.safe_load((ROOT / f"configs/landslide/l4s/seg_{item['model']}.yaml").read_text())
        for field in ('augmentation', 'optimizer', 'scheduler', 'training', 'seed'):
            assert cfg[field] == base[field]
        for split, count in [('train', 3799), ('val', 245), ('test', 800)]:
            ds = Landslide4SenseDataset(**cfg['dataset'][split])
            assert len(ds) == count
            if split == 'test':
                image, label, sid = ds[0]
        full_cfg = dict(cfg['dataset']['test'])
        indices = full_cfg.pop('channel_indices')
        full, full_label, full_sid = Landslide4SenseDataset(**full_cfg)[0]
        assert np.array_equal(image, full[indices])
        assert np.array_equal(label, full_label) and sid == full_sid
        assert image.shape == (item['input_channels'], 128, 128) and np.isfinite(image).all()
        torch.manual_seed(cfg['seed'])
        model = model_libs[cfg['model']['name']](**cfg['model']['kwargs'])
        model.eval()
        captured = {}
        def capture(module, args):
            captured['space'] = args[1].detach().clone()
            captured['mask'] = args[5].detach().clone()
        hook = model.encoder.register_forward_pre_hook(capture) if item['model'] == 'galileo_nano' else None
        with torch.no_grad():
            logits = model(torch.from_numpy(image[None]))
        assert logits.shape == (1, 2, 128, 128) and torch.isfinite(logits).all()
        if hook:
            hook.remove()
            sp, mask = captured['space'], captured['mask']
            if item['variant'] == 'A1':
                assert torch.all(mask[..., model._srtm_group_index] == 1)
                assert torch.all(sp[..., [model._slope_index, model._elevation_index]] == 0)
            else:
                assert torch.all(mask[..., model._srtm_group_index] == 0)
                present = model._elevation_index if item['variant'] == 'A2' else model._slope_index
                missing = model._slope_index if item['variant'] == 'A2' else model._elevation_index
                assert torch.equal(sp[..., present], torch.from_numpy(image[None, 12]))
                assert torch.all(sp[..., missing] == 0)
        model.train()
        x = torch.randn(2, item['input_channels'], 32, 32, requires_grad=True)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, torch.randint(2, (2, 32, 32)))
        loss.backward()
        grad = x.grad.abs().sum((0, 2, 3))
        expected = list(range(len(image))) if item['model'] == 'unet' else [i for i in range(len(image)) if i not in (0, 9)]
        assert torch.isfinite(grad).all() and torch.all(grad[expected] > 0)
        if item['model'] == 'galileo_nano':
            # Explicitly track existing A4's omission rather than claiming all
            # twelve optical channels enter the native ten-band encoder.
            assert grad[0] == 0 and grad[9] == 0
        model.eval()
        with torch.no_grad():
            result = sliding_window_inference(model, torch.from_numpy(image[:, :32, :48]),
                                               32, 16, 2, 'cpu', resolve_fn=lambda y: y,
                                               batch_size=1)
        assert result.shape == (32, 48)
        results.append(dict(model=item['model'], variant=item['variant'], status='passed',
                            input_channels=len(image), parameters=sum(p.numel() for p in model.parameters()),
                            original_channel_indices=indices, gradient_channels=expected))
        print(item['model'], item['variant'], 'passed', flush=True)
        del model, loss, pred, x, logits
        gc.collect()
    # Original full-terrain Galileo must agree with its legacy tensor slicing.
    cfg = yaml.safe_load((ROOT / 'configs/landslide/l4s/seg_galileo_nano.yaml').read_text())
    cfg['model']['kwargs']['checkpoint_path'] = str(ROOT / 'pretrained_weight/Galileo_nano_encoder.pt')
    model = model_libs['galileo'](**cfg['model']['kwargs']).eval()
    x = torch.randn(1, 14, 32, 32)
    with torch.no_grad():
        actual = model(x)
        s2 = x[:, model.s2_channel_indices].permute(0, 2, 3, 1).contiguous()
        terrain = x[:, (12, 13)].permute(0, 2, 3, 1).contiguous()
        tokens = model._encode_s2(s2, terrain)
        expected = model.classifier(model.head(model.decoder([(tokens, None)] * 4, (4, 4))))
        expected = torch.nn.functional.interpolate(expected, size=(32, 32), mode='bilinear', align_corners=True)
    assert torch.equal(actual, expected), (actual - expected).abs().max()
    payload = dict(variants=results, A4_legacy_forward_exact_match=True)
    (root / 'smoke_results.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
