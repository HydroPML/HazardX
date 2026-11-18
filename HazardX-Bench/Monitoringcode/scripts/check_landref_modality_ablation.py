"""Verify LANDREF B1--B4 data selection, native routing and input gradients."""
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
from src.core.sliding_window import sliding_window_inference_cd
from src.datasets.landslide_landref_cd import LANDREFCDDataset
from src.models.Galileo.GalileoLANDREF import S2_FEATURES, SAR_FEATURES
from src.models.Galileo import galileo_core as galileo
from src.tasks.cd import TaskCD


def check_native_routing(model):
    channels = model.in_channels
    x = torch.arange(1, channels + 1, dtype=torch.float32).view(1, channels, 1, 1).expand(1, channels, 16, 16)
    packed = model._pack_inputs(x)
    values, space, _, _, mask, space_mask, _, _, months = packed
    for t, date in enumerate(('PRE1', 'POST1')):
        for band, destination in zip(S2_FEATURES, model._galileo_s2_destinations):
            assert torch.all(values[..., t, destination] == model.lookup[f'{date}_{band}'] + 1)
        sar_group = tuple(galileo.SPACE_TIME_BANDS_GROUPS_IDX).index('S1')
        assert torch.all(mask[..., t, sar_group] == (0 if model.has_sar else 1))
        if model.has_sar:
            for band, dest in [('gVV', 'VV'), ('gVH', 'VH')]:
                assert torch.all(values[..., t, galileo.SPACE_TIME_BANDS.index(dest)] == model.lookup[f'{date}_{band}'] + 1)
    assert months.shape == (1, 2)
    assert torch.all(space_mask[..., model._srtm_group_index] == (0 if model.has_terrain else 1))
    if model.has_terrain:
        assert torch.all(space[..., model._elevation_index] == model.lookup['None_DEM'] + 1)
        assert torch.all(space[..., model._slope_index] == model.lookup['None_SLOPE'] + 1)


def main():
    torch.set_num_threads(2)
    base = ROOT / 'configs/landslide/landref_cd/modality_ablation'
    manifest = json.loads((base / 'manifest.json').read_text())
    report = []
    configs = {}
    for item in manifest:
        cfg = yaml.safe_load((ROOT / item['config']).read_text())
        configs[item['model'], item['variant']] = cfg
        torch.manual_seed(cfg['seed'])
        inp = cfg['dataset']['input']
        names = inp['pre_features'] + inp['post_features']
        assert len(names) == cfg['model']['kwargs']['in_channels'] == item['channels']
        assert any('gVV' in n for n in names) == cfg['experiment']['sar']
        assert ('None_DEM' in names) == cfg['experiment']['terrain']
        for split, expected_len in [('train', 355), ('val', 88), ('test', 44)]:
            ds = LANDREFCDDataset(**cfg['dataset'][split], input_cfg=inp)
            assert len(ds) == expected_len
            if split == 'test':
                pre, post, label, sid = ds[0]
        assert pre.shape[1:] == post.shape[1:] == (128, 128)
        assert len(pre) + len(post) == item['channels']
        assert np.isfinite(pre).all() and np.isfinite(post).all()
        model = model_libs[cfg['model']['name']](**cfg['model']['kwargs'])
        if cfg['model']['name'] == 'galileo_landref':
            check_native_routing(model)
        model.eval()
        with torch.no_grad():
            logits = TaskCD().run_model(model, cfg['model']['name'],
                                       (torch.from_numpy(pre[None]), torch.from_numpy(post[None])))
        assert logits.shape == (1, 2, 128, 128) and torch.isfinite(logits).all()
        # Nonzero gradient in every effective input channel detects silently
        # ignored SAR/terrain, including the eight derived SAR channels.
        model.train()
        x = torch.randn(2, item['channels'], 32, 32, requires_grad=True)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, torch.randint(2, (2, 32, 32)))
        loss.backward()
        gradients = x.grad.abs().sum((0, 2, 3))
        assert torch.isfinite(gradients).all() and torch.all(gradients > 0), (item, gradients)
        model.eval()
        with torch.no_grad():
            result = sliding_window_inference_cd(
                model, torch.from_numpy(pre[:, :32, :48]), torch.from_numpy(post[:, :32, :48]),
                32, 16, 2, 'cpu', resolve_fn=lambda output: output, batch_size=1)
        assert result.shape == (32, 48)
        report.append(dict(model=item['model'], variant=item['variant'], channels=item['channels'],
                           pre_channels=len(pre), post_channels=len(post), sample_id=sid,
                           parameters=sum(p.numel() for p in model.parameters()),
                           minimum_input_gradient=float(gradients.min()), status='passed'))
        print(item['model'], item['variant'], 'passed', flush=True)
        del model, logits, pred, loss, x
        gc.collect()
    # Only input selection and channel adapter metadata vary within a model.
    for family in ('unet', 'galileo_nano'):
        for variant in ('B2', 'B3', 'B4'):
            for key in ('augmentation', 'optimizer', 'scheduler', 'loss', 'training', 'inference', 'seed'):
                assert configs[family, variant][key] == configs[family, 'B1'][key]
    # The two models see exactly the same selected/scaled features.
    for variant in ('B1', 'B2', 'B3', 'B4'):
        assert configs['unet', variant]['dataset'] == configs['galileo_nano', variant]['dataset']
    path = base / 'smoke_results.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
