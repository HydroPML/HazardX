"""Regression checks for real failures found in S2WCD fvcore traces."""
import unittest
from unittest.mock import patch
import torch
from fvcore.nn import FlopCountAnalysis
from benchmark_s2wcd_efficiency import (
    configure_analysis, attention_flops, scan_flops, parser, repair_worker,
)


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.flip(-1)


class Counters(unittest.TestCase):
    def test_six_dimensional_layer_norm(self):
        model = torch.nn.LayerNorm(8).eval()
        x = torch.randn(1, 2, 3, 1, 7, 8)
        fa = configure_analysis(FlopCountAnalysis(model, (x,)))
        self.assertEqual(fa.total(), x.numel() * 5)
        self.assertFalse(fa.unsupported_ops())

    def test_pythonop_zero_handler(self):
        class Permutation(torch.nn.Module):
            def forward(self, x):
                return CrossScan.apply(x)
        fa = configure_analysis(FlopCountAnalysis(Permutation(), (torch.randn(2,3,4),)))
        self.assertEqual(fa.total(), 0)
        self.assertFalse(fa.unsupported_ops())

    def test_attention_and_scan_formulas(self):
        with patch('fvcore.nn.jit_handles.get_shape', side_effect=lambda x:x):
            self.assertEqual(attention_flops([[2,4,5,8],[2,4,7,8],[2,4,7,6]], None), 3920)
            self.assertEqual(scan_flops([[2,3,5],None,[3,7],None,None,[3]], None), 1920)

    def test_old_pytorch_fused_attention(self):
        import torch.nn.functional as F
        legacy = getattr(F, '_scaled_dot_product_attention', None)
        if legacy is None:
            self.skipTest('PyTorch 1.13 legacy SDPA not installed')
        class Attention(torch.nn.Module):
            def forward(self, q, k, v):
                return legacy(q,k,v, attn_mask=None, dropout_p=0., is_causal=False)[0]
        q,k,v = torch.randn(2,4,5,8),torch.randn(2,4,7,8),torch.randn(2,4,7,8)
        fa = configure_analysis(FlopCountAnalysis(Attention(), (q,k,v)))
        self.assertEqual(fa.total(), 2*4*5*7*16)
        self.assertFalse(fa.unsupported_ops())

    def test_antialiased_bicubic_upsampling(self):
        import torch.nn.functional as F
        class Interpolate(torch.nn.Module):
            def forward(self, x):
                return F.interpolate(x, size=(18,18), mode='bicubic', align_corners=False, antialias=True)
        fa = configure_analysis(FlopCountAnalysis(Interpolate(), (torch.randn(1,3,16,16),)))
        self.assertEqual(fa.total(), 1*3*18*18*16)
        self.assertFalse(fa.unsupported_ops())

    def test_repair_preserves_original_measurements(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'cd_test.yaml'
            config.write_text('task: cd\n')
            destination = Path(tmp) / 'cd_test.json'
            args = parser().parse_args([])
            old = {
                'status': 'ok', 'settings': vars(args), 'torch': 'test',
                'input_shape_per_time': [1,13,256,256], 'sample_ids': ['sample'],
                'precision': 'fp32', 'total_params_m': 1.,
                'throughput_pairs_s_mean': 12., 'peak_allocated_mib': 123.,
                'batch_latency_ms_runs': [83.333], 'signature': 'original-signature',
                'flops_status': 'failed', 'flops_error': 'old tracing failure',
            }
            destination.write_text(json.dumps(old))
            def fake_worker(args, config, result):
                result.update({**old, 'flops_status': 'supported_trace',
                               'flops_error': None, 'gops_fma1_per_pair': 2.,
                               'throughput_pairs_s_mean': 999., 'peak_allocated_mib': 999.})
            with patch('benchmark_s2wcd_efficiency.worker', side_effect=fake_worker):
                repair_worker(args, config, destination)
            updated = json.loads(destination.read_text())
            self.assertEqual(updated['throughput_pairs_s_mean'], 12.)
            self.assertEqual(updated['peak_allocated_mib'], 123.)
            self.assertEqual(updated['batch_latency_ms_runs'], old['batch_latency_ms_runs'])
            self.assertEqual(updated['signature'], 'original-signature')
            self.assertEqual(updated['gops_fma1_per_pair'], 2.)
            self.assertTrue(updated['metrics_complete'])
            self.assertEqual(json.loads((destination.parent/'backup_before_flops_repair'/destination.name).read_text()), old)

    def test_pool_and_interpolation(self):
        import torch.nn.functional as F
        class Pool(torch.nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, kernel_size=2)
        x = torch.randn(1,2,8,8)
        fa = configure_analysis(FlopCountAnalysis(Pool(), (x,)))
        self.assertEqual(fa.total(), 1*2*4*4*4)
        self.assertFalse(fa.unsupported_ops())
        class Interpolate(torch.nn.Module):
            def forward(self,x):
                return F.interpolate(x, size=(16,16), mode='bicubic', align_corners=False)
        fa = configure_analysis(FlopCountAnalysis(Interpolate(), (x,)))
        self.assertEqual(fa.total(), 1*2*16*16*16)
        self.assertFalse(fa.unsupported_ops())


if __name__ == '__main__':
    unittest.main()
