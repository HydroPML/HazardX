#!/usr/bin/env python3
"""Isolated, real-data CD patch benchmarks; see docs/S2WCD_efficiency.md."""
import argparse
import csv
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-dir', default='configs/wildfire/s2wcd_sen2_norm')
    p.add_argument('--models', nargs='+', help='Config stems, e.g. cd_unet cd_segformer_b0')
    p.add_argument('--output', default='output/efficiency/s2wcd_256_fp32')
    p.add_argument('--device', default='cuda:0', help='cpu is for validation, not GPU comparison')
    p.add_argument('--patch-size', type=positive, default=256)
    p.add_argument('--native-size', action='store_true', help='Use each config sliding_kernel; not capacity matched')
    p.add_argument('--batch-size', type=positive, default=1)
    p.add_argument('--samples', type=positive, default=3, help='First N test pairs, resident in device memory')
    p.add_argument('--warmup', type=positive, default=30)
    p.add_argument('--iterations', type=positive, default=100)
    p.add_argument('--repeats', type=positive, default=3)
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    p.add_argument('--cpu-threads', type=positive, default=4)
    p.add_argument('--checkpoint-root', help='Strictly load ROOT/<config-stem>/<checkpoint-name>')
    p.add_argument('--checkpoint-name', default='best.pth')
    p.add_argument('--experiment-suffix', default='', help='e.g. _seed43 for ROOT/cd_unet_seed43/best.pth')
    p.add_argument('--skip-flops', action='store_true')
    p.add_argument('--allow-busy-gpu', action='store_true', help='Explicitly allow timing on a GPU already busy before model loading')
    p.add_argument('--repair-flops', action='store_true', help='Recompute FLOPs from saved settings; preserve original timing/memory and back up JSON')
    p.add_argument('--timeout', type=positive, default=3600, help='Per-model timeout in seconds')
    p.add_argument('--resume', action='store_true', help='Reuse successful JSON only for identical settings/config')
    p.add_argument('--list', action='store_true')
    p.add_argument('--worker', help=argparse.SUPPRESS)
    return p


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def signature(args, config):
    settings = vars(args).copy()
    for key in ['worker', 'resume', 'list', 'models', 'output', 'timeout']:
        settings.pop(key, None)
    digest = hashlib.sha256(config.read_bytes() + json.dumps(settings, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(f'{sys.executable}:{platform.node()}:{os.environ.get("CUDA_VISIBLE_DEVICES", "all")}'.encode())
    if args.checkpoint_root:
        cp = Path(args.checkpoint_root) / (config.stem + args.experiment_suffix) / args.checkpoint_name
        if cp.exists():
            st = cp.stat()
            digest.update(f'{cp.resolve()}:{st.st_size}:{st.st_mtime_ns}'.encode())
    return digest.hexdigest()


def gpu_snapshot():
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,uuid,driver_version,utilization.gpu,memory.used',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except Exception as exc:
        return str(exc)



def selected_gpu_state(args):
    if not args.device.startswith('cuda'):
        return None
    logical = int(args.device.split(':')[1]) if ':' in args.device else 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    tokens = visible.split(',') if visible else []
    selector = tokens[logical].strip() if tokens else str(logical)
    try:
        text = subprocess.check_output([
            'nvidia-smi', '-i', selector, '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        util, memory = (float(x.strip()) for x in text.strip().split(','))
        return {'selector': selector, 'utilization_percent': util, 'used_mib': memory,
                'busy': util > 20 and memory > 2048}
    except Exception:
        return None



# Explicit policy: fvcore defaults + dominant fused attention matmuls and the
# repository's analytical selective-scan estimate. No cross-tool fallback.
IGNORED_OPS = [
    'aten::max_pool2d', 'aten::max_pool1d', 'aten::max_pool3d',
    'aten::abs', 'aten::feature_dropout', 'aten::dropout_',
    'aten::adaptive_max_pool2d', 'aten::unfold', 'aten::im2col', 'aten::meshgrid',
    'aten::repeat', 'aten::tile', 'aten::clone', 'aten::expand_as',
    'aten::fill_', 'aten::pixel_shuffle', 'aten::index_select',
    'aten::reciprocal', 'aten::sin', 'aten::cos', 'aten::clamp_min', 'aten::ne',
    'aten::square', 'aten::round', 'aten::le', 'aten::sign', 'aten::log',
    'aten::embedding', 'aten::sort', 'aten::scatter', 'aten::numel', 'aten::zero_',
    'aten::add', 'aten::add_', 'aten::sub', 'aten::rsub',
    'aten::mul', 'aten::mul_', 'aten::div', 'aten::div_',
    'aten::softmax', 'aten::_softmax', 'aten::gelu', 'aten::silu',
    'aten::sigmoid', 'aten::tanh', 'aten::exp', 'aten::neg',
    'aten::pow', 'aten::sqrt', 'aten::rsqrt', 'aten::mean',
    'aten::sum', 'aten::flip', 'aten::pad',
    'prim::PythonOp.CrossScan', 'prim::PythonOp.CrossScanTriton',
    'prim::PythonOp.CrossScan1b1', 'prim::PythonOp.CrossScanTriton1b1',
    'prim::PythonOp.CrossMerge', 'prim::PythonOp.CrossMergeTriton',
]


def attention_flops(inputs, outputs):
    """QK^T + AV, FMA=1; excludes softmax/mask/scale, like unfused attention."""
    import math
    from fvcore.nn.jit_handles import get_shape
    q, k, v = [get_shape(t) for t in inputs[:3]]
    if not q or not k or not v:
        raise ValueError('Unknown attention tensor shape')
    # q (..., heads, query_length, head_dim); k/v may share fewer heads (GQA).
    return math.prod(q[:-2]) * q[-2] * k[-2] * (q[-1] + v[-1])


def scan_flops(inputs, outputs):
    """VMamba analytical estimate: 9*B*L*D*N + B*D*L when D is present.

    Mirrors src/models/ChangeMamba/vmamba.py::flops_selective_scan_fn.
    This is an estimate, not an instruction count of a CUDA kernel.
    """
    from fvcore.nn.jit_handles import get_shape
    b, d, length = get_shape(inputs[0])
    n = get_shape(inputs[2])[1]
    has_d = len(inputs) > 5 and get_shape(inputs[5]) is not None
    return 9*b*length*d*n + (b*d*length if has_d else 0)


def zero_flops(inputs, outputs):
    # fvcore expands PythonOp names *after* its ignore check in old releases;
    # explicit zero handlers are needed for custom permutation/merge Functions.
    return 0


def element_flops(factor, output=False):
    def handle(inputs, outputs):
        import math
        from fvcore.nn.jit_handles import get_shape
        return factor * math.prod(get_shape((outputs if output else inputs)[0]))
    return handle


def layer_norm_flops(inputs, outputs):
    # Same fvcore estimate, without its unnecessary rank <= 5 restriction.
    import math
    from fvcore.nn.jit_handles import get_shape
    return math.prod(get_shape(inputs[0])) * (5 if get_shape(inputs[2]) is not None else 4)


def avg_pool_flops(inputs, outputs):
    import math
    from fvcore.nn.jit_handles import get_shape
    kernel = inputs[1].toIValue()
    if kernel is None:
        kernel = [v.toIValue() for v in inputs[1].node().inputs()]
    return math.prod(get_shape(outputs[0])) * math.prod(kernel)



def bicubic_aa_flops(inputs, outputs):
    # The observed DOFA operation upsamples a 16x16 positional grid to 18x18.
    # Use the same 4x4 support estimate as ordinary bicubic interpolation.
    # Do not silently apply that constant to anti-aliased downsampling, whose
    # support increases with the scale ratio.
    import math
    from fvcore.nn.jit_handles import get_shape
    source, target = get_shape(inputs[0]), get_shape(outputs[0])
    if any(a > b for a,b in zip(source[-2:], target[-2:])):
        raise NotImplementedError('Anti-aliased bicubic downsampling needs a scale-aware count')
    return 16 * math.prod(target)


def configure_analysis(analysis):
    for op in IGNORED_OPS:
        analysis.set_op_handle(op, zero_flops if op.startswith('prim::PythonOp.') else None)
    analysis.set_op_handle('aten::layer_norm', layer_norm_flops)
    analysis.set_op_handle('aten::upsample_bicubic2d', element_flops(16, output=True))
    analysis.set_op_handle('aten::_upsample_bicubic2d_aa', bicubic_aa_flops)
    analysis.set_op_handle('aten::upsample_linear1d', element_flops(2, output=True))
    analysis.set_op_handle('aten::prelu', element_flops(1))
    analysis.set_op_handle('aten::var', element_flops(3))
    analysis.set_op_handle('aten::norm', element_flops(2))
    analysis.set_op_handle('aten::linalg_vector_norm', element_flops(2))
    analysis.set_op_handle('aten::avg_pool2d', avg_pool_flops)
    for op in ['aten::scaled_dot_product_attention',
               'aten::_scaled_dot_product_attention',  # PyTorch 1.13 spelling
               'aten::_scaled_dot_product_flash_attention',
               'aten::_scaled_dot_product_efficient_attention',
               'aten::_scaled_dot_product_flash_attention_for_cpu',
               'aten::_scaled_dot_product_cudnn_attention']:
        analysis.set_op_handle(op, attention_flops)
    for op in ['SelectiveScanMamba', 'SelectiveScanOflex', 'SelectiveScanCore', 'SelectiveScanNRow']:
        analysis.set_op_handle('prim::PythonOp.' + op, scan_flops)
    return analysis


def worker(args, config, result):
    import copy
    import random
    import numpy as np
    import torch
    from src.core.config import Config
    from src.core.augmentation import build_transforms
    from src.core.registry import model_libs, dataset_libs
    from src.tasks.cd import TaskCD

    cfg = Config.from_yaml(config)
    if cfg.task != 'cd' or cfg.dataset['name'] not in ('s2_wcd', 's2wcd'):
        raise ValueError('This benchmark only supports S2WCD CD configurations')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select the correct PyTorch environment')
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    elif args.precision != 'fp32':
        raise ValueError('CPU validation supports fp32 only')

    result.update({
        'model': cfg.model['name'], 'settings': vars(args),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'python': platform.python_version(),
        'device': str(device), 'gpu_before': gpu_snapshot(),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'scope': 'real-data center-crop model forward; excludes IO, preprocessing, H2D, sliding merge and argmax',
        'unit': 'temporal pairs/s', 'fma_convention': 'fvcore: one fused multiply-add = one operation',
        'gpu_state_before': selected_gpu_state(args),
        'tf32': False, 'operator_policy': 'dominant_ops_v2',
        'explicitly_ignored_ops': IGNORED_OPS,
        'scan_estimate': '9*B*L*D*N + B*D*L (D present); VMamba reference',
    })
    size = int(cfg.get('inference.sliding_kernel')) if args.native_size else args.patch_size
    ds_kw = copy.deepcopy(cfg.dataset['test'])
    extra = ds_kw.pop('kwargs', {})
    ds_kw.update(extra)
    transforms = build_transforms(cfg.get('augmentation.val', {}), task='cd')
    ds = dataset_libs[cfg.dataset['name']](**ds_kw, transforms=transforms)
    needed = args.samples * args.batch_size
    if len(ds) < needed:
        raise ValueError(f'Need {needed} test pairs, found {len(ds)}')
    inputs, ids, original_shapes = [], [], []
    for start in range(0, needed, args.batch_size):
        pre_batch, post_batch = [], []
        for idx in range(start, start + args.batch_size):
            pre, post, _, sid = ds[idx]
            h, w = pre.shape[-2:]
            if pre.shape != post.shape or min(h, w) < size:
                raise ValueError(f'{sid}: cannot center crop {size} from {pre.shape}/{post.shape}')
            top, left = (h-size)//2, (w-size)//2
            pre_batch.append(torch.from_numpy(pre[:, top:top+size, left:left+size].copy()))
            post_batch.append(torch.from_numpy(post[:, top:top+size, left:left+size].copy()))
            ids.append(sid)
            original_shapes.append(list(pre.shape))
        inputs.append((torch.stack(pre_batch).to(device), torch.stack(post_batch).to(device)))
    result.update({'sample_ids': ids, 'original_shapes_per_time': original_shapes,
                   'input_shape_per_time': list(inputs[0][0].shape), 'patch_size': size,
                   'batch_size': args.batch_size, 'precision': args.precision,
                   'input_pool_batches': len(inputs)})
    model = model_libs[cfg.model['name']](**cfg.model.get('kwargs', {}))
    result['weights'] = 'constructor weights only (not necessarily fine-tuned)'
    if args.checkpoint_root:
        cp = Path(args.checkpoint_root) / (config.stem + args.experiment_suffix) / args.checkpoint_name
        # Same state_dict envelope as Trainer; strict matching avoids silently dropping weights.
        ckpt = torch.load(str(cp), map_location='cpu')
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        result['weights'] = str(cp.resolve())
    model = model.to(device).eval()
    result['total_params_m'] = sum(p.numel() for p in model.parameters()) / 1e6
    result['trainable_params_m'] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    handler = TaskCD()

    class Forward(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pre, post):
            return handler.run_model(self.model, cfg.model['name'], (pre, post))

    wrapped = Forward(model).eval()
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.precision, torch.float32)
    def autocast():
        return nullcontext() if args.precision == 'fp32' else torch.autocast(device_type=device.type, dtype=dtype)

    if not args.repair_flops:
        measure_forward(args, wrapped, inputs, device, sync, autocast, result)
    else:
        # Match the original post-warmup trace (e.g. cached positional encodings).
        with torch.inference_mode(), autocast():
            for pair in inputs:
                output = wrapped(*pair)
                del output
        sync()

    # Trace AFTER benchmarking, so instrumentation cannot contaminate timing/memory.
    measure_flops(args, wrapped, inputs, autocast, result)
    result['gpu_after'] = gpu_snapshot()


def measure_forward(args, wrapped, inputs, device, sync, autocast, result):
    import torch
    with torch.inference_mode(), autocast():
        # Output is destroyed before the next forward, avoiding overlapping output lifetimes.
        for i in range(args.warmup):
            output = wrapped(*inputs[i % len(inputs)])
            del output
        sync()
        durations, throughputs = [], []
        for _ in range(args.repeats):
            sync()
            t0 = time.perf_counter()
            for i in range(args.iterations):
                output = wrapped(*inputs[i % len(inputs)])
                del output
            sync()
            elapsed = time.perf_counter() - t0
            durations.append(elapsed * 1000 / args.iterations)
            throughputs.append(args.iterations * args.batch_size / elapsed)
        result.update({
            'batch_latency_ms_runs': durations,
            'batch_latency_ms_mean': statistics.mean(durations),
            'batch_latency_ms_std': statistics.stdev(durations) if len(durations) > 1 else 0,
            'throughput_pairs_s_runs': throughputs,
            'throughput_pairs_s_mean': statistics.mean(throughputs),
            'throughput_pairs_s_std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
        })
        if device.type == 'cuda':
            sync()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            for pair in inputs:
                output = wrapped(*pair)
                del output
            sync()
            peak = torch.cuda.max_memory_allocated(device)
            result.update({'baseline_allocated_mib': baseline / 1024**2,
                           'peak_allocated_mib': peak / 1024**2,
                           'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 1024**2,
                           'incremental_peak_mib': (peak-baseline) / 1024**2})
        else:
            result['peak_allocated_mib'] = None

def measure_flops(args, wrapped, inputs, autocast, result):
    import torch
    result['flops_status'] = 'skipped'
    result['gops_fma1_per_pair'] = None
    if not args.skip_flops:
        try:
            from fvcore.nn import FlopCountAnalysis
            with torch.inference_mode(), autocast():
                analysis = configure_analysis(FlopCountAnalysis(wrapped, inputs[0]))
                analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                total = analysis.total()
                unsupported = dict(analysis.unsupported_ops())
                result.update({
                    'counted_gops_fma1_per_pair': total / args.batch_size / 1e9,
                    'unsupported_ops': unsupported,
                    'uncalled_modules': sorted(analysis.uncalled_modules()),
                    'ops_by_operator_per_batch': dict(analysis.by_operator()),
                    'flops_status': 'partial' if unsupported else 'supported_trace',
                    # Only supported traces are eligible for the cost plot. No silent partial counts.
                    'gops_fma1_per_pair': None if unsupported else total / args.batch_size / 1e9,
                })
        except Exception:
            result['flops_status'] = 'failed'
            result['flops_error'] = traceback.format_exc()


FLOPS_KEYS = {
    'flops_status', 'gops_fma1_per_pair', 'counted_gops_fma1_per_pair',
    'unsupported_ops', 'uncalled_modules', 'ops_by_operator_per_batch', 'flops_error',
    'operator_policy', 'explicitly_ignored_ops', 'scan_estimate', 'fma_convention',
}


def repair_worker(args, config, destination):
    import shutil
    from datetime import datetime, timezone
    old = json.loads(destination.read_text())
    if old.get('status') != 'ok' or not old.get('settings'):
        raise ValueError('FLOPs repair requires a successful original inference measurement')
    backup = destination.parent / 'backup_before_flops_repair' / destination.name
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(destination, backup)
    settings = {**vars(args), **old['settings'], 'repair_flops': True, 'worker': str(config)}
    repaired_args = argparse.Namespace(**settings)
    fresh = {}
    method = 'retrace_after_warmup'
    try:
        if old.get('flops_status') == 'supported_trace' and old.get('operator_policy') == 'dominant_ops_v1':
            # v2 adds handlers only for previously unsupported ops and removes
            # the rank bound on the unchanged LayerNorm formula. A supported v1
            # trace therefore has exactly the same total under v2.
            fresh = {**old, 'operator_policy': 'dominant_ops_v2', 'explicitly_ignored_ops': IGNORED_OPS}
            method = 'equivalent_supported_v1_trace_reused'
        else:
            worker(repaired_args, config, fresh)
        for key in ['torch', 'input_shape_per_time', 'sample_ids', 'precision', 'total_params_m']:
            if fresh.get(key) != old.get(key):
                raise ValueError(f'Cannot merge FLOPs: original/new {key} differ: '
                                 f'{old.get(key)} / {fresh.get(key)}')
    except Exception:
        fresh = {'flops_status': 'failed', 'gops_fma1_per_pair': None,
                 'flops_error': traceback.format_exc()}
        print(fresh['flops_error'], file=sys.stderr)
    merged = {k: v for k, v in old.items() if k not in FLOPS_KEYS}
    merged.update({k: v for k, v in fresh.items() if k in FLOPS_KEYS})
    merged['flops_repair'] = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'signature': signature(repaired_args, config),
        'source_backup': str(backup), 'timing_and_memory_preserved': True,
        'trace_device': fresh.get('device'), 'method': method,
    }
    merged['metrics_complete'] = merged.get('flops_status') == 'supported_trace'
    save(destination, merged)


def audit_results(output, configs):
    import math
    report = {'expected_models': len(configs), 'missing_models': [],
              'incomplete_models': {}, 'flops_status_counts': {}, 'gpu_busy_before_models': [],
              'invalid_metrics': {}, 'consistency_issues': {}, 'protocols': [], 'complete_models': 0}
    protocols = set()
    metrics = ['total_params_m', 'trainable_params_m', 'throughput_pairs_s_mean',
               'batch_latency_ms_mean', 'peak_allocated_mib', 'gops_fma1_per_pair']
    for config in configs:
        path = output / (config.stem + '.json')
        if not path.exists():
            report['missing_models'].append(config.stem)
            continue
        row = json.loads(path.read_text())
        status = row.get('flops_status', 'missing')
        report['flops_status_counts'][status] = report['flops_status_counts'].get(status, 0) + 1
        invalid = [k for k in metrics if not isinstance(row.get(k), (int, float))
                   or not math.isfinite(row[k]) or row[k] < 0
                   or (k != 'trainable_params_m' and row[k] == 0)]
        if row.get('device') == 'cpu':
            invalid = [k for k in invalid if k != 'peak_allocated_mib']
        issues = []
        settings = row.get('settings', {})
        if row.get('status') == 'ok':
            protocols.add(json.dumps({k: row.get(k) for k in [
                'input_shape_per_time', 'sample_ids', 'precision', 'torch', 'gpu_name',
                'operator_policy']}, sort_keys=True))
            for key in ['throughput_pairs_s_runs', 'batch_latency_ms_runs']:
                values = row.get(key, [])
                if len(values) != settings.get('repeats') or not all(
                        isinstance(x, (int,float)) and math.isfinite(x) and x > 0 for x in values):
                    issues.append(key + ': invalid repeat values')
                else:
                    prefix = key.removesuffix('_runs')
                    if not math.isclose(statistics.mean(values), row.get(prefix+'_mean', -1), rel_tol=1e-9):
                        issues.append(key + ': mean mismatch')
                    sd = statistics.stdev(values) if len(values)>1 else 0
                    if not math.isclose(sd, row.get(prefix+'_std', -1), rel_tol=1e-9, abs_tol=1e-9):
                        issues.append(key + ': SD mismatch')
            for latency, speed in zip(row.get('batch_latency_ms_runs', []), row.get('throughput_pairs_s_runs', [])):
                if not math.isclose(latency*speed, 1000*row.get('batch_size', 0), rel_tol=1e-9):
                    issues.append('latency/throughput inconsistent')
            if row.get('trainable_params_m', 0) > row.get('total_params_m', 0):
                issues.append('trainable parameters exceed total')
            if row.get('peak_allocated_mib', 0) is not None and row.get('baseline_allocated_mib', 0) > row.get('peak_allocated_mib', 0):
                issues.append('baseline memory exceeds peak')
            counted = row.get('counted_gops_fma1_per_pair')
            by_op = row.get('ops_by_operator_per_batch')
            if counted is not None and by_op is not None and not math.isclose(
                    counted, sum(by_op.values()) / row['batch_size'] / 1e9, rel_tol=1e-9):
                issues.append('operator sum differs from total')
        if issues:
            report['consistency_issues'][config.stem] = issues
        if invalid:
            report['invalid_metrics'][config.stem] = invalid
        if invalid or issues or row.get('status') != 'ok' or status != 'supported_trace':
            report['incomplete_models'][config.stem] = {'status': row.get('status'), 'flops': status}
        else:
            report['complete_models'] += 1
        # This snapshot is taken before model loading; high utilization with large
        # preexisting allocation is concrete evidence of a competing workload.
        for line in row.get('gpu_before', '').splitlines():
            parts = [s.strip() for s in line.split(',')]
            if len(parts) >= 6:
                try:
                    if float(parts[-2].replace('%','').strip()) > 20 and float(parts[-1].replace('MiB','').strip()) > 2048:
                        report['gpu_busy_before_models'].append(config.stem)
                        break
                except ValueError:
                    pass
    report['protocols'] = [json.loads(x) for x in sorted(protocols)]
    report['numerically_complete'] = not (report['missing_models'] or report['incomplete_models'])
    report['uniform_protocol'] = len(protocols) == 1
    report['timing_requires_idle_gpu_rerun'] = bool(report['gpu_busy_before_models'])
    report['ready_for_fair_efficiency_comparison'] = (
        report['numerically_complete'] and report['uniform_protocol'] and not report['timing_requires_idle_gpu_rerun'])
    save(output / 'completeness_report.json', report)
    lines = ['# S2WCD 测量结果审计', '',
             f"预期 {len(configs)} 个配置；数值完整 {report['complete_models']} 个。",
             f"计算量状态：{report['flops_status_counts']}。", '']
    initial_path = output / 'backup_before_flops_repair' / 'completeness_report.json'
    if initial_path.exists():
        initial = json.loads(initial_path.read_text())
        lines += [f"修复前计算量状态：{initial['flops_status_counts']}。", '']
    if report['timing_requires_idle_gpu_rerun']:
        lines += [f"**{len(report['gpu_busy_before_models'])} 个模型的测量前快照显示 GPU 已被其他任务明显占用。**",
                  '速度和延迟应在空闲 GPU 上重测；补算计算量不会消除原速度测量的资源竞争。', '']
    if not report['uniform_protocol']:
        lines += ['**软件/测量协议存在差异；不能视为严格同环境的速度比较。**', '']
    lines += ['计算量口径：G operations/pair，FMA=1；插值、归一化、selective scan 等含解析估计。',
              '详细计数和排除规则见 docs/S2WCD_efficiency.md。', '',
              '| 配置 | 计算量状态 | G operations/pair | pairs/s（原测量） | 峰值 allocated MiB |',
              '|---|---|---:|---:|---:|']
    def fmt(value):
        return f'{value:.3f}' if isinstance(value, (int,float)) else 'N/A'
    for config in configs:
        path = output / (config.stem + '.json')
        row = json.loads(path.read_text()) if path.exists() else {}
        lines.append(f"| {config.stem} | {row.get('flops_status', 'missing')} | "
                     f"{fmt(row.get('gops_fma1_per_pair'))} | "
                     f"{fmt(row.get('throughput_pairs_s_mean'))} | {fmt(row.get('peak_allocated_mib'))} |")
    (output / 'audit_report.md').write_text('\n'.join(lines) + '\n')
    return report


def summarize(output, configs):
    rows = []
    for config in configs:
        path = output / (config.stem + '.json')
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        rows.append({k: v for k, v in row.items() if not isinstance(v, (dict, list))})
    audit_results(output, configs)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_excel(output / 'summary.xlsx', index=False)
    except ImportError:
        pass  # CSV/JSON remain the canonical outputs without optional Excel dependencies.


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = sorted(Path(args.config_dir).resolve().glob('*.yaml'))
    if args.models:
        configs = [c for c in configs if c.stem in args.models]
        missing = set(args.models) - {c.stem for c in configs}
        if missing:
            raise SystemExit(f'Unknown config stems: {sorted(missing)}')
    if not configs:
        raise SystemExit('No configurations found')
    if args.list:
        print('\n'.join(str(c) for c in configs))
        return
    if args.worker:
        config = Path(args.worker)
        result = {'config': str(config), 'config_stem': config.stem,
                  'signature': signature(args, config), 'status': 'ok'}
        if args.repair_flops:
            repair_worker(args, config, output / (config.stem + '.json'))
            return
        try:
            worker(args, config, result)
        except Exception:
            result.update(status='failed', error=traceback.format_exc())
            print(result['error'], file=sys.stderr)
        result['metrics_complete'] = result['status'] == 'ok' and result.get('flops_status') == 'supported_trace'
        save(output / (config.stem + '.json'), result)
        return
    print('GPU state before batch:\n' + gpu_snapshot(), flush=True)
    state = selected_gpu_state(args)
    if not args.repair_flops and not args.allow_busy_gpu and state and state['busy']:
        raise SystemExit('Selected GPU is already busy before model loading. Run on an idle GPU; '
                         '--allow-busy-gpu explicitly permits a contended diagnostic run.')
    for config in configs:
        dest = output / (config.stem + '.json')
        if args.resume and not args.repair_flops and dest.exists():
            old = json.loads(dest.read_text())
            if old.get('metrics_complete') and old.get('status') == 'ok' and old.get('signature') == signature(args, config):
                print(f'[reuse] {config.stem}', flush=True)
                continue
        print(f'[run] {config.stem}', flush=True)
        if not args.repair_flops:
            dest.unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', str(config)]
        with (output / (config.stem + ('.flops_repair.log' if args.repair_flops else '.log'))).open('w') as log:
            try:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                if completed.returncode or not dest.exists():
                    raise RuntimeError(f'Worker exited with code {completed.returncode}')
            except Exception as exc:
                if args.repair_flops:
                    print(f'[repair failed; original retained] {config.stem}: {exc}', flush=True)
                else:
                    save(dest, {'config_stem': config.stem, 'status': 'failed', 'error': str(exc)})
        row = json.loads(dest.read_text())
        print(f"[{row['status']}] {config.stem}; FLOPs={row.get('flops_status', 'unavailable')}", flush=True)
        summarize(output, sorted(Path(args.config_dir).resolve().glob('*.yaml')))
    summarize(output, sorted(Path(args.config_dir).resolve().glob('*.yaml')))
    print(f'Results: {output / "summary.csv"}')
    failed = [c.stem for c in configs if json.loads((output / (c.stem + '.json')).read_text()).get('metrics_complete', False) is not True]
    if failed:
        raise SystemExit(f'Incomplete or failed models ({len(failed)}): {", ".join(failed)}')


if __name__ == '__main__':
    main()
