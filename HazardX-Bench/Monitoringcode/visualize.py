"""Save qualitative segmentation/change-detection results for an experiment.

Example:
    python visualize.py --exp_path output/s2wcd_sen2/cd_changemamba
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Avoid Albumentations' network-based update check on compute nodes.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.core.config import Config
from src.core.registry import DUAL_INPUT_MODELS
from src.core.sliding_window import sliding_window_inference, sliding_window_inference_cd
from src.core.trainer import Trainer


def _save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a binary class-index mask as a displayable 0/255 PNG."""
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _error_map(pred: np.ndarray, target: np.ndarray) -> Image.Image:
    """TN=black, TP=green, FP=red, FN=blue, ignored=gray."""
    valid = target != 255
    target_fg = target > 0
    pred_fg = pred > 0
    out = np.zeros((*target.shape, 3), dtype=np.uint8)
    out[valid & target_fg & pred_fg] = (0, 200, 0)
    out[valid & ~target_fg & pred_fg] = (230, 30, 30)
    out[valid & target_fg & ~pred_fg] = (30, 100, 240)
    out[~valid] = (128, 128, 128)
    return Image.fromarray(out, mode="RGB")


def _rgb_image(data_dir: Path, folder: str, sample_id: str) -> Image.Image:
    """Load a prepared CD RGB view, or derive one from its multiband NPY."""
    path = data_dir / folder / f"{sample_id}.png"
    if path.is_file():
        return Image.open(path).convert("RGB")

    # FLOGA and similar adapters store the actual data below patch{size}.
    # Their first three Sentinel-2 channels are BGR, hence [2, 1, 0].
    npy_folder = folder.removesuffix("_vis")
    npy_path = data_dir / npy_folder / f"{sample_id}.npy"
    if npy_path.is_file():
        array = np.load(npy_path).astype(np.float32)
        if array.ndim != 3 or array.shape[0] < 3:
            raise ValueError(f"Expected a CHW multiband array, got {array.shape} at {npy_path}")
        rgb = np.moveaxis(array[[2, 1, 0]], 0, -1)
        output = np.zeros_like(rgb, dtype=np.uint8)
        for channel in range(3):
            values = rgb[..., channel]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            low, high = np.percentile(values[finite], (2, 98))
            if high <= low:
                high = low + 1.0
            output[..., channel] = (
                np.clip((values - low) / (high - low), 0, 1) * 255
            ).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")

    raise FileNotFoundError(
        f"Neither prepared RGB {path} nor source array {npy_path} exists."
    )


def _tiff_rgb(path: Path, channels=(0, 1, 2)) -> Image.Image:
    """Read a multiband TIFF and make a robust percentile-stretched RGB view."""
    import rasterio

    with rasterio.open(path) as source:
        array = source.read([channel + 1 for channel in channels]).astype(np.float32)
    rgb = np.moveaxis(array, 0, -1)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = np.percentile(values[finite], (2, 98))
        if high <= low:
            high = low + 1.0
        output[..., channel] = (np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def _h5_rgb(path: Path, channels=(3, 2, 1)) -> Image.Image:
    """Create the Landslide4Sense RGB view from its HWC HDF5 image."""
    import h5py

    with h5py.File(path, "r") as source:
        array = source["img"][:].astype(np.float32)
    rgb = array[..., list(channels)]
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = np.percentile(values[finite], (2, 98))
        if high <= low:
            high = low + 1.0
        output[..., channel] = (
            np.clip((values - low) / (high - low), 0, 1) * 255
        ).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def _panel(images: list[Image.Image], titles: list[str]) -> Image.Image:
    width, height = images[0].size
    header = 28
    canvas = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(zip(images, titles)):
        image = image.resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 8, 7), title, fill="black")
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize segmentation/CD predictions")
    parser.add_argument("--exp_path", required=True, help="Directory with config.yaml and checkpoint")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", default=None, help="Default: EXP_PATH/visualization/SPLIT")
    parser.add_argument("--limit", type=int, default=None, help="Only render the first N samples")
    parser.add_argument("--sw_batch", type=int, default=None,
                        help="Override sliding-window batch size from config; -1 uses all patches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_path = Path(args.exp_path).resolve()
    config = Config.from_yaml(str(exp_path / "config.yaml"))
    if config.task not in ("cd", "seg"):
        raise ValueError(f"visualize.py supports task=cd or seg, got {config.task!r}")
    config.resume = str(exp_path / args.checkpoint)

    trainer = Trainer(config)
    trainer.model.eval()
    dataset = trainer._build_dataset(args.split)
    # Visualization is intentionally sample-by-sample. Reusing a training
    # config such as L4S (eval_batch_size=32, num_workers=16) needlessly
    # prefetches many large 14-band HDF5 arrays and can exhaust host memory.
    worker_count = min(int(config.get("training", {}).get("num_workers", 0)), 2)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=worker_count,
        pin_memory=True, persistent_workers=worker_count > 0,
    )

    split_cfg = config.get("dataset", {})[args.split]
    dataset_root = Path(split_cfg["dataset_path"])
    cd_data_dir = Path(getattr(dataset, "data_dir", dataset_root))
    image_paths = {
        path.stem: path
        for sample in getattr(dataset, "samples", [])
        for path in [Path(sample[0])]
    }
    output_dir = Path(args.output_dir).resolve() if args.output_dir else exp_path / "visualization" / args.split
    pred_dir = output_dir / "pred"
    error_dir = output_dir / "error"
    panel_dir = output_dir / "panel"
    for directory in (pred_dir, error_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inference_cfg = config.get("inference", {})
    kernel = inference_cfg.get("sliding_kernel")
    stride = inference_cfg.get("sliding_stride", kernel)
    sw_batch = args.sw_batch
    if sw_batch is None:
        sw_batch = inference_cfg.get("sliding_batch_size", -1)
    use_sliding = kernel is not None

    info = DUAL_INPUT_MODELS.get(trainer.model_name, {})
    forward_fn = None
    if info.get("forward") == "dual":
        first, second = info["arg_names"]
        forward_fn = lambda model, pre, post: model(**{first: pre, second: post})

    rendered = 0
    with torch.inference_mode():
        for batch in loader:
            if config.task == "cd":
                pre, post, label, sample_ids = batch
            else:
                image, label, sample_ids = batch
            for batch_index, sample_id in enumerate(sample_ids):
                output_id = Path(sample_id).stem
                if config.task == "cd":
                    if use_sliding:
                        result = sliding_window_inference_cd(
                            trainer.model, pre[batch_index], post[batch_index], kernel, stride,
                            config.num_classes, trainer.device, forward_fn=forward_fn,
                            resolve_fn=trainer._resolve_logits, batch_size=sw_batch,
                            dual_head=trainer._is_dual_head,
                        )
                        pred = result[1] if isinstance(result, tuple) else result
                    else:
                        outputs = trainer.task_handler.run_model(
                            trainer.model, trainer.model_name,
                            (pre[batch_index:batch_index + 1].to(trainer.device),
                             post[batch_index:batch_index + 1].to(trainer.device)),
                        )
                        pred = trainer._resolve_logits(outputs).argmax(1)[0].cpu().numpy()
                else:
                    seg_forward_fn = None
                    if info.get("forward") == "dual":
                        first, second = info["arg_names"]
                        seg_forward_fn = lambda model, value: model(**{first: value, second: value})
                    if use_sliding:
                        pred = sliding_window_inference(
                            trainer.model, image[batch_index], kernel, stride,
                            config.num_classes, trainer.device, forward_fn=seg_forward_fn,
                            resolve_fn=trainer._resolve_logits, batch_size=sw_batch,
                        )
                    else:
                        inputs = image[batch_index:batch_index + 1].to(trainer.device)
                        outputs = trainer.task_handler.run_model(
                            trainer.model, trainer.model_name, inputs,
                        )
                        pred = trainer._resolve_logits(outputs).argmax(1)[0].cpu().numpy()
                target = label[batch_index].numpy()
                error = _error_map(pred, target)

                _save_mask(pred, pred_dir / f"{output_id}.png")
                error.save(error_dir / f"{output_id}.png")
                gt_display = Image.fromarray(
                    np.where(target == 255, 128, target * 255).astype(np.uint8), mode="L"
                ).convert("RGB")
                pred_display = Image.fromarray((pred * 255).astype(np.uint8), mode="L").convert("RGB")
                if config.task == "cd":
                    panel_images = [
                        _rgb_image(cd_data_dir, "T1_vis", sample_id),
                        _rgb_image(cd_data_dir, "T2_vis", sample_id),
                        gt_display, pred_display, error,
                    ]
                    panel_titles = [
                        "T1 (pre)", "T2 (post)", "Ground truth", "Prediction",
                        "TP green / FP red / FN blue",
                    ]
                else:
                    source_path = image_paths.get(sample_id) or image_paths.get(output_id)
                    if source_path is not None:
                        source_image = _tiff_rgb(source_path)
                    elif str(sample_id).lower().endswith((".h5", ".hdf5")):
                        source_image = _h5_rgb(dataset_root / sample_id)
                    else:
                        raise FileNotFoundError(f"Cannot find source image for sample {sample_id}")
                    panel_images = [source_image, gt_display, pred_display, error]
                    panel_titles = [
                        "Image (RGB)", "Ground truth", "Prediction",
                        "TP green / FP red / FN blue",
                    ]
                panel = _panel(panel_images, panel_titles)
                panel.save(panel_dir / f"{output_id}.jpg", quality=92)
                rendered += 1
                print(f"[{rendered}/{len(dataset)}] {sample_id}")
                if args.limit is not None and rendered >= args.limit:
                    break
            if args.limit is not None and rendered >= args.limit:
                break

    print(f"Saved {rendered} samples to {output_dir}")


if __name__ == "__main__":
    main()
