# Training commands

Run from Forecastingcode after installation, official initialization setup, and data preparation. Replace visible GPU IDs with available devices while retaining the rank count. Data/output path overrides are documented in README.md. Each group below has an independent static master port. No command in this file was used to start complete retraining during release preparation.

Use the PDE environment for ID06 and the main environment for the other models. Append --resume /path/to/last.pt to continue an interrupted run with its original cursor. Published best.pth is the best-validation snapshot.

## FloodCastBench / 01 UNet (seed 42; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29640 train.py --config config/FloodCastBench/unet.yaml
```

## FloodCastBench / 02 ConvLSTM-Scratch (seed 42; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29641 train.py --config config/FloodCastBench/convlstm.yaml
```

## FloodCastBench / 04 FNO (seed 42; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29642 train.py --config config/FloodCastBench/fno.yaml
```

## FloodCastBench / 05 EarthFormer-Scratch-fp32-lr3e-4 (seed 42; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29643 train.py --config config/FloodCastBench/earthformer.yaml
```

## FloodCastBench / 06 PDE-T-SC-S-PT-FT (seed 42; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29644 train.py --config config/FloodCastBench/pde_transformer.yaml
```

## FloodCastBench / 07 DPOT-S-PT-FT (seed 42; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29645 train.py --config config/FloodCastBench/dpot.yaml
```

## FloodCastBench / 08 Poseidon-T-PT-FT (seed 42; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29646 train.py --config config/FloodCastBench/poseidon.yaml
```

## FloodCastBench / 09 GPhyT-S-PT-FT (seed 42; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29647 train.py --config config/FloodCastBench/gphyt.yaml
```

## FloodCastBench / 11 Prithvi-EO2-300M-PT-FT (seed 42; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29648 train.py --config config/FloodCastBench/prithvi.yaml
```

## FloodSimBench / 01 UNet (seed 1; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29649 train.py --config config/FloodSimBench/unet.yaml
```

## FloodSimBench / 04 FNO (seed 1; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29650 train.py --config config/FloodSimBench/fno.yaml
```

## FloodSimBench / 02 ConvLSTM-Scratch (seed 1; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29651 train.py --config config/FloodSimBench/convlstm.yaml
```

## FloodSimBench / 05 EarthFormer-Scratch (seed 1; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29652 train.py --config config/FloodSimBench/earthformer.yaml
```

## FloodSimBench / 06 PDE-T-SC-S-PT-FT (seed 1; 4 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29653 train.py --config config/FloodSimBench/pde_transformer.yaml
```

## FloodSimBench / 07 DPOT-S-PT-FT (seed 1; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29654 train.py --config config/FloodSimBench/dpot.yaml
```

## FloodSimBench / 08 Poseidon-T-PT-FT (seed 1; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29655 train.py --config config/FloodSimBench/poseidon.yaml
```

## FloodSimBench / 09 GPhyT-S-PT-FT (seed 1; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29656 train.py --config config/FloodSimBench/gphyt.yaml
```

## FloodSimBench / 11 Prithvi-EO2-300M-PT-FT (seed 1; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29657 train.py --config config/FloodSimBench/prithvi.yaml
```

## LandslideCast / 02 ConvLSTM-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29658 train.py --config config/LandslideCast/convlstm.yaml
```

## LandslideCast / 04 FNO3D-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29659 train.py --config config/LandslideCast/fno.yaml
```

## LandslideCast / 06 PDE-T-SC-S-PT-FT (seed 0; 2 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29660 train.py --config config/LandslideCast/pde_transformer.yaml
```

## LandslideCast / 08 Poseidon-T-PT-FT (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29661 train.py --config config/LandslideCast/poseidon.yaml
```

## LandslideCast / 01 UNet3D-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29662 train.py --config config/LandslideCast/unet.yaml
```

## LandslideCast / 05 EarthFormer-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29663 train.py --config config/LandslideCast/earthformer.yaml
```

## LandslideCast / 07 DPOT-S-PT-FT (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29664 train.py --config config/LandslideCast/dpot.yaml
```

## LandslideCast / 11 Prithvi-EO2-300M-PT-FT (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29665 train.py --config config/LandslideCast/prithvi.yaml
```

## LandslideCast / 09 GPhyT-S-PT-FT-Euler (seed 0; 4 ranks)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29666 train.py --config config/LandslideCast/gphyt.yaml
```

## WildfireSpreadTS / 01 UNet2D-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29667 train.py --config config/WildfireSpreadTS/unet.yaml
```

## WildfireSpreadTS / 03 UTAE-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29668 train.py --config config/WildfireSpreadTS/utae.yaml
```

## WildfireSpreadTS / 02 ConvLSTM-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29669 train.py --config config/WildfireSpreadTS/convlstm.yaml
```

## WildfireSpreadTS / 04 FNO2D-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29670 train.py --config config/WildfireSpreadTS/fno.yaml
```

## WildfireSpreadTS / 05 EarthFormer-Scratch (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29671 train.py --config config/WildfireSpreadTS/earthformer.yaml
```

## WildfireSpreadTS / 06 PDE-T-SC-S-PT-FT (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29672 train.py --config config/WildfireSpreadTS/pde_transformer.yaml
```

## WildfireSpreadTS / 07 DPOT-S-PT-FT (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29673 train.py --config config/WildfireSpreadTS/dpot.yaml
```

## WildfireSpreadTS / 10 VideoMAEv2-PT-FT (seed 0; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29674 train.py --config config/WildfireSpreadTS/videomaev2.yaml
```

## WildfireSpreadTS / 11 Prithvi-EO2-300M-PT-FT (seed 1; 1 rank)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=static --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29675 train.py --config config/WildfireSpreadTS/prithvi.yaml
```

