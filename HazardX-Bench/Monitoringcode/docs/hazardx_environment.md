# hazardx 训练环境

当前机器的 Conda 环境：`/root/miniconda3/envs/hazardx`。
使用 Python 3.10、PyTorch 2.5.1 + CUDA 11.8、torchvision 0.20.1、NumPy 1.26.4。
系统驱动和系统 CUDA 未被替换；CUDA 编译工具安装在环境内。

## 运行滑坡实验

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hazardx
cd /root/workcsc/HazardX

# 只预览 48 个配置、两个 seed 对应的 96 条训练命令
DRY_RUN=1 bash shfile/train_landslide4sensen_two_seeds.sh

# 正式运行：GPU 0 跑 seed 43，GPU 1 跑 seed 44，两条队列并行
bash shfile/train_landslide4sensen_two_seeds.sh
```

也可以显式选择 Python，不依赖当前激活环境：

```bash
PYTHON_BIN=/root/miniconda3/envs/hazardx/bin/python \
bash /root/workcsc/HazardX/shfile/train_landslide4sensen_two_seeds.sh
```

输出根目录为 `/root/autodl-tmp/output/landslide`；日志与批次汇总位于其 `batch_logs`。

## 兼容性处理

- `pyproject.toml` 原来固定的 PyTorch 1.13.1 低于 SAM2 官方要求。此环境单独使用
  `requirements-hazardx.txt` 和 `constraints-hazardx.txt`；不要在环境中直接执行
  `pip install -e .`，以免重新安装旧版 PyTorch。
- MMCV 使用带算子扩展的 `mmcv-full 1.7.2`，不能同时安装同版本的纯 Python `mmcv`。
- ChangeMamba 的 `selective_scan` 从项目源码编译，包含 core 和 oflex 两个 CUDA 扩展。
  编译副本只生成 V100 对应的 SM70 内核；换成其他架构 GPU 时应重新编译。
- AnySat 源码位于 `/root/workcsc/AnySat`，DOFA 源码位于 `/root/workcsc/DOFA`。
  AnySat commit: `5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a`；
  DOFA commit: `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9`。
- `/root/workcsc/AnyDisasterMapping` 链接到 HazardX；项目的 `pretrained_weight` 链接到
  `/root/autodl-tmp/pretrained_weight`，兼容配置已有的权重路径。
- 数据集包的 `__init__.py` 已清除对已删除 infra 数据集的导入，仍由 registry 按需导入。

该说明记录当前机器的环境和运行方式。依赖清单不替代源码扩展的编译过程。

## 本机验证结果

- pip check：无依赖冲突；全部实验所需模型模块导入通过。
- V100 CUDA 张量运算通过；L4S 14 通道、LANDREF-CD 34 通道真实样本的 U-Net 前向和反向传播通过。
- selective_scan 的 core、oflex 扩展均通过 CUDA 前向和反向传播。
- 批量脚本 DRY_RUN 确认 48 个配置、seed 43/44、共 96 次实验。
- 未执行 96 次完整训练，也未逐一验证每个模型的完整训练流程。

已安装包快照见 `hazardx-installed-packages.txt`，其中本地扩展的构建路径仅用于记录安装来源。

## 双 GPU 调度

脚本默认 `GPU_IDS="0 1"`、`SEEDS="43 44"`，按顺序一一绑定。
每个 GPU 同时最多训练一个模型，两个 GPU 各自顺序遍历全部配置。
每个 worker 设置自己的 CUDA_VISIBLE_DEVICES，进程内看到的 `cuda:0` 对应被分配的物理 GPU。
无需在脚本外单独设置 CUDA_VISIBLE_DEVICES。

每批日志目录包含 `worker_0.log`、`worker_1.log`、每个实验的独立日志，
以及完成后的 `summary.tsv`。`STOP_ON_ERROR=1` 会在任一实验失败时停止两条队列；
默认记录失败并继续。Ctrl+C 或终止主脚本会清理两条训练进程组。

```bash
# 修改 GPU 与 seed（按位置配对）
GPU_IDS="0 1" SEEDS="43 44" bash shfile/train_landslide4sensen_two_seeds.sh
```

## 每个模型训练后自动测试

当前批量脚本仅遍历 L4S。每条 GPU 队列在训练成功后立即执行
`python test.py --exp_path <该模型的实际输出目录> --checkpoint best.pth --force`，
测试结束后再训练下一个模型。训练失败不会测试旧权重；测试失败会记入汇总，
默认继续下一模型，`STOP_ON_ERROR=1` 时停止两条队列。

测试输出 `test_metrics.json` 和 `test_log.txt` 保存在对应实验目录，
批次目录另存 `*_test.log`。summary.tsv 的 stage 列区分 train/test。
Trainer 已避免从保存配置测试时重复追加 seed 目录后缀。
修改仅对之后启动的批次生效，不改变正在运行的旧脚本。

## 已有 checkpoint 的自动分流

每个 L4S 模型和 seed 独立检查其实际输出目录，优先级如下：

1. 存在 best.pth：跳过训练，直接 test.py --checkpoint best.pth --force。
2. 不存在 best.pth，但存在 latest.pth：train.py --resume latest.pth，成功后测试新 best.pth。
3. 两者均不存在：从头训练，成功后测试 best.pth。

latest 恢复由 Trainer 处理，包含模型、优化器、调度器、epoch/iteration 和 best_metric。
已有 best 即使来自未完成训练，也按用户指定规则直接测试。
测试仍需对应的 config.yaml；文件损坏或缺少保存配置时会记录失败，不擅自覆盖权重。
summary.tsv 的 stage 为 train、resume 或 test。DRY_RUN=1 可查看每个实验当前选择的分支。

## L4S 表格汇总

每次测试成功后自动更新 `/root/autodl-tmp/output/landslide/l4s_summary/`：

- `l4s_results.xlsx`：seed42、seed43、seed44 独立 sheet，多级合并表头；每模型 Class0/Class1 两行。
- `l4s_seed<N>.csv` 和 `l4s_seed<N>.md`：相同内容的 CSV 和 Markdown。

列顺序为 model、class、IoU、F1、Precision、Recall、OA、mIoU。表格不含 encoder 和参数量。
指标转换为百分数，保留两位小数。Class0 为非滑坡，Class1 为滑坡；OA、mIoU 为全局指标。
参数量仍作为辅助信息保存在 model_info.json，不纳入结果表。
未生成 test_metrics.json 的项目显示 —，不使用示例数值代替真实结果。

手动更新：

```bash
python scripts/summarize_l4s_results.py
```

自动更新会采用脚本的 OUTPUT_ROOT；可通过 SUMMARY_ROOT 修改表格保存目录。
汇总过程使用文件锁和原子替换，避免两张 GPU 并发覆盖未写完的表格。
新启动的 test.py 才会记录 model_info.json；历史指标未记录参数量时可重新测试生成。
