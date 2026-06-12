# 知眠 / Zhimian

面向中国研究生人工智能创新大赛（AI+X 赛道）的“知眠”智能枕睡眠监测系统机器学习代码仓骨架。

当前版本目标：

- 提供可直接运行的 PyTorch 训练骨架
- 所有超参、路径、数据源均通过 YAML 管理
- 支持 `pretrain / finetune / adapt` 三类 Stage 独立启动
- 固定随机种子、保存 best checkpoint 和对应 config、记录 tensorboard 日志
- 在真实数据接入前，使用合成数据验证工程链路、张量形状与标签分布

## 目录结构

```text
zhimian/
  configs/
  data/
    raw/
    processed/
  src/
    datasets/
    preprocessing/
    models/
    training/
    eval/
    utils/
  tests/
```

## 快速开始

在 `zhimian/` 目录下运行：

```bash
python -m src.training.pretrain --config configs/pretrain.yaml
python -m src.training.finetune --config configs/finetune.yaml
python -m src.training.adapt --config configs/adapt.yaml
pytest
```

TensorBoard 日志默认输出到配置文件指定的 `run.output_root` 下：

```bash
tensorboard --logdir outputs
```

## 代码规范

- Python 3.11
- PyTorch
- OmegaConf
- black
- ruff
- pytest

## Assumptions

- 当前仓库中尚未接入 Stage 0 的真实原始数据文件，因此本骨架默认使用合成数据集 `synthetic_sleep` 验证训练、日志、checkpoint 与测试链路。
- 真实数据的目录结构、文件格式、标签映射在收到 Stage 0 后，必须先读取真实文件并报告观测到的结构，再实现对应的 dataset adapter；在此之前不会臆造真实数据格式。
- 当前温度模态在骨架中表示为时间序列形式的 `seq_len x 9`，用于承接 3×3 DS18B20 阵列；若真实采样格式不同，后续以真实文件为准调整。
- 当前 `pretrain / finetune / adapt` 三个 Stage 共享一套最小可运行网络定义，后续会随比赛方案迭代替换为更贴近任务的模型。
- 已观测到的数据包 `data/raw/simulated_sleep_night_u001` 是**合成测试数据**，不是硬件真实采集数据；目前仅用于 adapter、pipeline、训练接口和日志链路验证。
- 对 `simulated_sleep_night_u001` 的标签对齐规则采用：仅使用 `state_snapshots.csv` 中**精确时间戳命中** `physiology_stream.csv` 的样本作为监督样本，不对中间时刻做插值标签。
- `simulated_sleep_night_u001` 中仅提供标量 `skin_temp_c`，没有真实 3×3 DS18B20 温度阵列；当前 adapter 会把单个温度值在时间窗口内复制成 `seq_len x 9` 占位输入，以保持模型输入接口稳定。
