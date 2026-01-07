# README for TinyStories Dataset Support

这个项目已经添加了对 **TinyStories** 数据集的支持。

## TinyStories 数据集介绍

TinyStories 是一个包含 ~100 万个短故事的数据集，这些故事由 GPT-3.5 生成。
- **数据源**: https://huggingface.co/datasets/roneneldan/TinyStories
- **文件大小**: 相对较小，适合快速迭代和测试

## 使用步骤

### 1. 准备数据集

在 `data/tinystories/` 目录中运行数据准备脚本：

```bash
cd data/tinystories
python prepare.py
```

这个脚本将：
- 从 Hugging Face 下载 TinyStories 数据集
- 使用 GPT-2 tokenizer 进行分词
- 生成 `train.bin` 和 `val.bin` 二进制文件
- 保存 `meta.pkl` 元数据文件（包含词汇表大小）

### 2. 训练模型

有两种方式来训练：

**方式 A: 使用提供的配置文件（推荐）**
```bash
python train.py config/train_tinystories.py
```

**方式 B: 命令行参数指定**
```bash
python train.py --dataset=tinystories --batch_size=12 --max_iters=100000
```

## 配置说明

`config/train_tinystories.py` 包含为 TinyStories 优化的参数：

- **模型大小**: 比 GPT-2 小（6 层，6 注意头，384 嵌入维度）
- **block_size**: 256（因为故事较短）
- **max_iters**: 100,000（足以训练一个好的模型）
- **batch_size**: 12（可根据 GPU 内存调整）

## 调整参数

可以根据你的 GPU 内存和训练时间偏好调整参数：

```bash
# 更小的模型，更快的训练
python train.py --dataset=tinystories --n_layer=4 --n_head=4 --n_embd=256 --batch_size=16

# 更大的模型，更好的效果
python train.py --dataset=tinystories --n_layer=8 --n_head=8 --n_embd=512 --batch_size=8

# 调整学习率和迭代次数
python train.py --dataset=tinystories --learning_rate=8e-4 --max_iters=200000
```

## 多 GPU 训练

使用 DDP 在多个 GPU 上训练：

```bash
# 在 4 个 GPU 上训练
torchrun --standalone --nproc_per_node=4 train.py --dataset=tinystories
```

## 文件结构

```
data/tinystories/
├── prepare.py          # 数据准备脚本
├── train.bin           # 训练集（二进制格式）
├── val.bin             # 验证集（二进制格式）
└── meta.pkl            # 元数据（词汇表大小等）

config/
└── train_tinystories.py # TinyStories 训练配置

out-tinystories/        # 训练输出目录（训练后生成）
├── ckpt.pt             # 模型检查点
├── tokens.txt          # 训练日志
└── ...
```

## 注意事项

1. **首次下载数据集**: 首次运行 `prepare.py` 会下载数据集，可能需要一些时间和网络带宽
2. **词汇表**: TinyStories 使用 GPT-2 的 tokenizer，词汇表大小为 50,257
3. **恢复训练**: 使用 `--init_from=resume` 恢复之前的训练：
   ```bash
   python train.py --dataset=tinystories --init_from=resume
   ```

## 示例训练命令

### 快速测试（小模型）
```bash
python train.py config/train_tinystories.py --max_iters=10000
```

### 完整训练
```bash
python train.py config/train_tinystories.py
```

### 继续训练
```bash
python train.py config/train_tinystories.py --init_from=resume
```
