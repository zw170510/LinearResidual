# Train a GPT model with Linear Mixer (Grouped) on GSM8K dataset
# Good hyperparameters for math problem solving

# I/O
out_dir = 'out-gsm8k-linmix-grouped'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'gsm8k'
gradient_accumulation_steps = 4
batch_size = 16  # GSM8K 樣本較長，使用較小的 batch size
block_size = 512  # GSM8K 問題+答案通常較長

# Model - 使用中等大小的模型
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

# Linear Mixer: Grouped with Spectral Normalization
# 選擇以下三種配置之一：
# 1. Orthogonal Regularization (鼓勵近似正交)
# 2. Spectral Normalization (限制譜範數)
# 3. Gate Cap (限制 alpha 的範圍)

use_linear_mixer = True
linear_mixer_type = 'grouped'
linear_mixer_groups = 8             # 分為 8 組

linear_mixer_spectral_norm = None   # 限制 ||W||_2 <= 1.0
linear_mixer_gate_cap = None
linear_mixer_ortho_reg = 0.0

# 配置選項 1: Spectral Normalization (推薦用於數學推理)
# linear_mixer_spectral_norm = 1.0   # 限制 ||W||_2 <= 1.0
# linear_mixer_gate_cap = None
# linear_mixer_ortho_reg = 0.0

# 配置選項 2: Orthogonal Regularization (取消註釋以使用)
# linear_mixer_spectral_norm = None
# linear_mixer_gate_cap = None
# linear_mixer_ortho_reg = 0.01     # lambda_orth = 0.01

# 配置選項 3: Gate Cap (取消註釋以使用)
# linear_mixer_spectral_norm = None
# linear_mixer_gate_cap = 2.0        # 限制 alpha 在 [-2, 2]
# linear_mixer_ortho_reg = 0.0

use_mlp_linear = False              # MLP 部分不使用 linear mixer

# Optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 1e-1                 # Linear mixer 需要較強的 weight decay
beta1 = 0.9
beta2 = 0.95
lr_decay_iters = 10000
min_lr = 3e-5
warmup_iters = 500

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True

# NOTE: 
# - 如果使用 Orthogonal Regularization，train.py 會自動加入正交損失
# - Spectral Normalization 在 forward() 中自動應用，無需額外損失
# - Gate Cap 在 forward() 中自動限制 alpha 範圍
