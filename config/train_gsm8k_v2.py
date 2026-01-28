# Train a GPT model with Linear Mixer (Grouped) on GSM8K dataset
# Anti-overfitting configuration based on validation PPL analysis

# I/O
out_dir = 'out-gsm8k-lowrank'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# Data - 增加數據多樣性
dataset = 'gsm8k'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 512

# Model - 減少模型容量以降低 overfitting
n_layer = 6              # 從 8 減少到 6
n_head = 8
n_embd = 384             # 從 512 減少到 384
dropout = 0.2            # 從 0.1 增加到 0.2 (更強的正則化)

# Linear Mixer: Grouped with stronger regularization
use_linear_mixer = True
linear_mixer_type = 'lowrank'
linear_mixer_groups = 8
linear_mixer_rank = 96

# 使用 Spectral Normalization + Gate Cap 雙重約束
linear_mixer_spectral_norm = None   # 限制譜範數
linear_mixer_gate_cap = None        # 限制 alpha 範圍 (新增)
linear_mixer_ortho_reg = 0.0
# linear_mixer_spectral_norm = 1.0   # 限制譜範數
# linear_mixer_gate_cap = 1.5        # 限制 alpha 範圍 (新增)
# linear_mixer_ortho_reg = 0.0

use_mlp_linear = False

# Optimizer - 更保守的設定
learning_rate = 2e-4     # 從 3e-4 降低到 2e-4
max_iters = 5000         # 從 10000 減少到 5000 (提早停止)
weight_decay = 2e-1      # 從 1e-1 增加到 2e-1 (更強的 L2 正則)
beta1 = 0.9
beta2 = 0.95
lr_decay_iters = 5000
min_lr = 2e-5
warmup_iters = 300       # 從 500 減少到 300

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True

# 建議：觀察到 val PPL 在 iter ~1000 (PPL=12.66) 時最低
# 可以考慮在 PPL 開始上升時提早停止訓練
