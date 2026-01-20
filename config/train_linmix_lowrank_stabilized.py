# Configuration for Low-Rank Linear Mixer with Gate Cap + Ortho
# Combined stabilization strategies

out_dir = 'out-linmix-lowrank-stabilized'
eval_interval = 500
eval_iters = 100

# data
dataset = 'tinystories'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# Linear Mixer: Low-Rank with Combined Stabilization
use_linear_mixer = True
linear_mixer_type = 'lowrank'
linear_mixer_rank = 192
linear_mixer_gate_cap = 0.5         # alpha = 0.5 * sigmoid(a)
linear_mixer_ortho_reg = 0.01       # lambda_orth = 0.01
linear_mixer_spectral_norm = None
use_mlp_linear = False

# adamw optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# learning rate decay
warmup_iters = 200
lr_decay_iters = 20000
min_lr = 6e-5

# system
dtype = 'bfloat16'
compile = True

# NOTE: Low-rank is inherently more stable, so lighter stabilization needed
#       Gate cap=0.5 is moderate, ortho_reg=0.01 is conservative
#       Combined approach: weight decay + gating + orthogonality constraint
