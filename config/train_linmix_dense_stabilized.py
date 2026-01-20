# Configuration for Dense Linear Mixer with Full Stabilization
# W is d x d dense matrix - most expressive but requires careful stabilization
# USE ONLY FOR SMALL-SCALE EXPERIMENTS

out_dir = 'out-linmix-dense-stabilized'
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

# Linear Mixer: Dense with Full Stabilization
use_linear_mixer = True
linear_mixer_type = 'dense'         # Full d x d matrix
linear_mixer_gate_cap = 0.25        # Conservative gating
linear_mixer_ortho_reg = 0.01       # Orthogonal regularization
linear_mixer_spectral_norm = 1.0    # Spectral normalization
use_mlp_linear = False

# adamw optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1                 # Strong weight decay on W
beta1 = 0.9
beta2 = 0.95

# learning rate decay
warmup_iters = 200
lr_decay_iters = 20000
min_lr = 6e-5

# system
dtype = 'bfloat16'
compile = True

# WARNING: Dense linear mixer is expensive
#   - d x d dense W: 768 x 768 = 590K params per layer
#   - Computation: O(B * T * d^2) per layer
#   - Requires all three stabilization strategies:
#     1. Gate cap: limits alpha to [0, 0.25]
#     2. Orthogonal regularization: keeps M_eff near-isometric
#     3. Spectral norm: bounds ||W||_2 <= 1.0
#
# RECOMMENDATION: Use dense only at very small scales (n_embd <= 256)
#                 For n_embd=768, prefer grouped or lowrank
