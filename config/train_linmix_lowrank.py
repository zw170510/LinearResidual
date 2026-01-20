# Configuration for Low-Rank Linear Mixer
# W = U V^T with rank r << d, efficient and expressive

out_dir = 'out-linmix-lowrank'
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

# Linear Mixer: Low-Rank Parameterization
use_linear_mixer = True
linear_mixer_type = 'lowrank'
linear_mixer_rank = 192             # rank r = d/4 = 768/4 = 192
linear_mixer_gate_cap = None
linear_mixer_ortho_reg = 0.0
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

# NOTE: Low-rank factorization W = U V^T
#       - Parameter count: 2*d*r instead of d^2
#       - For rank=192: 2*768*192 = 295K params vs 590K for dense
#       - Less regularization needed due to capacity constraint
