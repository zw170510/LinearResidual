# Configuration for Grouped Linear Mixer with Gate Cap stabilization
# Baseline: alpha = 0.25 * sigmoid(a), standard weight decay

out_dir = 'out-linmix-grouped-gatecap'
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

# Linear Mixer: Grouped with Gate Cap
use_linear_mixer = True
linear_mixer_type = 'grouped'
linear_mixer_groups = 8
linear_mixer_gate_cap = 0.25        # alpha = 0.25 * sigmoid(a), capped to [0, 0.25]
linear_mixer_ortho_reg = 0.0        # No orthogonal regularization
linear_mixer_spectral_norm = None   # No spectral normalization
use_mlp_linear = False

# adamw optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1                 # Standard weight decay helps gate cap
beta1 = 0.9
beta2 = 0.95

# learning rate decay
warmup_iters = 200
lr_decay_iters = 20000
min_lr = 6e-5

# system
dtype = 'bfloat16'
compile = True
