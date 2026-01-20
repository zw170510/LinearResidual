# Configuration for Grouped Linear Mixer with Spectral Normalization
# Constrains ||W||_2 <= 1.0 via power iteration

out_dir = 'out-linmix-grouped-spectral'
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

# Linear Mixer: Grouped with Spectral Normalization
use_linear_mixer = True
linear_mixer_type = 'grouped'
linear_mixer_groups = 8
linear_mixer_gate_cap = None
linear_mixer_ortho_reg = 0.0
linear_mixer_spectral_norm = 1.0    # Constrain ||W||_2 <= 1.0
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

# NOTE: Spectral normalization uses power iteration inside forward()
#       No need for additional loss computation
