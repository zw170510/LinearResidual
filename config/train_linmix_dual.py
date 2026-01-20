# Configuration for Dual Linear Mixers (Attention + MLP)
# Apply linear mixer to both attention and MLP residuals

out_dir = 'out-linmix-dual'
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

# Linear Mixer: Dual (Attention + MLP)
use_linear_mixer = True
linear_mixer_type = 'grouped'
linear_mixer_groups = 8
use_mlp_linear = True               # Enable mixer on MLP residual too
linear_mixer_gate_cap = 0.25
linear_mixer_ortho_reg = 0.01
linear_mixer_spectral_norm = None

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

# NOTE: With use_mlp_linear=True, the forward pass becomes:
#   x' = x + linmix_attn(x) + Attn(LN(x))
#   x'' = x' + linmix_mlp(x') + MLP(LN(x'))
#
# This doubles the number of linear mixers and orthogonality losses
# Monitor total_mixer_loss carefully!
