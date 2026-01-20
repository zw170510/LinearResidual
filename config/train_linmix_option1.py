# Configuration for Option 1: Linear + Nonlinear Decomposition (Research Proposal)
# Gated Grouped Linear Mixer in Attention Residual

# I/O
out_dir = 'out-linmix-option1-16'
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

# Linear + Nonlinear Decomposition (Option 1)
use_linear_mixer = True              # Enable linear mixer
linear_mixer_groups = 16              # Block-diagonal groups
use_mlp_linear = False               # Don't add linear mixer to MLP (yet)

# adamw optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# learning rate decay settings
warmup_iters = 200
lr_decay_iters = 20000
min_lr = 6e-5

# system
dtype = 'bfloat16'
compile = True
