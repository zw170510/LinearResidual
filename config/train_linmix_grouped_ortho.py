# Configuration for Grouped Linear Mixer with Orthogonal Regularization
# Encourages M_eff = I + alpha*W to be near-isometric

out_dir = 'out-linmix-grouped-ortho'
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

# Linear Mixer: Grouped with Orthogonal Regularization
use_linear_mixer = True
linear_mixer_type = 'grouped'
linear_mixer_groups = 8
linear_mixer_gate_cap = None        # No gating, alpha can grow
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

# NOTE: In train.py, add mixer orthogonal losses to main loss:
#   mixer_losses = model.get_mixer_losses()
#   total_loss = ce_loss + sum(mixer_losses.values())
