"""
Diagnostic tool for analyzing linear mixer statistics during training.
Monitor alpha values, branch magnitudes, and other metrics.
"""

import os
import torch
import pickle
from pathlib import Path

def extract_linear_mixer_stats(model):
    """Extract statistics about linear mixer parameters and activations.
    
    Returns:
        dict with alpha values and layer info
    """
    stats = {
        'alpha_attn': [],
        'alpha_mlp': [],
        'layer_info': []
    }
    
    for layer_idx, block in enumerate(model.transformer.h):
        if hasattr(block, 'linmix_attn'):
            alpha_val = block.linmix_attn.alpha.item()
            stats['alpha_attn'].append(alpha_val)
            
            # Get weight norm based on linear mixer type
            linmix = block.linmix_attn
            if hasattr(linmix, 'W'):  # GatedGroupedLinear
                w_norm = linmix.W.norm().item()
            elif hasattr(linmix, 'U'):  # GatedLowRankLinear
                # For low-rank: W = U V^T, compute Frobenius norm
                W = linmix.U @ linmix.V.T
                w_norm = W.norm().item()
            else:
                w_norm = 0.0
            
            stats['layer_info'].append({
                'layer': layer_idx,
                'has_linmix_attn': True,
                'alpha_attn': alpha_val,
                'w_norm': w_norm,
            })
        
        if hasattr(block, 'linmix_mlp'):
            alpha_val = block.linmix_mlp.alpha.item()
            stats['alpha_mlp'].append(alpha_val)
    
    return stats

def print_linear_mixer_report(model, iter_num, metrics=None):
    """Print a readable report of linear mixer statistics.
    
    Args:
        model: GPT model
        iter_num: current iteration
        metrics: dict with additional metrics (loss, ppl, etc.)
    """
    stats = extract_linear_mixer_stats(model)
    
    print("\n" + "="*80)
    print(f"LINEAR MIXER DIAGNOSTICS (iter {iter_num})")
    print("="*80)
    
    if metrics:
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
    
    if stats['alpha_attn']:
        print(f"\nAttention Linear Mixer (αₐₜₜₙ):")
        print(f"  Mean: {sum(stats['alpha_attn'])/len(stats['alpha_attn']):.6f}")
        print(f"  Min:  {min(stats['alpha_attn']):.6f}")
        print(f"  Max:  {max(stats['alpha_attn']):.6f}")
        print(f"  Per-layer:")
        for info in stats['layer_info']:
            if 'alpha_attn' in info:
                print(f"    Layer {info['layer']:2d}: α={info['alpha_attn']:8.6f}, "
                      f"||W||={info['w_norm']:.4f}")
    
    if stats['alpha_mlp']:
        print(f"\nMLP Linear Mixer (αₘₗₚ):")
        print(f"  Mean: {sum(stats['alpha_mlp'])/len(stats['alpha_mlp']):.6f}")
        print(f"  Min:  {min(stats['alpha_mlp']):.6f}")
        print(f"  Max:  {max(stats['alpha_mlp']):.6f}")
    
    print("="*80 + "\n")

def save_linear_mixer_checkpoint(model, checkpoint, save_path):
    """Augment checkpoint with linear mixer diagnostic info.
    
    Args:
        model: GPT model
        checkpoint: checkpoint dict to augment
        save_path: path where checkpoint will be saved
    """
    stats = extract_linear_mixer_stats(model)
    checkpoint['linear_mixer_stats'] = stats
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint with linear mixer stats to {save_path}")

def load_and_analyze_checkpoint(ckpt_path):
    """Load checkpoint and analyze linear mixer evolution.
    
    Args:
        ckpt_path: path to checkpoint file
    """
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'linear_mixer_stats' in ckpt:
        stats = ckpt['linear_mixer_stats']
        print(f"\nLinear Mixer Stats at iter {ckpt.get('iter_num', '?')}:")
        print(f"  Attn α values: {[f'{a:.6f}' for a in stats.get('alpha_attn', [])]}")
        if stats.get('alpha_mlp'):
            print(f"  MLP α values:  {[f'{a:.6f}' for a in stats['alpha_mlp']]}")
    else:
        print("No linear mixer stats found in checkpoint")
    
    print(f"  Validation loss: {ckpt.get('best_val_loss', '?'):.4f}")

def compare_baseline_and_linmix(baseline_ckpt, linmix_ckpt):
    """Compare learning curves between baseline and linear mixer versions.
    
    Args:
        baseline_ckpt: path to baseline checkpoint
        linmix_ckpt: path to linear mixer checkpoint
    """
    import math
    
    print("\nComparing Baseline vs Linear Mixer...")
    print("="*80)
    
    baseline_loss = None
    baseline_ppl = None
    linmix_loss = None
    linmix_ppl = None
    
    if os.path.exists(baseline_ckpt):
        ckpt_b = torch.load(baseline_ckpt, map_location='cpu')
        baseline_loss = ckpt_b.get('best_val_loss', None)
        if baseline_loss is not None:
            baseline_ppl = math.exp(baseline_loss)
        print(f"Baseline   - Iter: {ckpt_b.get('iter_num', '?'):6d}")
        print(f"             Val Loss: {baseline_loss:.4f}" if baseline_loss else "             Val Loss: ?")
        print(f"             PPL:      {baseline_ppl:.4f}" if baseline_ppl else "             PPL:      ?")
    
    if os.path.exists(linmix_ckpt):
        ckpt_l = torch.load(linmix_ckpt, map_location='cpu')
        linmix_loss = ckpt_l.get('best_val_loss', None)
        if linmix_loss is not None:
            linmix_ppl = math.exp(linmix_loss)
        print(f"\nLinMix     - Iter: {ckpt_l.get('iter_num', '?'):6d}")
        print(f"             Val Loss: {linmix_loss:.4f}" if linmix_loss else "             Val Loss: ?")
        print(f"             PPL:      {linmix_ppl:.4f}" if linmix_ppl else "             PPL:      ?")
        
        if 'linear_mixer_stats' in ckpt_l:
            stats = ckpt_l['linear_mixer_stats']
            if stats['alpha_attn']:
                mean_alpha = sum(stats['alpha_attn']) / len(stats['alpha_attn'])
                print(f"             Mean α_attn: {mean_alpha:.6f} "
                      f"(learned: {mean_alpha > 1e-4})")
    
    # Calculate improvement metrics
    if baseline_loss is not None and linmix_loss is not None:
        loss_improvement = (baseline_loss - linmix_loss) / baseline_loss * 100
        print("\n" + "="*80)
        print("IMPROVEMENT METRICS:")
        print(f"  Loss:      {loss_improvement:+.2f}% {'✓' if loss_improvement > 0 else '✗'}")
        
        if baseline_ppl is not None and linmix_ppl is not None:
            ppl_improvement = (baseline_ppl - linmix_ppl) / baseline_ppl * 100
            ppl_ratio = linmix_ppl / baseline_ppl
            print(f"  PPL:       {ppl_improvement:+.2f}% {'✓' if ppl_improvement > 0 else '✗'}")
            print(f"  PPL Ratio: {ppl_ratio:.4f}x (lower is better)")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze linear mixer statistics')
    parser.add_argument('--baseline_ckpt', type=str, default='out/ckpt.pt',
                        help='Baseline checkpoint path')
    parser.add_argument('--linmix_ckpt', type=str, default='out-linmix-option1/ckpt.pt',
                        help='Linear mixer checkpoint path')
    parser.add_argument('--compare', action='store_true',
                        help='Compare baseline and linear mixer')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_baseline_and_linmix(args.baseline_ckpt, args.linmix_ckpt)
    else:
        load_and_analyze_checkpoint(args.linmix_ckpt)
