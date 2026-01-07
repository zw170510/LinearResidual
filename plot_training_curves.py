"""
Training metrics logger and visualizer.
Records training progress and enables comparison across experiments.
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

def create_metrics_logger(out_dir: str):
    """Create a metrics logger that saves to JSON.
    
    Args:
        out_dir: output directory for checkpoints
    
    Returns:
        MetricsLogger instance
    """
    return MetricsLogger(out_dir)

class MetricsLogger:
    """Log and save training metrics to JSON."""
    
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.metrics_file = os.path.join(out_dir, 'metrics.json')
        self.metrics = {
            'iter': [],
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rate': [],
            'alpha_attn_mean': [],  # For linear mixer
        }
        
        # Load existing metrics if available
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                print(f"Loaded {len(self.metrics['iter'])} existing metrics from {self.metrics_file}")
            except Exception as e:
                print(f"Warning: Could not load existing metrics: {e}")
    
    def log(self, iter_num: int, train_loss: float, val_loss: float, 
            learning_rate: float, alpha_attn_mean: float = None):
        """Log metrics for a training step.
        
        Args:
            iter_num: iteration number
            train_loss: training loss
            val_loss: validation loss
            learning_rate: current learning rate
            alpha_attn_mean: mean alpha value (optional, for linear mixer)
        """
        self.metrics['iter'].append(iter_num)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_ppl'].append(math.exp(train_loss))
        self.metrics['val_ppl'].append(math.exp(val_loss))
        self.metrics['learning_rate'].append(learning_rate)
        
        if alpha_attn_mean is not None:
            self.metrics['alpha_attn_mean'].append(alpha_attn_mean)
        else:
            self.metrics['alpha_attn_mean'].append(None)
    
    def save(self):
        """Save metrics to JSON file."""
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> Dict:
        """Get all logged metrics."""
        return self.metrics.copy()

def load_metrics(out_dir: str) -> Dict:
    """Load metrics from a training run.
    
    Args:
        out_dir: output directory containing metrics.json
    
    Returns:
        Dictionary with metrics
    """
    metrics_file = os.path.join(out_dir, 'metrics.json')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_comparison(baseline_dir: str = 'out-baseline', 
                   linmix_dir: str = 'out-linmix-option1',
                   output_file: str = 'comparison.png'):
    """Plot comparison between baseline and linear mixer training.
    
    Args:
        baseline_dir: directory with baseline training
        linmix_dir: directory with linear mixer training
        output_file: output file for the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    # Load metrics
    baseline_metrics = load_metrics(baseline_dir)
    linmix_metrics = load_metrics(linmix_dir)
    
    if baseline_metrics is None or linmix_metrics is None:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Comparison: Baseline vs Linear Mixer', fontsize=16)
    
    # Plot 1: Validation Loss
    ax = axes[0, 0]
    if baseline_metrics['iter']:
        ax.plot(baseline_metrics['iter'], baseline_metrics['val_loss'], 
               label='Baseline', linewidth=2, marker='o', markersize=4)
    if linmix_metrics['iter']:
        ax.plot(linmix_metrics['iter'], linmix_metrics['val_loss'], 
               label='Linear Mixer', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation PPL
    ax = axes[0, 1]
    if baseline_metrics['iter']:
        ax.plot(baseline_metrics['iter'], baseline_metrics['val_ppl'], 
               label='Baseline', linewidth=2, marker='o', markersize=4)
    if linmix_metrics['iter']:
        ax.plot(linmix_metrics['iter'], linmix_metrics['val_ppl'], 
               label='Linear Mixer', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation PPL')
    ax.set_title('Validation Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    ax = axes[1, 0]
    if baseline_metrics['iter']:
        ax.plot(baseline_metrics['iter'], baseline_metrics['train_loss'], 
               label='Baseline', linewidth=2, alpha=0.7)
    if linmix_metrics['iter']:
        ax.plot(linmix_metrics['iter'], linmix_metrics['train_loss'], 
               label='Linear Mixer', linewidth=2, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Alpha values (if available)
    ax = axes[1, 1]
    if linmix_metrics.get('alpha_attn_mean'):
        alpha_vals = [a for a in linmix_metrics['alpha_attn_mean'] if a is not None]
        if alpha_vals:
            alpha_iters = [linmix_metrics['iter'][i] for i, a in enumerate(linmix_metrics['alpha_attn_mean']) if a is not None]
            ax.plot(alpha_iters, alpha_vals, 'g-o', linewidth=2, markersize=4)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean α (Attention)')
            ax.set_title('Linear Mixer - Alpha Learning')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No alpha values recorded', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No linear mixer data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")
    plt.close()

def print_convergence_analysis(baseline_dir: str = 'out-baseline',
                              linmix_dir: str = 'out-linmix-option1'):
    """Analyze and print convergence statistics.
    
    Args:
        baseline_dir: directory with baseline training
        linmix_dir: directory with linear mixer training
    """
    baseline_metrics = load_metrics(baseline_dir)
    linmix_metrics = load_metrics(linmix_dir)
    
    if baseline_metrics is None or linmix_metrics is None:
        return
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    # Convergence speed: iterations to reach certain PPL thresholds
    thresholds = [50, 30, 20, 15]
    
    print("\nIterations to reach PPL threshold:")
    print("-" * 80)
    print(f"{'PPL':<10} {'Baseline':<20} {'Linear Mixer':<20} {'Speedup':<15}")
    print("-" * 80)
    
    for threshold in thresholds:
        baseline_iter = None
        linmix_iter = None
        
        # Find first iteration where PPL < threshold
        for i, ppl in enumerate(baseline_metrics['val_ppl']):
            if ppl < threshold:
                baseline_iter = baseline_metrics['iter'][i]
                break
        
        for i, ppl in enumerate(linmix_metrics['val_ppl']):
            if ppl < threshold:
                linmix_iter = linmix_metrics['iter'][i]
                break
        
        baseline_str = f"{baseline_iter}" if baseline_iter else "Not reached"
        linmix_str = f"{linmix_iter}" if linmix_iter else "Not reached"
        
        if baseline_iter and linmix_iter:
            speedup = baseline_iter / linmix_iter
            speedup_str = f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower"
        else:
            speedup_str = "N/A"
        
        print(f"{threshold:<10} {baseline_str:<20} {linmix_str:<20} {speedup_str:<15}")
    
    # Stability analysis: variance of last N checkpoints
    print("\n\nStability Analysis (variance of last 5 checkpoints):")
    print("-" * 80)
    
    n_recent = min(5, len(baseline_metrics['val_ppl']))
    
    baseline_recent = baseline_metrics['val_ppl'][-n_recent:]
    linmix_recent = linmix_metrics['val_ppl'][-n_recent:]
    
    baseline_var = np.var(baseline_recent) if baseline_recent else 0
    linmix_var = np.var(linmix_recent) if linmix_recent else 0
    
    print(f"Baseline variance:   {baseline_var:.6f}")
    print(f"Linear Mixer variance: {linmix_var:.6f}")
    print(f"Stability improvement: {(1 - linmix_var/baseline_var)*100:+.2f}%" if baseline_var > 0 else "N/A")
    
    # Final performance
    print("\n\nFinal Performance:")
    print("-" * 80)
    final_baseline_ppl = baseline_metrics['val_ppl'][-1] if baseline_metrics['val_ppl'] else None
    final_linmix_ppl = linmix_metrics['val_ppl'][-1] if linmix_metrics['val_ppl'] else None
    
    if final_baseline_ppl and final_linmix_ppl:
        improvement = (final_baseline_ppl - final_linmix_ppl) / final_baseline_ppl * 100
        print(f"Baseline final PPL:     {final_baseline_ppl:.2f}")
        print(f"Linear Mixer final PPL: {final_linmix_ppl:.2f}")
        print(f"Improvement:            {improvement:+.2f}%")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Analyze and visualize training metrics')
    parser.add_argument('--baseline_dir', type=str, default='out-baseline',
                        help='Baseline output directory')
    parser.add_argument('--linmix_dir', type=str, default='out-linmix-option1',
                        help='Linear mixer output directory')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plot')
    parser.add_argument('--analyze', action='store_true', help='Print convergence analysis')
    parser.add_argument('--output', type=str, default='comparison.png',
                        help='Output plot file')
    
    args = parser.parse_args()
    
    if args.analyze:
        print_convergence_analysis(args.baseline_dir, args.linmix_dir)
    
    if args.plot:
        plot_comparison(args.baseline_dir, args.linmix_dir, args.output)
    
    if not args.plot and not args.analyze:
        # Default: do both
        print_convergence_analysis(args.baseline_dir, args.linmix_dir)
        plot_comparison(args.baseline_dir, args.linmix_dir, args.output)
