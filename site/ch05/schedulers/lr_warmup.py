"""
Learning Rate Warmup Implementations

This module provides various learning rate warmup strategies commonly used
in training deep neural networks.
"""

import math
from typing import Optional


class LinearWarmup:
    """
    Linear learning rate warmup.
    Linearly increases learning rate from 0 to base_lr over warmup_steps.
    """
    
    def __init__(self, base_lr: float, warmup_steps: int):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_steps: Number of steps for warmup period
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        return self.base_lr


class ExponentialWarmup:
    """
    Exponential learning rate warmup.
    Exponentially increases learning rate from start_lr to base_lr.
    """
    
    def __init__(self, base_lr: float, warmup_steps: int, start_lr: float = 1e-7):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_steps: Number of steps for warmup period
            start_lr: Initial learning rate (should be very small)
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Exponential interpolation
            factor = (step + 1) / self.warmup_steps
            return self.start_lr * (self.base_lr / self.start_lr) ** factor
        return self.base_lr


class CosineWarmup:
    """
    Cosine learning rate warmup.
    Smoothly increases learning rate following a cosine curve.
    """
    
    def __init__(self, base_lr: float, warmup_steps: int):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_steps: Number of steps for warmup period
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Cosine interpolation from 0 to base_lr
            progress = (step + 1) / self.warmup_steps
            return self.base_lr * (1 - math.cos(progress * math.pi)) / 2
        return self.base_lr


class WarmupWithDecay:
    """
    Combines warmup with learning rate decay.
    Warms up linearly, then applies cosine decay.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        """
        Args:
            base_lr: Peak learning rate after warmup
            warmup_steps: Number of steps for warmup period
            total_steps: Total number of training steps
            min_lr: Minimum learning rate at end of training
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        
        # Cosine decay after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


def plot_warmup_schedules(warmup_steps: int = 1000, total_steps: int = 10000):
    """
    Plot different warmup schedules for visualization.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    base_lr = 1e-3
    
    schedulers = {
        'Linear Warmup': LinearWarmup(base_lr, warmup_steps),
        'Exponential Warmup': ExponentialWarmup(base_lr, warmup_steps),
        'Cosine Warmup': CosineWarmup(base_lr, warmup_steps),
        'Warmup + Cosine Decay': WarmupWithDecay(base_lr, warmup_steps, total_steps)
    }
    
    steps = range(min(total_steps, 5000))
    
    plt.figure(figsize=(12, 6))
    for name, scheduler in schedulers.items():
        lrs = [scheduler.get_lr(step) for step in steps]
        plt.plot(steps, lrs, label=name, linewidth=2)
    
    plt.axvline(x=warmup_steps, color='red', linestyle='--', 
                label=f'Warmup End ({warmup_steps} steps)', alpha=0.5)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Warmup Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/warmup_schedules.png', dpi=150)
    print("Plot saved to warmup_schedules.png")


if __name__ == "__main__":
    # Example usage
    print("Learning Rate Warmup Examples\n")
    
    base_lr = 1e-3
    warmup_steps = 1000
    
    # Test different warmup strategies
    schedulers = {
        'Linear': LinearWarmup(base_lr, warmup_steps),
        'Exponential': ExponentialWarmup(base_lr, warmup_steps),
        'Cosine': CosineWarmup(base_lr, warmup_steps),
    }
    
    test_steps = [0, 250, 500, 750, 1000, 1500]
    
    for name, scheduler in schedulers.items():
        print(f"{name} Warmup:")
        for step in test_steps:
            lr = scheduler.get_lr(step)
            print(f"  Step {step:4d}: lr = {lr:.6f}")
        print()
    
    # Generate visualization
    plot_warmup_schedules()
