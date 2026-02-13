"""
Cyclical Learning Rate Implementations

This module provides various cyclical learning rate strategies for training
neural networks, including the original CLR paper implementations and
the 1cycle policy.
"""

import math
from typing import Optional, Literal


class CyclicLR:
    """
    Cyclical Learning Rate (CLR) as proposed by Leslie Smith.
    
    Reference: "Cyclical Learning Rates for Training Neural Networks"
    https://arxiv.org/abs/1506.01186
    """
    
    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: Literal['triangular', 'triangular2', 'exp_range'] = 'triangular',
        gamma: float = 0.99994
    ):
        """
        Args:
            base_lr: Lower boundary of learning rate
            max_lr: Upper boundary of learning rate
            step_size: Number of training steps in half a cycle
            mode: One of {triangular, triangular2, exp_range}
            gamma: Decay constant for exp_range mode
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** step
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        return lr


class OneCycleLR:
    """
    1cycle learning rate policy by Leslie Smith.
    
    Consists of:
    1. Warmup: lr increases from initial_lr to max_lr
    2. Annealing: lr decreases from max_lr to final_lr
    
    Reference: "Super-Convergence: Very Fast Training of Neural Networks"
    https://arxiv.org/abs/1708.07120
    """
    
    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: Literal['cos', 'linear'] = 'cos'
    ):
        """
        Args:
            max_lr: Peak learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing lr
            div_factor: Initial lr = max_lr / div_factor
            final_div_factor: Final lr = max_lr / final_div_factor
            anneal_strategy: 'cos' or 'linear' annealing
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.anneal_strategy = anneal_strategy
        
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.step_up:
            # Increasing phase
            progress = step / self.step_up
            return self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (step - self.step_up) / self.step_down
            
            if self.anneal_strategy == 'cos':
                # Cosine annealing
                cos_out = math.cos(math.pi * progress)
                return self.final_lr + (self.max_lr - self.final_lr) * (1 + cos_out) / 2
            else:
                # Linear annealing
                return self.max_lr - (self.max_lr - self.final_lr) * progress


class CosineAnnealingWarmRestarts:
    """
    Cosine Annealing with Warm Restarts (SGDR).
    
    Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    https://arxiv.org/abs/1608.03983
    """
    
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        t_0: int,
        t_mult: int = 2
    ):
        """
        Args:
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            t_0: Number of steps until first restart
            t_mult: Factor to increase cycle length after each restart
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.t_0 = t_0
        self.t_mult = t_mult
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        # Find which cycle we're in
        t_cur = step
        t_i = self.t_0
        cycle = 0
        
        while t_cur >= t_i:
            t_cur -= t_i
            t_i *= self.t_mult
            cycle += 1
        
        # Cosine annealing within current cycle
        progress = t_cur / t_i
        lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        return lr


class ExponentialCyclicLR:
    """
    Exponential cyclical learning rate.
    Learning rate cycles between base_lr and max_lr with exponential curves.
    """
    
    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        cycle_length: int,
        decay_rate: float = 0.96
    ):
        """
        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            cycle_length: Number of steps per complete cycle
            decay_rate: Factor to reduce max_lr after each cycle
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.decay_rate = decay_rate
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        cycle = step // self.cycle_length
        step_in_cycle = step % self.cycle_length
        
        # Reduce max_lr over time
        current_max_lr = self.max_lr * (self.decay_rate ** cycle)
        
        # Exponential interpolation within cycle
        progress = step_in_cycle / self.cycle_length
        
        if progress < 0.5:
            # Increasing phase (exponential)
            phase_progress = progress * 2
            lr = self.base_lr + (current_max_lr - self.base_lr) * (math.exp(phase_progress) - 1) / (math.e - 1)
        else:
            # Decreasing phase (exponential)
            phase_progress = (progress - 0.5) * 2
            lr = current_max_lr - (current_max_lr - self.base_lr) * (math.exp(phase_progress) - 1) / (math.e - 1)
        
        return lr


def plot_cyclical_schedules(total_steps: int = 10000):
    """
    Plot different cyclical learning rate schedules.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    schedulers = {
        'Triangular CLR': CyclicLR(1e-4, 1e-3, step_size=1000, mode='triangular'),
        'Triangular2 CLR': CyclicLR(1e-4, 1e-3, step_size=1000, mode='triangular2'),
        '1cycle': OneCycleLR(1e-3, total_steps, pct_start=0.3),
        'Cosine Warm Restarts': CosineAnnealingWarmRestarts(1e-3, 1e-5, t_0=2000),
    }
    
    steps = range(total_steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, scheduler) in enumerate(schedulers.items()):
        lrs = [scheduler.get_lr(step) for step in steps]
        axes[idx].plot(steps, lrs, linewidth=2, color=f'C{idx}')
        axes[idx].set_xlabel('Training Step')
        axes[idx].set_ylabel('Learning Rate')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/cyclical_schedules.png', dpi=150)
    print("Plot saved to cyclical_schedules.png")


if __name__ == "__main__":
    print("Cyclical Learning Rate Examples\n")
    
    # Test different strategies
    total_steps = 10000
    test_steps = [0, 1000, 2000, 5000, 7500, 10000]
    
    schedulers = {
        'Triangular CLR': CyclicLR(1e-4, 1e-3, step_size=1000),
        '1cycle': OneCycleLR(1e-3, total_steps),
        'SGDR': CosineAnnealingWarmRestarts(1e-3, 1e-5, t_0=2000),
    }
    
    for name, scheduler in schedulers.items():
        print(f"{name}:")
        for step in test_steps:
            if step <= total_steps:
                lr = scheduler.get_lr(step)
                print(f"  Step {step:5d}: lr = {lr:.6e}")
        print()
    
    # Generate visualization
    plot_cyclical_schedules()
