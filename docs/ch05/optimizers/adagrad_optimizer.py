"""
AdaGrad (Adaptive Gradient) Optimizer
======================================

AdaGrad adapts the learning rate for each parameter based on the historical
sum of squared gradients. Parameters with large gradients get smaller learning
rates, while parameters with small gradients get larger learning rates.

Key features:
- Accumulates squared gradients over all time steps
- Automatically decreases learning rate for frequently updated parameters
- Good for sparse gradients (e.g., NLP, recommender systems)
- Learning rate can become too small (monotonically decreasing)

Paper: "Adaptive Subgradient Methods for Online Learning" by Duchi et al. (2011)
"""

import numpy as np


class AdaGrad:
    """
    AdaGrad optimizer implementation.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Initial learning rate (global step size)
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        # State variables
        self.cache = {}  # Accumulated sum of squared gradients
    
    def update(self, params, grads):
        """
        Update parameters using AdaGrad algorithm.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to update
        grads : dict
            Dictionary of gradients for each parameter
        
        Returns:
        --------
        dict : Updated parameters
        """
        updated_params = {}
        
        for key in params.keys():
            # Initialize cache if not exists
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Accumulate squared gradients
            self.cache[key] += grads[key] ** 2
            
            # Update parameters
            # Divide learning rate by the square root of accumulated squared gradients
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return updated_params


def demo_adagrad():
    """
    Demonstration of AdaGrad optimizer on a simple quadratic function.
    Minimize f(x, y) = x^2 + y^2
    """
    print("=" * 60)
    print("AdaGrad Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x^2 + y^2")
    print()
    
    # Initialize parameters
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizer
    optimizer = AdaGrad(learning_rate=1.0)  # AdaGrad can use higher initial LR
    
    # Optimization loop
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'f(x,y)':<12}")
    print("-" * 60)
    
    for i in range(50):
        # Compute gradients: df/dx = 2x, df/dy = 2y
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y']
        }
        
        # Update parameters
        params = optimizer.update(params, grads)
        
        # Compute function value
        f_val = params['x']**2 + params['y']**2
        
        if i % 10 == 0:
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} {f_val[0]:<12.6f}")
    
    print()
    print(f"Final values: x = {params['x'][0]:.8f}, y = {params['y'][0]:.8f}")
    print(f"Function value: f(x,y) = {f_val[0]:.8f}")
    print()


def demo_sparse_gradients():
    """
    Demonstrate AdaGrad's advantage with sparse gradients.
    Simulates a scenario where some parameters are updated infrequently.
    """
    print("=" * 60)
    print("AdaGrad with Sparse Gradients")
    print("=" * 60)
    print("Parameters x, y, z where z is rarely updated (sparse)")
    print()
    
    # Initialize parameters
    params = {
        'x': np.array([5.0]),
        'y': np.array([5.0]),
        'z': np.array([5.0])  # This will be updated sparsely
    }
    
    # Initialize optimizer
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'z':<12}")
    print("-" * 60)
    
    for i in range(50):
        # Most iterations: only x and y have gradients
        # Every 10th iteration: z also has a gradient
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y'],
            'z': 2 * params['z'] if i % 10 == 0 else np.array([0.0])
        }
        
        # Update parameters
        params = optimizer.update(params, grads)
        
        if i % 10 == 0:
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} {params['z'][0]:<12.6f}")
    
    print()
    print("Notice: z converges slower because it's updated less frequently,")
    print("but AdaGrad gives it a relatively larger effective learning rate!")
    print()


def show_learning_rate_decay():
    """
    Visualize how AdaGrad's effective learning rate decreases over time.
    """
    print("=" * 60)
    print("AdaGrad Learning Rate Decay")
    print("=" * 60)
    print("Effective learning rate = lr / sqrt(sum of squared gradients)")
    print()
    
    # Single parameter optimization
    param = np.array([10.0])
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Param':<15} {'Effective LR':<15}")
    print("-" * 60)
    
    effective_lrs = []
    
    for i in range(50):
        # Constant gradient
        grad = 2 * param
        
        # Calculate effective learning rate before update
        if i == 0:
            effective_lr = optimizer.learning_rate
        else:
            effective_lr = optimizer.learning_rate / np.sqrt(optimizer.cache['param'] + optimizer.epsilon)
        
        effective_lrs.append(effective_lr)
        
        # Update parameter
        params = {'param': param}
        grads = {'param': grad}
        updated = optimizer.update(params, grads)
        param = updated['param']
        
        if i % 10 == 0:
            print(f"{i:<12} {param[0]:<15.6f} {effective_lr[0]:<15.6f}")
    
    print()
    print("Notice: The effective learning rate monotonically decreases.")
    print("This can cause AdaGrad to stop learning prematurely in some cases.")
    print()


if __name__ == "__main__":
    demo_adagrad()
    print("\n")
    demo_sparse_gradients()
    print("\n")
    show_learning_rate_decay()
