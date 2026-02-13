"""
Adam (Adaptive Moment Estimation) Optimizer
============================================

Adam combines the benefits of both RMSprop and momentum-based gradient descent.
It computes adaptive learning rates for each parameter using both first moment
(mean) and second moment (variance) of gradients.

Key features:
- Maintains exponential moving average of gradients (first moment)
- Maintains exponential moving average of squared gradients (second moment)
- Bias correction for both moments
- Generally works well with default hyperparameters

Paper: "Adam: A Method for Stochastic Optimization" by Kingma & Ba (2014)
"""

import numpy as np


class Adam:
    """
    Adam optimizer implementation.
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for parameter updates
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State variables
        self.m = {}  # First moment vector (mean of gradients)
        self.v = {}  # Second moment vector (variance of gradients)
        self.t = 0   # Time step
    
    def update(self, params, grads):
        """
        Update parameters using Adam algorithm.
        
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
        self.t += 1
        
        updated_params = {}
        
        for key in params.keys():
            # Initialize moment vectors if not exists
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params


def demo_adam():
    """
    Demonstration of Adam optimizer on a simple quadratic function.
    Minimize f(x, y) = x^2 + y^2
    """
    print("=" * 60)
    print("Adam Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x^2 + y^2")
    print()
    
    # Initialize parameters
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizer
    optimizer = Adam(learning_rate=0.1)
    
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


if __name__ == "__main__":
    demo_adam()
