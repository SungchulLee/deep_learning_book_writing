"""
RMSprop (Root Mean Square Propagation) Optimizer
================================================

RMSprop is an adaptive learning rate method that addresses some of AdaGrad's
limitations by using a moving average of squared gradients instead of
accumulating all past squared gradients.

Key features:
- Uses exponential moving average of squared gradients
- Divides learning rate by the root of this average
- Works well on non-stationary problems (unlike AdaGrad)
- No bias correction (unlike Adam)

Developed by: Geoffrey Hinton (Coursera lecture)
"""

import numpy as np


class RMSprop:
    """
    RMSprop optimizer implementation.
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for parameter updates
    rho : float, default=0.9
        Decay rate for moving average of squared gradients
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        
        # State variables
        self.cache = {}  # Moving average of squared gradients
    
    def update(self, params, grads):
        """
        Update parameters using RMSprop algorithm.
        
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
            
            # Update moving average of squared gradients
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * (grads[key] ** 2)
            
            # Update parameters
            # Divide learning rate by the root of the moving average
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return updated_params


def demo_rmsprop():
    """
    Demonstration of RMSprop optimizer on a simple quadratic function.
    Minimize f(x, y) = x^2 + y^2
    """
    print("=" * 60)
    print("RMSprop Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x^2 + y^2")
    print()
    
    # Initialize parameters
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizer
    optimizer = RMSprop(learning_rate=0.1)
    
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


def compare_with_without_rmsprop():
    """
    Compare gradient descent with and without RMSprop adaptation.
    Shows how RMSprop handles different gradient scales.
    """
    print("=" * 60)
    print("RMSprop vs Standard Gradient Descent")
    print("=" * 60)
    print("Minimizing f(x, y) = 100*x^2 + y^2 (ill-conditioned)")
    print()
    
    # Initialize parameters
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_sgd = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizers
    optimizer_rmsprop = RMSprop(learning_rate=0.1)
    lr_sgd = 0.001  # Much smaller LR needed for SGD on ill-conditioned problems
    
    print(f"{'Iteration':<12} {'RMSprop f(x,y)':<20} {'SGD f(x,y)':<20}")
    print("-" * 60)
    
    for i in range(100):
        # Compute gradients: df/dx = 200x, df/dy = 2y
        grads_rmsprop = {
            'x': 200 * params_rmsprop['x'],
            'y': 2 * params_rmsprop['y']
        }
        grads_sgd = {
            'x': 200 * params_sgd['x'],
            'y': 2 * params_sgd['y']
        }
        
        # Update parameters
        params_rmsprop = optimizer_rmsprop.update(params_rmsprop, grads_rmsprop)
        params_sgd['x'] = params_sgd['x'] - lr_sgd * grads_sgd['x']
        params_sgd['y'] = params_sgd['y'] - lr_sgd * grads_sgd['y']
        
        # Compute function values
        f_rmsprop = 100 * params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_sgd = 100 * params_sgd['x']**2 + params_sgd['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_rmsprop[0]:<20.6f} {f_sgd[0]:<20.6f}")
    
    print()
    print("Notice: RMSprop converges faster on this ill-conditioned problem!")
    print()


if __name__ == "__main__":
    demo_rmsprop()
    print("\n")
    compare_with_without_rmsprop()
