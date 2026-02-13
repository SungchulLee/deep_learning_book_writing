"""
Optimizer Comparison: Adam vs RMSprop vs AdaGrad
================================================

This script compares the three adaptive learning rate optimizers on various
optimization problems to illustrate their strengths and differences.
"""

import numpy as np
import sys


# Import our optimizer implementations
from adam_optimizer import Adam
from rmsprop_optimizer import RMSprop
from adagrad_optimizer import AdaGrad


def test_simple_quadratic():
    """
    Test all three optimizers on a simple quadratic function.
    Minimize f(x, y) = x^2 + y^2
    """
    print("=" * 80)
    print("TEST 1: Simple Quadratic Function")
    print("=" * 80)
    print("Minimizing f(x, y) = x^2 + y^2")
    print("Starting point: x=10, y=10")
    print()
    
    # Initialize parameters for each optimizer
    params_adam = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizers
    adam = Adam(learning_rate=0.1)
    rmsprop = RMSprop(learning_rate=0.1)
    adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<15} {'RMSprop f(x,y)':<15} {'AdaGrad f(x,y)':<15}")
    print("-" * 80)
    
    for i in range(50):
        # Compute gradients
        grads_adam = {'x': 2 * params_adam['x'], 'y': 2 * params_adam['y']}
        grads_rmsprop = {'x': 2 * params_rmsprop['x'], 'y': 2 * params_rmsprop['y']}
        grads_adagrad = {'x': 2 * params_adagrad['x'], 'y': 2 * params_adagrad['y']}
        
        # Update parameters
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # Compute function values
        f_adam = params_adam['x']**2 + params_adam['y']**2
        f_rmsprop = params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_adagrad = params_adagrad['x']**2 + params_adagrad['y']**2
        
        if i % 10 == 0:
            print(f"{i:<12} {f_adam[0]:<15.8f} {f_rmsprop[0]:<15.8f} {f_adagrad[0]:<15.8f}")
    
    print("\nConclusion: All three converge well on this simple problem.")
    print()


def test_ill_conditioned():
    """
    Test on an ill-conditioned problem where gradients have different scales.
    Minimize f(x, y) = 100*x^2 + y^2
    """
    print("=" * 80)
    print("TEST 2: Ill-Conditioned Problem")
    print("=" * 80)
    print("Minimizing f(x, y) = 100*x^2 + y^2")
    print("Starting point: x=10, y=10")
    print("(x direction has much larger gradients than y direction)")
    print()
    
    # Initialize parameters
    params_adam = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizers
    adam = Adam(learning_rate=0.1)
    rmsprop = RMSprop(learning_rate=0.1)
    adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<15} {'RMSprop f(x,y)':<15} {'AdaGrad f(x,y)':<15}")
    print("-" * 80)
    
    for i in range(100):
        # Compute gradients: df/dx = 200x, df/dy = 2y
        grads_adam = {'x': 200 * params_adam['x'], 'y': 2 * params_adam['y']}
        grads_rmsprop = {'x': 200 * params_rmsprop['x'], 'y': 2 * params_rmsprop['y']}
        grads_adagrad = {'x': 200 * params_adagrad['x'], 'y': 2 * params_adagrad['y']}
        
        # Update parameters
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # Compute function values
        f_adam = 100 * params_adam['x']**2 + params_adam['y']**2
        f_rmsprop = 100 * params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_adagrad = 100 * params_adagrad['x']**2 + params_adagrad['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_adam[0]:<15.6f} {f_rmsprop[0]:<15.6f} {f_adagrad[0]:<15.6f}")
    
    print("\nConclusion: Adaptive methods handle different gradient scales automatically!")
    print()


def test_noisy_gradients():
    """
    Test with noisy gradients to see how robust each optimizer is.
    Minimize f(x, y) = x^2 + y^2 with added gradient noise
    """
    print("=" * 80)
    print("TEST 3: Noisy Gradients")
    print("=" * 80)
    print("Minimizing f(x, y) = x^2 + y^2 with noisy gradient estimates")
    print("Starting point: x=10, y=10")
    print()
    
    np.random.seed(42)
    
    # Initialize parameters
    params_adam = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # Initialize optimizers
    adam = Adam(learning_rate=0.1)
    rmsprop = RMSprop(learning_rate=0.1)
    adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<15} {'RMSprop f(x,y)':<15} {'AdaGrad f(x,y)':<15}")
    print("-" * 80)
    
    for i in range(100):
        # Compute gradients with noise
        noise_x = np.random.randn() * 0.5
        noise_y = np.random.randn() * 0.5
        
        grads_adam = {
            'x': 2 * params_adam['x'] + noise_x,
            'y': 2 * params_adam['y'] + noise_y
        }
        grads_rmsprop = {
            'x': 2 * params_rmsprop['x'] + noise_x,
            'y': 2 * params_rmsprop['y'] + noise_y
        }
        grads_adagrad = {
            'x': 2 * params_adagrad['x'] + noise_x,
            'y': 2 * params_adagrad['y'] + noise_y
        }
        
        # Update parameters
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # Compute function values
        f_adam = params_adam['x']**2 + params_adam['y']**2
        f_rmsprop = params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_adagrad = params_adagrad['x']**2 + params_adagrad['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_adam[0]:<15.6f} {f_rmsprop[0]:<15.6f} {f_adagrad[0]:<15.6f}")
    
    print("\nConclusion: Adam's momentum helps smooth out noisy gradients better.")
    print()


def test_rosenbrock():
    """
    Test on the Rosenbrock function, a classic optimization benchmark.
    Minimize f(x, y) = (1-x)^2 + 100*(y - x^2)^2
    """
    print("=" * 80)
    print("TEST 4: Rosenbrock Function (Challenging Benchmark)")
    print("=" * 80)
    print("Minimizing f(x, y) = (1-x)^2 + 100*(y - x^2)^2")
    print("Global minimum at (1, 1)")
    print("Starting point: x=-1, y=-1")
    print()
    
    # Initialize parameters
    params_adam = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    params_rmsprop = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    params_adagrad = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    
    # Initialize optimizers
    adam = Adam(learning_rate=0.01)
    rmsprop = RMSprop(learning_rate=0.01)
    adagrad = AdaGrad(learning_rate=0.1)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<18} {'RMSprop f(x,y)':<18} {'AdaGrad f(x,y)':<18}")
    print("-" * 80)
    
    for i in range(1000):
        # Compute Rosenbrock gradients
        # df/dx = -2(1-x) - 400x(y - x^2)
        # df/dy = 200(y - x^2)
        
        grad_x_adam = -2*(1 - params_adam['x']) - 400*params_adam['x']*(params_adam['y'] - params_adam['x']**2)
        grad_y_adam = 200*(params_adam['y'] - params_adam['x']**2)
        
        grad_x_rmsprop = -2*(1 - params_rmsprop['x']) - 400*params_rmsprop['x']*(params_rmsprop['y'] - params_rmsprop['x']**2)
        grad_y_rmsprop = 200*(params_rmsprop['y'] - params_rmsprop['x']**2)
        
        grad_x_adagrad = -2*(1 - params_adagrad['x']) - 400*params_adagrad['x']*(params_adagrad['y'] - params_adagrad['x']**2)
        grad_y_adagrad = 200*(params_adagrad['y'] - params_adagrad['x']**2)
        
        grads_adam = {'x': grad_x_adam, 'y': grad_y_adam}
        grads_rmsprop = {'x': grad_x_rmsprop, 'y': grad_y_rmsprop}
        grads_adagrad = {'x': grad_x_adagrad, 'y': grad_y_adagrad}
        
        # Update parameters
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # Compute function values
        f_adam = (1 - params_adam['x'])**2 + 100*(params_adam['y'] - params_adam['x']**2)**2
        f_rmsprop = (1 - params_rmsprop['x'])**2 + 100*(params_rmsprop['y'] - params_rmsprop['x']**2)**2
        f_adagrad = (1 - params_adagrad['x'])**2 + 100*(params_adagrad['y'] - params_adagrad['x']**2)**2
        
        if i % 200 == 0:
            print(f"{i:<12} {f_adam[0]:<18.6f} {f_rmsprop[0]:<18.6f} {f_adagrad[0]:<18.6f}")
    
    print()
    print(f"Final positions:")
    print(f"  Adam:    x={params_adam['x'][0]:.6f}, y={params_adam['y'][0]:.6f}")
    print(f"  RMSprop: x={params_rmsprop['x'][0]:.6f}, y={params_rmsprop['y'][0]:.6f}")
    print(f"  AdaGrad: x={params_adagrad['x'][0]:.6f}, y={params_adagrad['y'][0]:.6f}")
    print(f"  (Target: x=1.0, y=1.0)")
    print()
    print("Conclusion: Adam often performs best on challenging optimization landscapes.")
    print()


def print_summary():
    """
    Print a summary of optimizer characteristics.
    """
    print("=" * 80)
    print("OPTIMIZER SUMMARY")
    print("=" * 80)
    print()
    
    print("AdaGrad (2011):")
    print("  ✓ Adapts learning rate per parameter")
    print("  ✓ Good for sparse gradients")
    print("  ✗ Learning rate monotonically decreases (can stop learning)")
    print("  • Best for: Sparse data, NLP, recommender systems")
    print()
    
    print("RMSprop (2012):")
    print("  ✓ Uses moving average of squared gradients")
    print("  ✓ Fixes AdaGrad's diminishing learning rates")
    print("  ✓ Works well on non-stationary problems")
    print("  • Best for: RNNs, non-stationary objectives")
    print()
    
    print("Adam (2014):")
    print("  ✓ Combines RMSprop + Momentum")
    print("  ✓ Includes bias correction")
    print("  ✓ Usually works well with default hyperparameters")
    print("  ✓ Most popular optimizer in deep learning")
    print("  • Best for: General purpose, default choice")
    print()
    
    print("Hyperparameter Recommendations:")
    print("  Adam:    lr=0.001, beta1=0.9, beta2=0.999")
    print("  RMSprop: lr=0.001, rho=0.9")
    print("  AdaGrad: lr=0.01")
    print()


if __name__ == "__main__":
    print("\n")
    test_simple_quadratic()
    print("\n")
    test_ill_conditioned()
    print("\n")
    test_noisy_gradients()
    print("\n")
    test_rosenbrock()
    print("\n")
    print_summary()
