#!/usr/bin/env python3
"""
==============================================================================
EXERCISE 01: Implement Your Own SGD Optimizer
==============================================================================

OBJECTIVE:
Implement a basic Stochastic Gradient Descent optimizer from scratch.
This will deepen your understanding of how optimizers work under the hood.

DIFFICULTY: ⭐⭐ (Intermediate)

REQUIREMENTS:
- Complete beginner tutorials 01-06
- Understand gradient descent concept
- Basic Python classes

WHAT YOU'LL LEARN:
- How optimizers manage parameters
- Why we use classes for optimizers
- How learning rate affects convergence
- The difference between SGD and Adam (bonus)

TIME: 30-45 minutes

==============================================================================
"""

import torch
import matplotlib.pyplot as plt


class SimpleSGD:
    """
    TODO: Implement a simple Stochastic Gradient Descent optimizer.
    
    Your optimizer should:
    1. Store a list of parameters to optimize
    2. Store the learning rate
    3. Implement step() method to update parameters
    4. Implement zero_grad() method to clear gradients
    
    HINTS:
    - Use for loop to iterate over parameters
    - Remember: new_param = old_param - learning_rate * gradient
    - Use torch.no_grad() context for parameter updates
    - Check if param.grad is not None before using it
    """
    
    def __init__(self, parameters, learning_rate=0.01):
        """
        Initialize the optimizer.
        
        Args:
            parameters: List or iterator of parameters to optimize
            learning_rate: Step size for gradient descent
        """
        # TODO: Your code here
        # Store parameters as a list
        # Store learning rate
        pass
    
    def step(self):
        """
        Perform a single optimization step.
        Updates all parameters based on their gradients.
        """
        # TODO: Your code here
        # Iterate over all parameters
        # For each parameter with gradient:
        #   - Update: param = param - lr * param.grad
        # Remember to use torch.no_grad()!
        pass
    
    def zero_grad(self):
        """
        Zero out the gradients of all parameters.
        """
        # TODO: Your code here
        # Iterate over parameters
        # Set each param.grad to None (or call .zero_() if not None)
        pass


def test_optimizer():
    """
    Test your optimizer on a simple linear regression problem.
    """
    print("="*70)
    print("TESTING YOUR SGD OPTIMIZER")
    print("="*70)
    
    # Generate synthetic data: y = 3x + 2 + noise
    torch.manual_seed(42)
    n = 100
    x = torch.randn(n, 1)
    y_true = 3 * x + 2 + 0.5 * torch.randn(n, 1)
    
    # Initialize parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    # Create your optimizer
    # TODO: Uncomment the next line after implementing SimpleSGD
    # optimizer = SimpleSGD([w, b], learning_rate=0.1)
    
    # Training loop
    epochs = 100
    losses = []
    
    print("\nStarting training...\n")
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = x * w + b
        loss = ((y_pred - y_true) ** 2).mean()
        
        # Backward pass
        # TODO: Zero gradients using your optimizer
        # optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        # TODO: Take optimization step using your optimizer
        # optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}, "
                  f"w = {w.item():.4f}, b = {b.item():.4f}")
    
    print(f"\nFinal parameters: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"True parameters:  w = 3.0000, b = 2.0000")
    print(f"Error: w_err = {abs(w.item() - 3.0):.4f}, b_err = {abs(b.item() - 2.0):.4f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss with Your SGD Optimizer')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_01_loss.png', dpi=150)
    print("\nLoss plot saved as 'exercise_01_loss.png'")


# ==============================================================================
# BONUS CHALLENGES (Optional)
# ==============================================================================

def bonus_challenge_1():
    """
    BONUS 1: Add momentum to your optimizer
    
    Momentum helps accelerate SGD in relevant directions and dampens oscillations.
    
    Update rule with momentum:
        velocity = momentum * velocity + gradient
        parameter = parameter - learning_rate * velocity
    
    Try implementing SimpleSGDWithMomentum class!
    """
    pass


def bonus_challenge_2():
    """
    BONUS 2: Compare your SGD with PyTorch's built-in optimizer
    
    Import torch.optim and compare:
    - Your SimpleSGD
    - torch.optim.SGD
    - torch.optim.Adam
    
    Which converges fastest? Why?
    """
    pass


# ==============================================================================
# SOLUTION HINTS
# ==============================================================================

"""
HINT 1: Storing parameters
    self.parameters = list(parameters)
    self.lr = learning_rate

HINT 2: Step function structure
    with torch.no_grad():
        for param in self.parameters:
            if param.grad is not None:
                param -= self.lr * param.grad

HINT 3: Zero grad function
    for param in self.parameters:
        if param.grad is not None:
            param.grad = None

COMPLETE SOLUTION: See solution_01_sgd.py (don't peek unless stuck!)
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("YOUR TASK:")
    print("="*70)
    print("""
    1. Implement the SimpleSGD class above
    2. Complete the TODOs in test_optimizer()
    3. Run this script and check if training works
    4. Try different learning rates (0.01, 0.1, 0.5)
    5. (Optional) Attempt the bonus challenges
    """)
    print("="*70)
    
    # TODO: Uncomment after implementing
    # test_optimizer()
    
    print("\n💡 TIP: Start by implementing __init__, then zero_grad, then step")
    print("💡 TIP: Use print statements to debug your implementation")
    print("💡 TIP: Compare with beginner/05_complete_linear_regression.py")
