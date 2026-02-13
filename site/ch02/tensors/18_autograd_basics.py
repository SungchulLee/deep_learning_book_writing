"""Tutorial 18: Autograd Basics - Automatic differentiation fundamentals"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. requires_grad - Tracking Computations")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"x = {x}")
    print(f"x.requires_grad: {x.requires_grad}")
    y = torch.tensor([5.0, 6.0])
    print(f"y = {y}")
    print(f"y.requires_grad: {y.requires_grad}")
    z = x + y  # z inherits requires_grad from x
    print(f"\nz = x + y = {z}")
    print(f"z.requires_grad: {z.requires_grad}")
    
    header("2. Backward Pass - Computing Gradients")
    x = torch.tensor(3.0, requires_grad=True)
    print(f"x = {x}")
    y = x ** 2  # y = x^2
    print(f"y = x^2 = {y}")
    y.backward()  # Compute dy/dx
    print(f"x.grad (dy/dx = 2x = 6): {x.grad}")
    
    header("3. Gradient Accumulation")
    x = torch.tensor(5.0, requires_grad=True)
    for i in range(3):
        y = x ** 2
        y.backward()
        print(f"Iteration {i+1}: x.grad = {x.grad}")
    print("\nNote: Gradients ACCUMULATE! Use zero_grad() to reset.")
    
    header("4. Zeroing Gradients")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3
    y.backward()
    print(f"First backward: x.grad = {x.grad}")
    x.grad.zero_()  # Reset gradient
    print(f"After zero_grad(): x.grad = {x.grad}")
    y = x ** 2
    y.backward()
    print(f"Second backward: x.grad = {x.grad}")
    
    header("5. Multiple Variables")
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)
    z = x**2 + y**3  # z = x^2 + y^3
    print(f"x = {x}, y = {y}")
    print(f"z = x^2 + y^3 = {z}")
    z.backward()
    print(f"x.grad (dz/dx = 2x = 6): {x.grad}")
    print(f"y.grad (dz/dy = 3y^2 = 48): {y.grad}")
    
    header("6. Vector-Jacobian Product")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2  # Element-wise
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    gradient = torch.tensor([1.0, 1.0, 1.0])
    y.backward(gradient)  # Need gradient for non-scalar output
    print(f"x.grad = {x.grad}")  # [2, 4, 6]
    
    header("7. Detaching from Graph")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    print(f"y requires_grad: {y.requires_grad}")
    z = y.detach()  # Detach from computation graph
    print(f"z requires_grad: {z.requires_grad}")
    w = z * 3
    print(f"w requires_grad: {w.requires_grad}")
    
    header("8. No-Grad Context")
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x requires_grad: {x.requires_grad}")
    with torch.no_grad():
        y = x ** 2
        print(f"Inside no_grad, y requires_grad: {y.requires_grad}")
    print("Use no_grad() during inference to save memory!")
    
    header("9. Practical: Simple Loss Function")
    prediction = torch.tensor(2.5, requires_grad=True)
    target = torch.tensor(3.0)
    loss = (prediction - target) ** 2
    print(f"Prediction: {prediction}")
    print(f"Target: {target}")
    print(f"Loss (MSE): {loss}")
    loss.backward()
    print(f"Gradient: {prediction.grad}")
    print("Gradient tells us to increase prediction!")

if __name__ == "__main__":
    main()
