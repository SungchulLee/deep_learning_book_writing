"""Tutorial 19: Gradient Computation - Advanced gradient operations"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Higher-Order Gradients")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3  # y = x^3
    print(f"y = x^3 where x = {x}")
    grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"dy/dx = 3x^2 = {grad_y}")
    grad2_y = torch.autograd.grad(grad_y, x)[0]
    print(f"d²y/dx² = 6x = {grad2_y}")
    
    header("2. Gradient of Multiple Outputs")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y1 = x[0] ** 2
    y2 = x[1] ** 3
    print(f"x = {x}")
    print(f"y1 = x[0]^2 = {y1}")
    print(f"y2 = x[1]^3 = {y2}")
    grad_x = torch.autograd.grad([y1, y2], x, grad_outputs=[torch.tensor(1.0), torch.tensor(1.0)])[0]
    print(f"Gradient: {grad_x}")
    
    header("3. Jacobian Matrix")
    def f(x):
        return torch.stack([x[0]**2, x[1]**2, x[0]*x[1]])
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = f(x)
    print(f"x = {x}")
    print(f"f(x) = {y}")
    jacobian = torch.autograd.functional.jacobian(f, x)
    print(f"Jacobian:\n{jacobian}")
    
    header("4. Gradient Checking")
    def numerical_gradient(f, x, eps=1e-5):
        grad = torch.zeros_like(x)
        for i in range(x.numel()):
            x_plus = x.clone()
            x_plus.view(-1)[i] += eps
            x_minus = x.clone()
            x_minus.view(-1)[i] -= eps
            grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    def f(x): return (x**2).sum()
    y = f(x)
    y.backward()
    auto_grad = x.grad.clone()
    x.grad.zero_()
    num_grad = numerical_gradient(f, x)
    print(f"Autograd: {auto_grad}")
    print(f"Numerical: {num_grad}")
    print(f"Close? {torch.allclose(auto_grad, num_grad)}")
    
    header("5. Gradient Accumulation Pattern")
    model_output = torch.tensor(0.0, requires_grad=True)
    accumulated_loss = 0
    for i in range(3):
        loss = (model_output - i) ** 2
        loss.backward()
        accumulated_loss += loss.item()
        print(f"Step {i+1}: grad = {model_output.grad}")
    print(f"Total accumulated loss: {accumulated_loss}")
    
    header("6. Gradient Masking")
    x = torch.randn(5, requires_grad=True)
    y = x ** 2
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    y.backward(mask)
    print(f"x = {x}")
    print(f"Gradient (masked): {x.grad}")
    print("Only positions with mask=1.0 get gradients!")
    
    header("7. Practical: L2 Regularization")
    weights = torch.randn(10, requires_grad=True)
    predictions = weights.sum()
    target = torch.tensor(5.0)
    loss = (predictions - target) ** 2
    reg_lambda = 0.01
    regularization = reg_lambda * (weights ** 2).sum()
    total_loss = loss + regularization
    print(f"Loss: {loss.item():.4f}")
    print(f"Regularization: {regularization.item():.4f}")
    print(f"Total: {total_loss.item():.4f}")
    total_loss.backward()
    print(f"Gradient includes regularization term!")

if __name__ == "__main__":
    main()
