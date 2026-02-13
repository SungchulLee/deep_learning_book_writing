"""Tutorial 20: Custom Autograd Functions - Extending PyTorch's differentiation"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def main():
    header("1. Custom Function: Square")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    square_fn = SquareFunction.apply
    y = square_fn(x)
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    y.sum().backward()
    print(f"x.grad = {x.grad}")
    
    header("2. Custom Function: ReLU")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    relu_fn = ReLUFunction.apply
    y = relu_fn(x)
    print(f"x = {x}")
    print(f"ReLU(x) = {y}")
    y.sum().backward()
    print(f"x.grad = {x.grad}")
    
    header("3. When to Use Custom Functions")
    print("""
    Use custom autograd functions when:
    1. Implementing new operations not in PyTorch
    2. Need custom gradient behavior
    3. Optimizing performance of specific operations
    4. Interfacing with external libraries
    5. Research: testing new activation functions
    """)
    
    header("4. Built-in vs Custom")
    x = torch.randn(5, requires_grad=True)
    built_in = torch.relu(x)
    custom = ReLUFunction.apply(x)
    print(f"Built-in ReLU: {built_in}")
    print(f"Custom ReLU: {custom}")
    print(f"Match? {torch.allclose(built_in, custom)}")

if __name__ == "__main__":
    main()
