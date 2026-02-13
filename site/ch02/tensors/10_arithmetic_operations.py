"""
Tutorial 10: Arithmetic Operations
===================================

Master element-wise and tensor arithmetic operations in PyTorch.

Key Concepts:
- Element-wise operations (+, -, *, /, **)
- In-place operations (add_, mul_, etc.)
- Mathematical functions (sqrt, exp, log, etc.)
- Aggregation vs element-wise
- Broadcasting basics
"""

import torch


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # -------------------------------------------------------------------------
    # 1. Basic Element-wise Arithmetic
    # -------------------------------------------------------------------------
    header("1. Basic Element-wise Arithmetic")
    
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    # Addition
    c_add = a + b
    print(f"a + b = {c_add}")  # tensor([6., 8., 10., 12.])
    
    # Also: torch.add(a, b)
    c_add_fn = torch.add(a, b)
    print(f"torch.add(a, b) = {c_add_fn}")
    
    # Subtraction
    c_sub = a - b
    print(f"\na - b = {c_sub}")  # tensor([-4., -4., -4., -4.])
    
    # Multiplication (element-wise, NOT matrix multiplication)
    c_mul = a * b
    print(f"a * b = {c_mul}")  # tensor([5., 12., 21., 32.])
    
    # Division
    c_div = b / a
    print(f"b / a = {c_div}")  # tensor([5., 3., 2.3333, 2.])
    
    # Floor division
    c_floordiv = b // a
    print(f"b // a = {c_floordiv}")  # tensor([5., 3., 2., 2.])
    
    # Modulo (remainder)
    c_mod = b % a
    print(f"b % a = {c_mod}")  # tensor([0., 0., 1., 0.])
    
    # Power
    c_pow = a ** 2
    print(f"a ** 2 = {c_pow}")  # tensor([1., 4., 9., 16.])
    
    # -------------------------------------------------------------------------
    # 2. In-place Operations (modify tensor in memory)
    # -------------------------------------------------------------------------
    header("2. In-place Operations")
    
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Original x = {x}")
    print(f"Memory address: {id(x)}")
    
    # In-place operations end with underscore (_)
    x.add_(10)  # x = x + 10
    print(f"After x.add_(10) = {x}")
    print(f"Memory address: {id(x)}")  # Same address!
    
    x.mul_(2)  # x = x * 2
    print(f"After x.mul_(2) = {x}")
    
    x.div_(4)  # x = x / 4
    print(f"After x.div_(4) = {x}")
    
    # Warning: In-place operations on tensors that require gradients can cause errors!
    # y = torch.tensor([1.0], requires_grad=True)
    # y.add_(1)  # RuntimeError: Can't perform in-place on leaf variable with grad
    
    # -------------------------------------------------------------------------
    # 3. Scalar Operations
    # -------------------------------------------------------------------------
    header("3. Scalar Operations")
    
    vec = torch.tensor([1, 2, 3, 4, 5])
    print(f"vec = {vec}")
    
    # Scalars broadcast automatically
    vec_plus_10 = vec + 10
    print(f"vec + 10 = {vec_plus_10}")
    
    vec_times_2 = vec * 2
    print(f"vec * 2 = {vec_times_2}")
    
    vec_pow_2 = vec ** 2
    print(f"vec ** 2 = {vec_pow_2}")
    
    # -------------------------------------------------------------------------
    # 4. Mathematical Functions
    # -------------------------------------------------------------------------
    header("4. Mathematical Functions")
    
    x = torch.tensor([0.0, 1.0, 4.0, 9.0])
    print(f"x = {x}\n")
    
    # Square root
    sqrt_x = torch.sqrt(x)
    print(f"sqrt(x) = {sqrt_x}")
    
    # Exponential
    exp_x = torch.exp(x)
    print(f"exp(x) = {exp_x}")
    
    # Natural logarithm (log base e)
    x_pos = torch.tensor([1.0, 2.718, 7.389])
    log_x = torch.log(x_pos)
    print(f"\nlog({x_pos}) = {log_x}")
    
    # Base 10 logarithm
    log10_x = torch.log10(x_pos)
    print(f"log10({x_pos}) = {log10_x}")
    
    # Absolute value
    x_neg = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
    abs_x = torch.abs(x_neg)
    print(f"\nabs({x_neg}) = {abs_x}")
    
    # Sign function
    sign_x = torch.sign(x_neg)
    print(f"sign({x_neg}) = {sign_x}")
    
    # Rounding operations
    x_float = torch.tensor([1.2, 2.5, -3.7, 4.9])
    print(f"\nx = {x_float}")
    print(f"round(x) = {torch.round(x_float)}")
    print(f"floor(x) = {torch.floor(x_float)}")
    print(f"ceil(x) = {torch.ceil(x_float)}")
    print(f"trunc(x) = {torch.trunc(x_float)}")  # Remove decimal part
    
    # -------------------------------------------------------------------------
    # 5. Trigonometric Functions
    # -------------------------------------------------------------------------
    header("5. Trigonometric Functions")
    
    angles = torch.tensor([0.0, torch.pi/4, torch.pi/2, torch.pi])
    print(f"angles = {angles}")
    
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    tan_angles = torch.tan(angles)
    
    print(f"sin(angles) = {sin_angles}")
    print(f"cos(angles) = {cos_angles}")
    print(f"tan(angles) = {tan_angles}")
    
    # Inverse trigonometric
    values = torch.tensor([0.0, 0.5, 1.0])
    print(f"\nvalues = {values}")
    print(f"arcsin(values) = {torch.asin(values)}")
    print(f"arccos(values) = {torch.acos(values)}")
    print(f"arctan(values) = {torch.atan(values)}")
    
    # -------------------------------------------------------------------------
    # 6. Clipping and Clamping
    # -------------------------------------------------------------------------
    header("6. Clipping and Clamping")
    
    x = torch.tensor([-5.0, -2.0, 0.0, 3.0, 10.0])
    print(f"x = {x}")
    
    # Clamp values to range [min, max]
    clamped = torch.clamp(x, min=-3.0, max=5.0)
    print(f"clamp(x, -3, 5) = {clamped}")  # [-3., -2., 0., 3., 5.]
    
    # Only minimum
    clamped_min = torch.clamp(x, min=0.0)
    print(f"clamp(x, min=0) = {clamped_min}")  # ReLU-like behavior
    
    # Only maximum
    clamped_max = torch.clamp(x, max=2.0)
    print(f"clamp(x, max=2) = {clamped_max}")
    
    # -------------------------------------------------------------------------
    # 7. Comparison Operations
    # -------------------------------------------------------------------------
    header("7. Comparison Operations")
    
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    print(f"a > b: {a > b}")
    print(f"a >= b: {a >= b}")
    print(f"a < b: {a < b}")
    print(f"a <= b: {a <= b}")
    
    # Element-wise maximum/minimum
    print(f"\ntorch.max(a, b) (element-wise): {torch.max(a, b)}")
    print(f"torch.min(a, b) (element-wise): {torch.min(a, b)}")
    
    # -------------------------------------------------------------------------
    # 8. Matrix Operations (2D tensors)
    # -------------------------------------------------------------------------
    header("8. Matrix Operations")
    
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print(f"A =\n{A}\n")
    print(f"B =\n{B}\n")
    
    # Element-wise multiplication
    C_elem = A * B
    print(f"A * B (element-wise) =\n{C_elem}")
    
    # Matrix multiplication
    C_matmul = A @ B  # or torch.matmul(A, B)
    print(f"\nA @ B (matrix multiplication) =\n{C_matmul}")
    
    # Also: torch.mm() for 2D matrix multiplication
    C_mm = torch.mm(A, B)
    print(f"torch.mm(A, B) =\n{C_mm}")
    
    # -------------------------------------------------------------------------
    # 9. Reduction Operations
    # -------------------------------------------------------------------------
    header("9. Reduction Operations")
    
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    print(f"x =\n{x}\n")
    
    # Sum all elements
    total = torch.sum(x)
    print(f"sum(x) = {total}")
    
    # Sum along dimension 0 (collapse rows)
    sum_dim0 = torch.sum(x, dim=0)
    print(f"sum(x, dim=0) = {sum_dim0}")  # [5., 7., 9.]
    
    # Sum along dimension 1 (collapse columns)
    sum_dim1 = torch.sum(x, dim=1)
    print(f"sum(x, dim=1) = {sum_dim1}")  # [6., 15.]
    
    # Mean
    mean_all = torch.mean(x)
    print(f"\nmean(x) = {mean_all}")
    
    mean_dim0 = torch.mean(x, dim=0)
    print(f"mean(x, dim=0) = {mean_dim0}")
    
    # Min and Max
    print(f"\nmin(x) = {torch.min(x)}")
    print(f"max(x) = {torch.max(x)}")
    
    # argmin and argmax (return indices)
    print(f"argmin(x) = {torch.argmin(x)}")  # Flattened index
    print(f"argmax(x) = {torch.argmax(x)}")
    
    # -------------------------------------------------------------------------
    # 10. Common Patterns and Tips
    # -------------------------------------------------------------------------
    header("10. Common Patterns and Tips")
    
    print("""
    Key Takeaways:
    
    1. **Element-wise Operations**
       - Most operators (+, -, *, /) work element-wise
       - Use @ or torch.matmul() for matrix multiplication
    
    2. **In-place Operations**
       - End with underscore: add_(), mul_(), etc.
       - Modify tensor in memory (no new tensor created)
       - Can't use on tensors with requires_grad=True
    
    3. **Broadcasting**
       - Scalars automatically broadcast to tensor shape
       - See tutorial 11 for detailed broadcasting rules
    
    4. **Function vs Method**
       - torch.add(a, b) == a.add(b) == a + b
       - Use what's most readable for your code
    
    5. **Performance**
       - In-place operations save memory but be careful with gradients
       - Use torch.* functions for better optimization potential
    """)
    
    # -------------------------------------------------------------------------
    # Practice Exercises
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    Try these:
    
    1. Compute: (x^2 + 2*x + 1) for x = [0, 1, 2, 3, 4]
    2. Normalize values to range [0, 1]: (x - min) / (max - min)
    3. Compute L2 norm (Euclidean length) of a vector
    4. Element-wise max of three tensors
    5. Sigmoid function: 1 / (1 + exp(-x))
    """)
    
    # Solutions
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    ex1 = x**2 + 2*x + 1
    print(f"\n1. (x^2 + 2*x + 1) = {ex1}")
    
    x2 = torch.tensor([3.0, 5.0, 1.0, 9.0])
    ex2 = (x2 - x2.min()) / (x2.max() - x2.min())
    print(f"2. Normalized = {ex2}")
    
    vec = torch.tensor([3.0, 4.0])
    ex3 = torch.sqrt(torch.sum(vec ** 2))
    print(f"3. L2 norm = {ex3}")
    
    t1 = torch.tensor([1, 5, 3])
    t2 = torch.tensor([2, 4, 6])
    t3 = torch.tensor([3, 3, 3])
    ex4 = torch.max(torch.max(t1, t2), t3)
    print(f"4. Element-wise max = {ex4}")
    
    x5 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    ex5 = 1 / (1 + torch.exp(-x5))
    print(f"5. Sigmoid = {ex5}")


if __name__ == "__main__":
    main()
