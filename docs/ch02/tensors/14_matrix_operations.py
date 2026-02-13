"""Tutorial 14: Matrix Operations - Linear algebra essentials for ML"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Matrix Multiplication")
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    print(f"A =\n{A}\nB =\n{B}\n")
    C = A @ B  # Matrix multiplication
    print(f"A @ B =\n{C}")
    print(f"torch.mm(A, B) =\n{torch.mm(A, B)}")
    print(f"torch.matmul(A, B) =\n{torch.matmul(A, B)}")
    A_batch = torch.randn(10, 3, 4)  # Batch of matrices
    B_batch = torch.randn(10, 4, 5)
    C_batch = A_batch @ B_batch  # Batch matrix multiplication
    print(f"\nBatch: {A_batch.shape} @ {B_batch.shape} = {C_batch.shape}")
    
    header("2. Dot Product")
    v1 = torch.tensor([1, 2, 3], dtype=torch.float32)
    v2 = torch.tensor([4, 5, 6], dtype=torch.float32)
    dot = torch.dot(v1, v2)
    print(f"v1 = {v1}\nv2 = {v2}")
    print(f"dot(v1, v2) = {dot}")  # 1*4 + 2*5 + 3*6 = 32
    dot_manual = (v1 * v2).sum()
    print(f"Manual: (v1 * v2).sum() = {dot_manual}")
    
    header("3. Matrix-Vector Multiplication")
    M = torch.randn(3, 4)
    v = torch.randn(4)
    result = M @ v  # or torch.mv(M, v)
    print(f"M: {M.shape}, v: {v.shape}")
    print(f"M @ v: {result.shape}")  # (3,)
    result_mv = torch.mv(M, v)
    print(f"torch.mv(M, v): {result_mv.shape}")
    
    header("4. Outer Product")
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5])
    outer = torch.outer(v1, v2)
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"outer(v1, v2) =\n{outer}")
    print(f"Shape: {outer.shape}")  # (3, 2)
    
    header("5. Matrix Transpose")
    M = torch.arange(6).reshape(2, 3)
    print(f"M =\n{M}")
    print(f"M.T =\n{M.T}")
    print(f"M.transpose(0, 1) =\n{M.transpose(0, 1)}")
    
    header("6. Matrix Inverse")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"A =\n{A}")
    A_inv = torch.inverse(A)
    print(f"A^(-1) =\n{A_inv}")
    identity = A @ A_inv
    print(f"A @ A^(-1) ≈ I:\n{identity}")
    
    header("7. Determinant and Trace")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    det = torch.det(A)
    trace = torch.trace(A)
    print(f"A =\n{A}")
    print(f"det(A) = {det}")
    print(f"trace(A) = {trace}")  # Sum of diagonal
    
    header("8. Matrix Norms")
    A = torch.randn(3, 4)
    print(f"A shape: {A.shape}")
    fro_norm = torch.norm(A, p='fro')  # Frobenius norm
    print(f"Frobenius norm: {fro_norm:.4f}")
    nuc_norm = torch.norm(A, p='nuc')  # Nuclear norm
    print(f"Nuclear norm: {nuc_norm:.4f}")
    
    header("9. Solving Linear Systems")
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([9.0, 8.0])
    print(f"Solve Ax = b")
    print(f"A =\n{A}\nb = {b}")
    x = torch.linalg.solve(A, b)
    print(f"x = {x}")
    print(f"Verification: A @ x = {A @ x}")

if __name__ == "__main__":
    main()
