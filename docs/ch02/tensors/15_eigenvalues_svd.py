"""Tutorial 15: Eigenvalues and SVD - Advanced matrix decompositions"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Eigenvalues and Eigenvectors")
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    print(f"A =\n{A}")
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    v1 = eigenvectors[:, 0].real
    lambda1 = eigenvalues[0].real
    print(f"\nVerification: A @ v1 ≈ λ1 * v1")
    print(f"A @ v1 = {A @ v1}")
    print(f"λ1 * v1 = {lambda1 * v1}")
    
    header("2. Singular Value Decomposition (SVD)")
    M = torch.randn(4, 3)
    print(f"M shape: {M.shape}")
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    print(f"U shape: {U.shape}")  # (4, 3)
    print(f"S shape: {S.shape}")  # (3,) - singular values
    print(f"Vh shape: {Vh.shape}")  # (3, 3)
    print(f"\nSingular values: {S}")
    M_reconstructed = U @ torch.diag(S) @ Vh
    print(f"Reconstruction error: {torch.norm(M - M_reconstructed):.2e}")
    
    header("3. Matrix Rank")
    M = torch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0], 
                      [7.0, 8.0, 9.0]])
    print(f"M =\n{M}")
    rank = torch.linalg.matrix_rank(M)
    print(f"Rank: {rank}")  # This matrix is rank-deficient
    M_full = torch.randn(3, 3)
    print(f"\nRandom matrix rank: {torch.linalg.matrix_rank(M_full)}")
    
    header("4. QR Decomposition")
    A = torch.randn(5, 3)
    Q, R = torch.linalg.qr(A)
    print(f"A shape: {A.shape}")
    print(f"Q shape: {Q.shape}")  # (5, 3) - orthonormal columns
    print(f"R shape: {R.shape}")  # (3, 3) - upper triangular
    print(f"Q is orthonormal: {torch.allclose(Q.T @ Q, torch.eye(3))}")
    print(f"Reconstruction: {torch.allclose(Q @ R, A)}")
    
    header("5. Cholesky Decomposition")
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])  # Positive definite
    print(f"A (positive definite) =\n{A}")
    L = torch.linalg.cholesky(A)
    print(f"L (lower triangular) =\n{L}")
    print(f"L @ L.T =\n{L @ L.T}")  # Should equal A
    
    header("6. Practical: PCA with SVD")
    data = torch.randn(100, 10)  # 100 samples, 10 features
    print(f"Data shape: {data.shape}")
    data_centered = data - data.mean(dim=0)
    U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)
    n_components = 3
    print(f"Top {n_components} principal components:")
    print(f"Explained variance: {S[:n_components]**2 / (S**2).sum()}")
    data_reduced = data_centered @ Vh.T[:, :n_components]
    print(f"Reduced data shape: {data_reduced.shape}")

if __name__ == "__main__":
    main()
