# 2차원 주성분 분석 PyTorch

이 보기는 PyTorch 텐서 연산과 특잇값 쪼개기로 주성분 분석을 짜서 NumPy 판과 같은 2차원에서 1차원 줄이기를 한다. GPU로 빨라지는 큰 자료 묶음을 다룰 때, 주성분 분석이 더 큰 깊은 배움 물길의 한 조각일 때, 미분할 수 있는 차원 줄이기를 위해 쏘기를 지나 저절로 미분해야 할 때 PyTorch를 쓰면 이롭다.

## 1. 코드

```python
"""2차원 주성분 분석 PyTorch."""
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 150

# === 촐레스키로 자료 만들기 =============================================
mean_true = torch.tensor([2.0, -1.0], device=device)
cov_true = torch.tensor([[3.0, 2.2], [2.2, 2.0]], device=device)
L = torch.linalg.cholesky(cov_true)
Z = torch.randn(n, 2, device=device)
X = mean_true + Z @ L.T

# === 가운데 맞추고 특잇값 쪼개기 셈하기 ==================================================
mu = X.mean(dim=0, keepdim=True)
X_centered = X - mu
U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
V = Vh.T
pc1 = V[:, 0]

explained_variance = (S ** 2) / (n - 1)
variance_ratios = explained_variance / explained_variance.sum()

# === 쏘고 다시 세우기 =================================================
scores_1d = X_centered @ pc1
X_recon = torch.outer(scores_1d, pc1) + mu
reconstruction_error = ((X - X_recon) ** 2).mean()

print(f"PC1 explains {variance_ratios[0].item():.1%} of variance")
print(f"Reconstruction MSE: {reconstruction_error.item():.6f}")

if __name__ == "__main__":
    pass
```

**출력:**

```
PC1 explains 96.1% of variance
Reconstruction MSE: 0.104872
```

## 2. 논의

PyTorch 짜기는 NumPy 판을 거의 줄 단위로 그대로 옮긴 것이며 일부러 그렇게 했다. PyTorch가 NumPy를 쓰던 이에게 낯익도록 만들어졌기 때문이다. 핵심 차이는 `torch.linalg.svd`을 쓴다는 것(이는 $V^T$이 아니라 $V^H$을 돌려주지만 실수 행렬에서는 같다), `.to(device)`으로 기기를 또렷이 다룬다는 것, 아무 텐서에나 `requires_grad=True`을 두어 주성분 분석 물길을 지나는 기울기를 셈할 수 있다는 것이다.

PyTorch에서 여러 변수 정규 표본을 만들려면 촐레스키 쪼개기 재주가 필요하다. 곧 공분산 $\Sigma = LL^\top$이 주어질 때 $Z \sim \mathcal{N}(0, I)$에 대해 바꿈 $X = \mu + ZL^\top$이 $\mathcal{N}(\mu, \Sigma)$의 표본을 낸다. 이는 NumPy의 `multivariate_normal`과 같되 기본 텐서 연산만 쓴다.

큰 규모의 쓰임새에서는 `torch.linalg.svd`을 GPU로 돌리면 CPU 기반 NumPy에 견주어 주성분 분석 셈 시간이 10~50배 줄 수 있다. 특잇값 쪼개기가 병목인, 표본이 수백만이거나 특징이 수천인 자료 묶음에서 중요하다. 신경망 층과 매끄럽게 어우러지므로 주성분 분석을 끝에서 끝까지 익히는 모델 안의 미분할 수 있는 미리 다듬기 모듈로 쓸 수도 있다.

## 연습문제

**연습문제 1.**
CPU에서 NumPy(`np.linalg.svd`)와 PyTorch(`torch.linalg.svd`)로 10000 x 784 아무 행렬의 특잇값 쪼개기 시간을 재어라. GPU가 있으면 PyTorch 판을 GPU에서도 재어라. 얼마나 빨라졌는지 알려라.

??? success "연습문제 1 풀이"
    ```python
    import time
    X_large_np = np.random.randn(10000, 784).astype(np.float32)
    X_large_pt = torch.from_numpy(X_large_np)

    t0 = time.time()
    np.linalg.svd(X_large_np, full_matrices=False)
    t_numpy = time.time() - t0

    t0 = time.time()
    torch.linalg.svd(X_large_pt, full_matrices=False)
    t_torch_cpu = time.time() - t0

    print(f"NumPy CPU: {t_numpy:.2f}s")
    print(f"PyTorch CPU: {t_torch_cpu:.2f}s")
    ```
    CPU에서는 둘 다 LAPACK을 쓰므로 PyTorch와 NumPy의 성능이 비슷하다. GPU에서는 이 크기의 행렬에서 PyTorch 특잇값 쪼개기가 10~50배 빠를 수 있다.

---

**연습문제 2.**
들임 자료에 `requires_grad=True`을 두어 주성분 쏘기를 미분할 수 있게 하라. 첫 자료 점에 대한 다시 세우기 어긋남의 기울기를 셈하라. 이 기울기는 기하로 무엇을 뜻하는가?

??? success "연습문제 2 풀이"
    ```python
    X_diff = X.clone().requires_grad_(True)
    mu_d = X_diff.mean(dim=0, keepdim=True)
    Xc_d = X_diff - mu_d
    _, _, Vh_d = torch.linalg.svd(Xc_d, full_matrices=False)
    pc1_d = Vh_d[0]
    scores = Xc_d @ pc1_d
    recon = torch.outer(scores, pc1_d) + mu_d
    mse = ((X_diff - recon) ** 2).mean()
    mse.backward()
    print(f"Gradient for point 0: {X_diff.grad[0]}")
    ```
    기울기는 자료 점을 그 방향으로 옮겼을 때 다시 세우기 어긋남이 가장 많이 줄어드는 쪽을 가리킨다. 기하로는 그 점에서 주성분1에 직교하는 성분(남는 방향)에 비례한다.

---

**연습문제 3.**
묶음 주성분 분석을 짜라. 곧 서로 얽히지 않은 자료 묶음 $B$개를 뜻하는 꼴 `(B, N, D)`의 텐서가 주어질 때 묶음 특잇값 쪼개기를 써서 되풀이 없이 자료 묶음마다 주성분 분석을 나란히 셈하라.

??? success "연습문제 3 풀이"
    ```python
    B, N, D = 5, 100, 10
    X_batch = torch.randn(B, N, D)
    mu_batch = X_batch.mean(dim=1, keepdim=True)
    Xc_batch = X_batch - mu_batch
    U_b, S_b, Vh_b = torch.linalg.svd(Xc_batch, full_matrices=False)
    # Vh_b 꼴: (묶음, D, D), 자료 묶음마다의 첫 주성분:
    pc1_batch = Vh_b[:, 0, :]  # (묶음, D)
    print(f"Batch PCs shape: {pc1_batch.shape}")
    ```
    `torch.linalg.svd`은 묶음 들임을 본디 받쳐 서로 얽히지 않은 특잇값 쪼개기를 나란히 셈한다. 특히 GPU에서 효율이 좋고 파이썬 되풀이의 군더더기를 피한다.

## 정리하며

**다룬 것** — 2차원 주성분 분석 PyTorch

PyTorch 짜기는 NumPy 판을 거의 줄 단위로 그대로 옮긴 것이며 일부러 그렇게 했다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
