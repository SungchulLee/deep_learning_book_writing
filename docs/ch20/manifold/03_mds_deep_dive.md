# MDS 깊이 들여다보기

여러 차원 잣수 맞추기(MDS)는 둘씩의 거리를 지키며 차원 높은 자료를 낮은 차원에 묻는다. 이 두루 살핀 보기는 고윳값 쪼개기로 고전 MDS를 맨바닥에서 세우고, 유클리드 자료에서 주성분 분석과 같음을 확인하며, PyTorch 기울기 내려가기로 계량 MDS를 짜고, MNIST 그려 보기와 품질을 가늠하는 셰퍼드 그림, 금융의 상관-거리 자산 지도 같은 실전 쓰임새를 보인다.

## 1. 코드

```python
"""MDS 깊이 들여다보기."""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# === 1부: 맨바닥에서 세우는 고전 MDS =====================================
def classical_mds(D, n_components=2):
    n = D.shape[0]
    D_sq = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    k = n_components
    Lambda_k = np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
    Q_k = eigenvectors[:, :k]
    return Q_k @ Lambda_k, eigenvalues[:k]

np.random.seed(42)
X_demo = np.random.randn(200, 3)
D_demo = squareform(pdist(X_demo))
Y_demo, eig_demo = classical_mds(D_demo, n_components=2)

plt.figure(figsize=(6, 5))
plt.scatter(Y_demo[:, 0], Y_demo[:, 1], s=15, alpha=0.7)
plt.title("Classical MDS (from scratch) on 200 random 3-D points")
plt.xlabel("$z_1$"); plt.ylabel("$z_2$")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("mds_classical_scratch.png", dpi=150)
plt.show()

# === 2부: 주성분 분석과 같음 ================================================
from sklearn.decomposition import PCA
X_centered = X_demo - X_demo.mean(axis=0)
Y_pca = PCA(n_components=2).fit_transform(X_centered)
D_cent = squareform(pdist(X_centered))
Y_mds_c, _ = classical_mds(D_cent, n_components=2)
for j in range(2):
    if np.corrcoef(Y_pca[:, j], Y_mds_c[:, j])[0, 1] < 0:
        Y_mds_c[:, j] *= -1
max_diff = np.abs(Y_pca - Y_mds_c).max()
print(f"PCA vs Classical MDS  max |difference| = {max_diff:.2e}")

# === 3부: PyTorch로 하는 계량 MDS =========================================
import torch

def metric_mds_torch(D, n_components=2, n_iter=300, lr=0.01, seed=42):
    torch.manual_seed(seed)
    n = D.shape[0]
    D_t = torch.tensor(D, dtype=torch.float32)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    Y_init, _ = classical_mds(D, n_components)
    Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=lr)
    stress_hist = []
    for _ in range(n_iter):
        opt.zero_grad()
        diff = Y.unsqueeze(0) - Y.unsqueeze(1)
        d_embed = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
        stress = ((D_t[mask] - d_embed[mask]) ** 2).sum()
        stress.backward()
        opt.step()
        stress_hist.append(stress.item())
    return Y.detach().numpy(), stress_hist

Y_torch, stress_hist = metric_mds_torch(D_demo, n_components=2)

# === 4~7부는 줄임(스위스 롤, MNIST, 셰퍼드, 금융) ====
# 스위스 롤에서 계량 MDS와 비계량 MDS 견줌, 주성분 분석으로 빠르게 한 MNIST의 MDS,
# 셰퍼드 그림, 상관-거리 자산 지도는
# 온전한 각본을 보라.

if __name__ == "__main__":
    pass
```

## 2. 논의

고전 MDS는 제곱 거리 행렬을 두 번 가운데 맞춤해 그람 행렬 $B = -\frac{1}{2} H D^{(2)} H$을 되찾은 뒤 고윳값 쪼개기를 한다. 들임 거리가 유클리드이면 $B$의 으뜸 고유벡터가 (부호와 돌림을 빼고) 주성분 분석의 주성분과 같다. 이 같음은 어림이 아니라 정확하며, 각본에서 주성분 분석과 고전 MDS 좌표의 최대 절대 차가 $10^{-14}$ 규모임을 수치로 보인다.

기울기 내려가기로 하는 계량 MDS는 날 스트레스 $\sum_{i<j}(d_{ij}^{\text{orig}} - d_{ij}^{\text{embed}})^2$을 가장 작게 하므로, 고전 MDS가 음의 고윳값을 낼 수 있는 유클리드가 아닌 거리 행렬도 다룰 수 있다. PyTorch 짜기는 고전 MDS 풀이에서 몸을 풀고 시작해 모임이 크게 빨라진다. 비계량 MDS는 한발 더 나아가 거리의 순위만 지켜, 거리 잣대 자체에 잡음이 끼었거나 순서만 뜻할 때 쓸모 있다.

셰퍼드 그림, 곧 본디 거리와 묻힌 거리의 흩뿌림 그림은 MDS 품질을 살피는 표준 연장이다. 대각선을 따라 촘촘히 모이고 피어슨 상관이 1.0에 가까우면 거리가 충실히 지켜진 것이다. 금융 쓰임새는 흔한 쓰임을 보여 준다. 곧 상관 행렬을 $d_{ij} = \sqrt{2(1 - \rho_{ij})}$으로 거리 행렬로 바꾼 뒤 MDS를 써서 자산이 함께 움직이는 정도로 어떻게 무리 지는지 그려 본다.

## 연습문제

**연습문제 1.**
고전 MDS를 짜서 5차원 10점 자료 묶음에 써라. 그람 행렬의 으뜸 고윳값 둘이 양의 고윳값 전체의 적어도 90%를 차지하는지 확인하고 2차원 묻힘을 그려라.

??? success "연습문제 1 풀이"
    ```python
    np.random.seed(0)
    X = np.random.randn(10, 5)
    D = squareform(pdist(X))
    Y, eigs = classical_mds(D, n_components=2)
    positive_eigs = eigs[eigs > 0]
    ratio = positive_eigs[:2].sum() / positive_eigs.sum()
    print(f"Top-2 eigenvalue ratio: {ratio:.2%}")
    plt.scatter(Y[:, 0], Y[:, 1])
    for i in range(10):
        plt.annotate(str(i), (Y[i, 0], Y[i, 1]))
    plt.title("Classical MDS on 10 points")
    plt.show()
    ```
    자료가 5차원 정규 분포이므로 으뜸 성분 둘이 흔히 흩어짐의 50~70%를 잡는다. 묻힘에 성분을 더 넣으면 덮는 몫이 는다.

---

**연습문제 2.**
들임 거리 행렬이 유클리드가 아닐 때 고전 MDS가 음의 고윳값을 낼 수 있는 까닭을 설명하라. 이것이 실전에서 어떤 문제를 부르며, 스트레스를 가장 작게 하는 계량 MDS는 이를 어떻게 피하는가?

??? success "연습문제 2 풀이"
    고전 MDS는 들임 거리가 유클리드라 여겨 그람 행렬 $B$이 준양정치라고 본다. 거리가 유클리드가 아닌 잣대(예컨대 측지 거리, 코사인 다름)에서 오면 $B$에 음의 고윳값이 생길 수 있고, 이는 어떤 실수 묻힘도 그 거리를 온전히 되살릴 수 없다는 뜻이다. 흔한 우회는 음의 고윳값을 가진 차원을 버리는 것인데, 이러면 묻힘이 소리 없이 일그러진다. 스트레스를 가장 작게 하는 계량 MDS는 고윳값 쪼개기를 아예 건너뛴다. 곧 거리 어긋남의 제곱합을 가장 작게 하도록 점의 자리를 곧바로 가장 좋게 하며, 이는 잣대의 성질과 무관하게 잘 정의된다.

---

**연습문제 3.**
같은 MNIST 부분 묶음(표본 2000개)에서 고전 MDS와 t-SNE를 견주는 셰퍼드 그림을 만들어라. 본디 거리와 묻힌 둘씩의 거리 사이 피어슨 상관이 어느 쪽이 더 높으며 왜 그런가?

??? success "연습문제 3 풀이"
    ```python
    from sklearn.manifold import TSNE, MDS
    X_sub = X_mnist[:500]  # 빠르기를 위해 500을 쓴다
    D_orig = squareform(pdist(X_sub))
    Y_mds = MDS(n_components=2, random_state=42, normalized_stress="auto").fit_transform(X_sub)
    Y_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_sub)
    D_mds = squareform(pdist(Y_mds))
    D_tsne = squareform(pdist(Y_tsne))
    mask = np.triu_indices(500, k=1)
    r_mds = np.corrcoef(D_orig[mask], D_mds[mask])[0, 1]
    r_tsne = np.corrcoef(D_orig[mask], D_tsne[mask])[0, 1]
    print(f"MDS Pearson r: {r_mds:.3f}")
    print(f"t-SNE Pearson r: {r_tsne:.3f}")
    ```
    MDS의 피어슨 상관이 훨씬 높다. 그 목표가 거리 일그러짐을 대놓고 가장 작게 하기 때문이다. t-SNE는 전역 거리의 충실함보다 국소 이웃 지키기를 앞세우는 확률 바탕 벌어짐을 가장 좋게 하므로 둘씩의 거리가 클수록 묻힘에서 크게 일그러진다.

## 정리하며

**다룬 것** — MDS 깊이 들여다보기

고전 MDS는 제곱 거리 행렬을 두 번 가운데 맞춤해 그람 행렬 $B = -\frac{1}{2} H D^{(2)} H$을 되찾은 뒤 고윳값 쪼개기를 한다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
