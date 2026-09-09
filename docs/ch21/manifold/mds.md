# 여러 차원 잣수 맞추기
둘씩의 거리를 지키며 차원 높은 자료를 낮은 차원에 묻기.

---

**여러 차원 잣수 맞추기(MDS)**는 점 사이의 거리가 본디 공간의 거리에 가까운 낮은 차원 묻힘을 찾는다. 전역 흩어짐을 지키는 주성분 분석과 달리 MDS는 거리 지키기를 곧바로 가장 좋게 하며, 그래서 선형 방법과 다양체 배움을 잇는 자연스러운 다리가 된다.

---

## 1. 문제 정식화

### 들임

거리(또는 다름) 행렬 $\mathbf{D} \in \mathbb{R}^{n \times n}$. 여기서 $d_{ij}$은 점 $i$과 $j$ 사이 거리이다:

$$d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$$

### 목표

다음을 채우는 낮은 차원 좌표 $\mathbf{Y} = [\mathbf{y}_1, \ldots, \mathbf{y}_n]^T \in \mathbb{R}^{n \times k}$을 찾아라:

$$\|\mathbf{y}_i - \mathbf{y}_j\| \approx d_{ij} \quad \forall\; i, j$$

---

## 2. 고전 MDS

고전 MDS는 거리가 유클리드일 때 **닫힌 꼴 풀이**를 주며, **두 번 가운데 맞춤**으로 거리 행렬을 안쪽 곱 행렬로 바꾼다.

### 두 번 가운데 맞춤

성분이 $d_{ij}^2$인 제곱 거리 행렬 $\mathbf{D}^{(2)}$이 주어질 때:

$$\mathbf{B} = -\frac{1}{2} \mathbf{H} \mathbf{D}^{(2)} \mathbf{H}$$

여기서 $\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T$은 가운데 맞춤 행렬이다.

### 두 번 가운데 맞춤이 통하는 까닭

가운데를 맞춘 자료 $\mathbf{X}$의 유클리드 거리에서:

$$d_{ij}^2 = \|\mathbf{x}_i - \mathbf{x}_j\|^2 = \mathbf{x}_i^T\mathbf{x}_i - 2\mathbf{x}_i^T\mathbf{x}_j + \mathbf{x}_j^T\mathbf{x}_j$$

두 번 가운데 맞춤이 대각 항을 없애 그람 행렬을 되찾는다:

$$b_{ij} = \mathbf{x}_i^T \mathbf{x}_j \quad \Longrightarrow \quad \mathbf{B} = \mathbf{X}\mathbf{X}^T$$

### 고윳값 쪼개기

$$\mathbf{B} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$$

$k$차원 묻힘은 다음과 같다:

$$\mathbf{Y} = \mathbf{Q}_k \mathbf{\Lambda}_k^{1/2}$$

여기서 $\mathbf{Q}_k$과 $\mathbf{\Lambda}_k$은 으뜸 $k$개 고유벡터와 고윳값을 담는다.

---

## 3. 알고리즘: 고전 MDS

1. 제곱 거리 행렬 $\mathbf{D}^{(2)}$을 셈한다
2. 두 번 가운데 맞춤: $\mathbf{B} = -\frac{1}{2}\mathbf{H}\mathbf{D}^{(2)}\mathbf{H}$
3. $\mathbf{B} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$으로 고윳값을 쪼갠다
4. 으뜸 $k$개 양의 고윳값과 고유벡터를 고른다
5. 묻는다: $\mathbf{Y} = \mathbf{Q}_k \mathbf{\Lambda}_k^{1/2}$

---

## 4. 구현

```python
import numpy as np

def classical_mds(D, n_components=2):
    """
    두 번 가운데 맞춤으로 하는 고전(계량) MDS.
    
    인수:
        D: 거리 행렬 [n, n](대칭, 대각이 0)
        n_components: 묻힘 차원
    
    반환값:
        Y: 묻힌 좌표 [n, n_components]
        eigenvalues: 으뜸 고윳값
    """
    n = D.shape[0]
    
    # 제곱 거리 행렬
    D_sq = D ** 2
    
    # 가운데 맞춤 행렬
    H = np.eye(n) - np.ones((n, n)) / n
    
    # 두 번 가운데 맞춤 -> 그람 행렬
    B = -0.5 * H @ D_sq @ H
    
    # 고윳값 쪼개기
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # 내림차순 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 으뜸 k개 양의 고윳값
    k = n_components
    Lambda_k = np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
    Q_k = eigenvectors[:, :k]
    
    Y = Q_k @ Lambda_k
    
    return Y, eigenvalues[:k]
```

---

## 5. 주성분 분석과 같음

거리 행렬이 가운데를 맞춘 자료의 유클리드 거리이면 고전 MDS는 (돌림과 되비침을 빼고) **주성분 분석과 같은 묻힘**을 낸다.

### 증명

특잇값 쪼개기가 $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$인 가운데 맞춘 자료 $\mathbf{X}$에서:

- 주성분 점수: $\mathbf{Z} = \mathbf{U}_k\mathbf{\Sigma}_k$
- 그람 행렬: $\mathbf{B} = \mathbf{X}\mathbf{X}^T = \mathbf{U}\mathbf{\Sigma}^2\mathbf{U}^T$
- MDS 묻힘: $\mathbf{Y} = \mathbf{U}_k\mathbf{\Sigma}_k$

두 묻힘은 똑같다.

### 확인

```python
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)
X = np.random.randn(100, 10)
X_centered = X - X.mean(axis=0)

# 주성분 분석
Y_pca = PCA(n_components=2).fit_transform(X_centered)

# 고전 MDS
D = squareform(pdist(X_centered))
Y_mds, _ = classical_mds(D, n_components=2)

# 부호를 맞춘다
for i in range(2):
    if np.corrcoef(Y_pca[:, i], Y_mds[:, i])[0, 1] < 0:
        Y_mds[:, i] *= -1

print(f"Max difference: {np.abs(Y_pca - Y_mds).max():.2e}")
# 기계 엡실론에 가깝다
```

**출력:**

```
Max difference: 1.55e-14
```

---

## 6. 계량 MDS(스트레스 가장 작게 하기)

거리가 온전히 유클리드가 아니면 고전 MDS가 음의 고윳값을 낼 수 있다. **계량 MDS**는 스트레스 함수를 곧바로 가장 작게 한다.

### 스트레스 함수

크러스컬의 날 스트레스:

$$\text{Stress}(\mathbf{Y}) = \sqrt{\frac{\sum_{i < j} (d_{ij} - \|\mathbf{y}_i - \mathbf{y}_j\|)^2}{\sum_{i < j} d_{ij}^2}}$$

### SMACOF 알고리즘

복잡한 함수를 위로 눌러 잣수 맞추기:

1. $\mathbf{Y}^{(0)}$을 첫자리매김한다(예컨대 고전 MDS 결과로)
2. 되풀이 $t$마다 구트만 바꿈을 셈한다:

   $$\mathbf{Y}^{(t+1)} = \frac{1}{n} \mathbf{Z}(\mathbf{Y}^{(t)}) \mathbf{Y}^{(t)}$$

   여기서 $\mathbf{Z}$은 비 $d_{ij} / \|\mathbf{y}_i^{(t)} - \mathbf{y}_j^{(t)}\|$에서 나온다
3. 스트레스가 모일 때까지 되풀이한다

### 스트레스 풀이하기

| 스트레스 | 품질 |
|--------|---------|
| < 0.05 | 아주 좋음 |
| 0.05~0.10 | 좋음 |
| 0.10~0.20 | 그런대로 |
| > 0.20 | 나쁨 |

---

## 7. PyTorch로 하는 계량 MDS

```python
import torch

def metric_mds_torch(D, n_components=2, n_iter=300, lr=0.01, seed=42):
    """
    스트레스에 기울기 내려가기를 써서 하는 계량 MDS.
    
    인수:
        D: 거리 행렬 [n, n](numpy 배열)
        n_components: 묻힘 차원
        n_iter: 가장 좋게 하기 되풀이 수
        lr: 학습률
    
    반환값:
        Y: 묻힌 좌표 [n, n_components]
        stress_history: 되풀이마다의 스트레스
    """
    torch.manual_seed(seed)
    n = D.shape[0]
    D_tensor = torch.tensor(D, dtype=torch.float32)
    
    # 위쪽 세모 가림막
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    
    # 고전 MDS 결과로 첫자리매김한다
    Y_init, _ = classical_mds(D, n_components)
    Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.Adam([Y], lr=lr)
    stress_history = []
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        
        # 묻힘에서 둘씩의 거리
        diff = Y.unsqueeze(0) - Y.unsqueeze(1)       # [n, n, k]
        dist_embed = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
        
        # 스트레스
        stress = ((D_tensor[mask] - dist_embed[mask]) ** 2).sum()
        stress.backward()
        optimizer.step()
        
        stress_history.append(stress.item())
    
    return Y.detach().numpy(), stress_history
```

---

## 8. 비계량 MDS

비계량 MDS는 거리의 크기가 아니라 **순위**만 지킨다.

### 목표

다음을 채우는 $\mathbf{Y}$을 찾아라:

$$d_{ij} < d_{kl} \implies \|\mathbf{y}_i - \mathbf{y}_j\| < \|\mathbf{y}_k - \mathbf{y}_l\|$$

### 한쪽으로만 가는 회귀를 쓴 스트레스

$$\text{Stress}_{\text{NM}} = \sqrt{\frac{\sum_{i < j} (\hat{d}_{ij} - f(d_{ij}))^2}{\sum_{i < j} \hat{d}_{ij}^2}}$$

여기서 $f$은 등위 회귀로 맞춘, 한쪽으로만 가는 함수이다.

### 언제 쓸까

- 다름이 순서만 뜻한다(정확한 거리가 아니라 순위)
- 본디 잣대를 모르거나 믿을 수 없다
- 사람이 느낀 닮음 자료

---

## 9. scikit-learn 겉면

```python
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# --- 특징으로 하는 계량 MDS ---
mds_metric = MDS(
    n_components=2,
    metric=True,
    dissimilarity='euclidean',
    random_state=42,
    normalized_stress='auto'
)
Y_metric = mds_metric.fit_transform(X)
print(f"Stress: {mds_metric.stress_:.4f}")

# --- 미리 셈한 거리 행렬로 하기 ---
D = squareform(pdist(X))
mds_pre = MDS(
    n_components=2,
    metric=True,
    dissimilarity='precomputed',
    random_state=42,
    normalized_stress='auto'
)
Y_pre = mds_pre.fit_transform(D)

# --- 비계량 MDS ---
mds_nm = MDS(
    n_components=2,
    metric=False,
    dissimilarity='precomputed',
    random_state=42,
    normalized_stress='auto'
)
Y_nm = mds_nm.fit_transform(D)
```

**출력:**

```
Stress: 9775.5425
```

---

## 10. 품질 값매김: 셰퍼드 그림

셰퍼드 그림은 본디 거리와 묻힌 거리를 그린다. 온전한 묻힘은 대각선 위에 놓인다:

```python
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def shepard_diagram(D_original, Y_embedded):
    """
    셰퍼드 그림으로 MDS 품질을 가늠한다.
    
    인수:
        D_original: 본디 거리 행렬 [n, n]
        Y_embedded: 묻힌 좌표 [n, k]
    """
    D_embed = squareform(pdist(Y_embedded))
    mask = np.triu_indices_from(D_original, k=1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(D_original[mask], D_embed[mask], alpha=0.3, s=10)
    
    lims = [0, max(D_original[mask].max(), D_embed[mask].max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect preservation')
    
    ax.set_xlabel('Original Distance')
    ax.set_ylabel('Embedded Distance')
    ax.set_title('Shepard Diagram')
    ax.legend()
    plt.tight_layout()
    plt.show()
```

---

## 11. 쓰임새: 상관-거리 자산 지도

상관 바탕 거리로 자산을 견주는 계량 금융에 MDS가 잘 맞는다.

### 상관 거리

수익률 상관 $\rho_{ij}$이 주어질 때:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

이는 제대로 된 잣대이다. 상관이 완전한 자산은 거리가 0, 상관이 없으면 $\sqrt{2}$, 상관이 음으로 완전하면 2이다.

```python
import numpy as np
import matplotlib.pyplot as plt

def asset_mds_map(returns, asset_names):
    """
    상관 거리에 MDS를 써서 자산 사이 관계를 그려 본다.
    
    인수:
        returns: 수익률 행렬 [기간 수, 자산 수]
        asset_names: 자산 이름 글줄의 목록
    """
    corr = np.corrcoef(returns.T)
    D = np.sqrt(2 * (1 - corr))
    
    Y, eigenvalues = classical_mds(D, n_components=2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(Y[:, 0], Y[:, 1], s=100, alpha=0.7)
    
    for i, name in enumerate(asset_names):
        ax.annotate(name, (Y[i, 0], Y[i, 1]),
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_title('Asset Structure via Correlation-Distance MDS')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# 만든 업종 수익률로 보는 보기
np.random.seed(42)
n_periods = 500
market = np.random.randn(n_periods)
tech   = 0.8 * market + 0.2 * np.random.randn(n_periods)
semis  = 0.7 * market + 0.5 * tech + 0.2 * np.random.randn(n_periods)
banks  = 0.6 * market + 0.3 * np.random.randn(n_periods)
energy = 0.3 * market + 0.5 * np.random.randn(n_periods)
gold   = -0.1 * market + 0.6 * np.random.randn(n_periods)
bonds  = -0.3 * market + 0.4 * np.random.randn(n_periods)

returns = np.column_stack([market, tech, semis, banks, energy, gold, bonds])
names = ['Market', 'Tech', 'Semis', 'Banks', 'Energy', 'Gold', 'Bonds']

asset_mds_map(returns, names)
```

---

## 12. MDS 변형 견줌

| 변형 | 지키는 것 | 방법 | 복잡도 |
|---------|-----------|--------|------------|
| **고전** | 유클리드 거리 | 고윳값 쪼개기 | $O(n^3)$ |
| **계량** | 거리의 크기 | SMACOF(되풀이) | $O(n^2 T)$ |
| **비계량** | 거리의 순위 | SMACOF + 등위 회귀 | $O(n^2 T)$ |

---

## 13. 복잡도 분석

| 걸음 | 고전 MDS | 계량 MDS |
|------|--------------|------------|
| 거리 행렬 | $O(n^2 d)$ | $O(n^2 d)$ |
| 두 번 가운데 맞춤 | $O(n^2)$ | — |
| 고윳값 쪼개기 | $O(n^3)$ | — |
| SMACOF 되풀이 | — | $O(n^2 T)$ |
| **전체** | $O(n^2 d + n^3)$ | $O(n^2 d + n^2 T)$ |

---

## 14. 한계

| 한계 | 결과 | 다루는 방법 |
|------------|-------------|--------------|
| **$O(n^2)$ 기억 공간** | 거리 행렬 전체가 필요하다 | 랜드마크 MDS |
| **표본 밖이 안 됨** | 새 점을 쏠 수 없다 | 매개변수를 가진 방법 |
| **전역에 치우침** | 국소 이웃을 일그러뜨릴 수 있다 | t-SNE, UMAP |
| **유클리드 가정** | 유클리드가 아닌 자료에서 음의 고윳값 | 계량 MDS |

---

## 15. 다음은

MDS는 전역 거리 짜임을 지키지만 비선형 다양체를 "펴지는" 못한다. **Isomap**은 유클리드 거리를 자료 다양체를 따라 셈한 측지 거리로 갈음해 고전 MDS를 넓힌다.

---

## 연습문제

**연습문제 1.**
고전 MDS와 계량 MDS의 차이를 설명하라. 둘은 언제 같아지는가?

??? success "연습문제 1 풀이"
    **고전 MDS**는 두 번 가운데 맞춘 거리 행렬의 고윳값 쪼개기로 안쪽 곱을(따라서 유클리드 거리를) 지키는 묻힘을 찾는다. **계량 MDS**는 스트레스 함수 $\text{Stress} = \sqrt{\sum_{i<j} (d_{ij} - \hat{d}_{ij})^2 / \sum_{i<j} d_{ij}^2}$을 가장 작게 한다. 여기서 $d_{ij}$은 본디 거리, $\hat{d}_{ij}$은 묻힌 거리이다. 거리가 유클리드이고 온전히 지킬 수 있을 때(묻힘 차원이 자료 차원과 같을 때) 둘은 같아진다. 고전 MDS가 더 빠르지만(닫힌 꼴) 유클리드 거리를 가정하고, 계량 MDS는 아무 다름이나 다룬다.

---

**연습문제 2.**
차원 높은 자료를 그려 보는 데 t-SNE와 MDS를 견주어라. 저마다 센 점과 약한 점은 무엇인가?

??? success "연습문제 2 풀이"
    **MDS**는 둘씩의 거리를 전역으로 지켜 자료의 전체 짜임을 간직한다. 다만 2차원으로 줄일 때 국소 이웃을 눌러 버려 무리를 가려내기 어렵다. **t-SNE**는 낮은 차원 공간에서 꼬리가 두꺼운 스튜던트 $t$ 분포를 써서 국소 짜임을 도드라지게 하고 잘 갈라진 무리를 낸다. 그러나 t-SNE는 전역 짜임을 일그러뜨리고(무리 사이 거리가 뜻이 없다), 웃매개변수(헷갈림도)에 민감하며, 볼록하지 않아 돌릴 때마다 결과가 다르다. 전역 관계를 알고 싶으면 MDS를, 무리를 가려내려면 t-SNE를 쓴다.

---

**연습문제 3.**
제곱 거리 행렬의 두 번 가운데 맞춤에서 고전 MDS 알고리즘을 이끌어 내어라.

??? success "연습문제 3 풀이"
    제곱 거리 $D^{(2)}_{ij} = d_{ij}^2$이 주어질 때 가운데 맞춤 행렬 $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$을 정한다. 안쪽 곱 행렬은 $B = -\frac{1}{2} H D^{(2)} H$이다. 거리가 유클리드이면 $B = XX^\top$이며 $X$은 가운데 맞춘 좌표 행렬이다. $B = V \Lambda V^\top$으로 고윳값을 쪼개고 으뜸 $k$개 고유벡터를 취하면 $X_k = V_k \Lambda_k^{1/2}$이다. 이것이 최소 제곱 뜻에서 거리를 가장 좋게 지키는 $k$차원 묻힘이다. $\square$

---

**연습문제 4.**
t-SNE는 낮은 차원 공간에서 왜 정규 분포가 아니라 스튜던트 $t$ 분포를 쓰는가?

??? success "연습문제 4 풀이"
    **붐빔 문제** 때문이다. 차원 높은 자료를 2차원에 묻으면 중간쯤 되는 거리를 충실히 지킬 넓이가 모자란다. 높은 차원에서 어지간히 떨어진 점들이 지나치게 가까이 밀린다. 스튜던트 $t$ 분포는 정규 분포보다 꼬리가 두꺼워, 어지간히 떨어진 점을 묻힘에서 훨씬 멀리 두어도 값이 크게 늘지 않는다. 그러면 가까운 무리가 말끔히 갈라질 자리가 생겨 붐빔 문제가 풀리고 무리가 더 잘 갈라진 그림이 나온다.

## 정리하며

| 개념 | 핵심 |
|---------|-----------|
| **목표** | 낮은 차원에서 둘씩의 거리 지키기 |
| **고전 MDS** | 두 번 가운데 맞춤 + 고윳값 쪼개기로 닫힌 꼴 |
| **계량 MDS** | 되풀이로 스트레스 가장 작게 하기(SMACOF) |
| **비계량 MDS** | 거리의 순위만 지킨다 |
| **주성분 분석과 같음** | 유클리드 거리의 고전 MDS = 주성분 분석 |
| **금융 쓰임** | 자산 꾸러미 짜임을 보는 상관-거리 지도 |

---
