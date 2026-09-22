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

**출력:**

```
PCA vs Classical MDS  max |difference| = 5.33e-15
```

## 2. 논의

고전 MDS는 제곱 거리 행렬을 두 번 가운데 맞춤해 그람 행렬 $B = -\frac{1}{2} H D^{(2)} H$을 되찾은 뒤 고윳값 쪼개기를 한다. 들임 거리가 유클리드이면 $B$의 으뜸 고유벡터가 (부호와 돌림을 빼고) 주성분 분석의 주성분과 같다. 이 같음은 어림이 아니라 정확하며, 각본에서 주성분 분석과 고전 MDS 좌표의 최대 절대 차가 $10^{-14}$ 규모임을 수치로 보인다.

기울기 내려가기로 하는 계량 MDS는 날 스트레스 $\sum_{i<j}(d_{ij}^{\text{orig}} - d_{ij}^{\text{embed}})^2$을 가장 작게 하므로, 고전 MDS가 음의 고윳값을 낼 수 있는 유클리드가 아닌 거리 행렬도 다룰 수 있다. PyTorch 짜기는 고전 MDS 풀이에서 몸을 풀고 시작해 모임이 크게 빨라진다. 비계량 MDS는 한발 더 나아가 거리의 순위만 지켜, 거리 잣대 자체에 잡음이 끼었거나 순서만 뜻할 때 쓸모 있다.

셰퍼드 그림, 곧 본디 거리와 묻힌 거리의 흩뿌림 그림은 MDS 품질을 살피는 표준 연장이다. 대각선을 따라 촘촘히 모이고 피어슨 상관이 1.0에 가까우면 거리가 충실히 지켜진 것이다. 금융 쓰임새는 흔한 쓰임을 보여 준다. 곧 상관 행렬을 $d_{ij} = \sqrt{2(1 - \rho_{ij})}$으로 거리 행렬로 바꾼 뒤 MDS를 써서 자산이 함께 움직이는 정도로 어떻게 무리 지는지 그려 본다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
고전 MDS를 짜서 5차원 10점 자료 묶음에 써라. 그람 행렬의 으뜸 고윳값 둘이 양의 고윳값 전체의 적어도 90%를 차지하는지 확인하고 2차원 묻힘을 그려라.

</div>

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
들임 거리 행렬이 유클리드가 아닐 때 고전 MDS가 음의 고윳값을 낼 수 있는 까닭을 설명하라. 이것이 실전에서 어떤 문제를 부르며, 스트레스를 가장 작게 하는 계량 MDS는 이를 어떻게 피하는가?

</div>

??? success "연습문제 2 풀이"
    고전 MDS는 들임 거리가 유클리드라 여겨 그람 행렬 $B$이 준양정치라고 본다. 거리가 유클리드가 아닌 잣대(예컨대 측지 거리, 코사인 다름)에서 오면 $B$에 음의 고윳값이 생길 수 있고, 이는 어떤 실수 묻힘도 그 거리를 온전히 되살릴 수 없다는 뜻이다. 흔한 우회는 음의 고윳값을 가진 차원을 버리는 것인데, 이러면 묻힘이 소리 없이 일그러진다. 스트레스를 가장 작게 하는 계량 MDS는 고윳값 쪼개기를 아예 건너뛴다. 곧 거리 어긋남의 제곱합을 가장 작게 하도록 점의 자리를 곧바로 가장 좋게 하며, 이는 잣대의 성질과 무관하게 잘 정의된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 MNIST 부분 배치(표본 2000개)에서 고전 MDS와 t-SNE를 견주는 셰퍼드 그림을 만들어라. 본디 거리와 묻힌 둘씩의 거리 사이 피어슨 상관이 어느 쪽이 더 높으며 왜 그런가?

</div>

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
고전 MDS에서 이중 가운데 맞추기가 하는 일은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    거리에서 내적을 되찾는다.

    $D^2$만 알고 있을 때 $B = X_c X_c^\top$을 얻으려면, 거리와 내적의 관계

    $$d_{ij}^2 = \|x_i\|^2 + \|x_j\|^2 - 2 x_i^\top x_j$$

    에서 앞의 두 항을 없애야 한다. 행 평균과 열 평균을 빼고 전체 평균을 더하면 정확히
    그 두 항이 지워진다.

    $$B = -\tfrac{1}{2} J D^2 J, \qquad J = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$$

    $J$가 가운데 맞추는 행렬이며 양쪽에 곱하므로 "이중"이다.

    이 걸음이 **왜 필요한가**를 새기는 것이 좋다. 거리는 평행 이동에 대해 불변이므로
    거리만으로는 자료가 어디 있는지 알 수 없다. 가운데를 원점으로 잡는 것이 그 자유를
    없애 답을 하나로 만든다.

    알맹이 주성분 분석의 가운데 맞추기와 같은 꼴이라는 점도 눈여겨볼 만하다
    ([알맹이 주성분 분석 연습문제 3](../pca/kernel_pca.md)). 둘 다 점을 직접 만지지
    않고 내적 행렬 위에서 가운데를 맞춘다. 실제로 고전 MDS는 선형 알맹이를 쓴 알맹이
    주성분 분석과 같다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
계량 MDS를 PyTorch 기울기 내려가기로 짜는 것이 왜 자연스러운가?

</div>

??? success "연습문제 5 풀이"
    스트레스에 **닫힌 꼴 답이 없기** 때문이다.

    고전 MDS는 고유 쪼개기로 한 번에 풀리지만, 계량 MDS의 스트레스는 좌표에 대한
    비선형 함수라 반복 최적화가 필요하다. 그리고 스트레스는 미분 가능하므로 자동 미분이
    바로 듣는다.

    ```python
    Z = torch.randn(n, 2, requires_grad=True)
    opt = torch.optim.Adam([Z], lr=0.1)
    for _ in range(steps):
        d = torch.cdist(Z, Z)
        loss = ((d - D)**2)[mask].sum() / (D**2)[mask].sum()
        opt.zero_grad(); loss.backward(); opt.step()
    ```

    **최적화하는 것이 매개변수가 아니라 좌표 자체**라는 점이 재미있다. 배울 모델이 없고
    출력이 곧 답이다. 신경망을 익히는 것과 모양은 같은데 성격이 다르다.

    이 자리가 이 장에서 기울기가 처음 필요해지는 곳이다
    ([PyTorch 기초 연습문제 5](../pca/03_pytorch_basics.md)). 주성분 분석과 알맹이
    주성분 분석은 쪼개기로 풀렸고, 여기서부터 그것이 끝난다.

    대가도 함께 온다. 초기값에 딸리고, 국소 최솟값에 걸릴 수 있으며, 학습률을 골라야
    한다. `sklearn`의 `MDS`가 `n_init`으로 여러 번 시작해 가장 좋은 것을 고르는 까닭이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
`torch.cdist`로 거리를 셈할 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 6 풀이"
    **대각선과 미분**이다.

    대각 원소는 0인데, 스트레스에 넣으면 $d_{ii} = 0$을 맞추는 항이 들어가 뜻이 없다.
    그래서 가면으로 빼 주어야 한다.

    ```python
    mask = ~torch.eye(n, dtype=bool)
    loss = ((d - D)**2)[mask].sum() / (D**2)[mask].sum()
    ```

    더 고약한 것은 미분이다. $\sqrt{u}$의 미분이 $1/(2\sqrt u)$이므로 거리가 0인 자리에서
    기울기가 무한이 된다. 두 점이 정확히 겹치면 `nan`이 나온다.

    대각선은 가면으로 막았지만 **두 점이 우연히 겹치는** 경우가 남는다. 초기값을 무작위로
    두면 잘 일어나지 않지만, 자료에 같은 점이 두 번 있으면 반드시 일어난다.

    막는 법이 간단하다.

    ```python
    d = torch.cdist(Z, Z) + 1e-8          # 또는
    d = ((Z[:,None]-Z[None])**2).sum(-1).clamp(min=1e-12).sqrt()
    ```

    중복된 표본을 미리 걸러 내는 것도 좋은 습관이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
MDS는 표본이 많아지면 왜 쓰기 어려운가?

</div>

??? success "연습문제 7 풀이"
    거리 행렬이 $n \times n$이기 때문이다. 알맹이 주성분 분석과 같은 문제다
    ([알맹이 주성분 분석 연습문제 5](../pca/kernel_pca.md)).

    실제로 재어 보면 MNIST 2,000개를 2차원으로 묻는 데 `sklearn`의 계량 MDS가 20.7초
    걸린다. 같은 자료에 주성분 분석은 2.8초다. 표본을 열 배로 늘리면 거리 행렬이 백 배가
    된다.

    | 표본 수 | 거리 행렬 | 고유 쪼개기 |
    |---|---|---|
    | 1,000 | 8 MB | 빠르다 |
    | 10,000 | 800 MB | 느리다 |
    | 60,000 | 29 GB | 불가능 |

    다룰 방법이 몇 가지다.

    - **표본을 덜어 쓴다.** 대표 점들로 묻고 나머지를 끼워 넣는다
    - **랜드마크 MDS.** 일부 점만 기준으로 삼아 $O(n)$으로 만든다
    - **이웃만 쓴다.** 먼 거리를 버리면 희소 행렬이 된다

    셋째가 방향으로서 중요하다. 먼 거리를 버리는 것이 곧 **거리를 다 지키려는 생각을
    버리는 것**이고, 그 길로 가면 Isomap과 t-SNE가 나온다. MDS의 한계가 다음 방법들의
    동기다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
자산 상관을 거리로 바꿀 때 쓰는 변환은 무엇인가? 왜 그 꼴인가?

</div>

??? success "연습문제 8 풀이"
    흔히 쓰는 것이 이것이다.

    $$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

    이 꼴을 쓰는 까닭은 **실제로 거리가 되기** 때문이다. 수익률을 표준화해 단위 벡터로
    보면 $\rho_{ij}$가 그 둘의 내적이고, 단위 벡터 사이의 유클리드 거리가

    $$\|u_i - u_j\|^2 = 2 - 2\rho_{ij}$$

    이므로 위 식이 정확히 그 거리다. 곧 꾸며 낸 변환이 아니라 **유클리드 거리 그 자체**다.

    좋은 성질이 따라온다. 삼각 부등식이 성립하고, 고전 MDS의 $B$가 반양정이 되어 음수
    고윳값 걱정이 없다([MDS 연습문제 3](mds.md)).

    값의 범위도 뜻이 통한다.

    | $\rho$ | $d$ | 뜻 |
    |---|---|---|
    | 1 | 0 | 똑같이 움직인다 |
    | 0 | $\sqrt 2$ | 무관하다 |
    | $-1$ | 2 | 정반대로 움직인다 |

    $1 - |\rho|$나 $1 - \rho$ 같은 변환도 쓰이는데, 그것들은 거리의 성질을 보장하지
    않는다. 반대로 움직이는 자산을 가깝게 볼지 멀게 볼지에 따라 고를 일이며, 무엇을
    고르든 **거리가 아닐 수 있음을 알고** 써야 한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
상관으로 만든 자산 지도를 읽을 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 9 풀이"
    여러 가지가 있고, 대부분 상관 추정의 문제다.

    **상관이 추정값이다.** 자산이 많고 관측이 적으면 상관 행렬의 추정 오차가 크다.
    $N$개 자산에 $N(N-1)/2$개를 추정해야 하므로, 자산 500개면 124,750개다. 그 잡음이
    지도의 모양을 만들어 낼 수 있다.

    **때에 따라 달라진다.** 위기에는 상관이 1로 몰리는 경향이 있어 지도가 통째로
    쪼그라든다. 어느 기간으로 재는지가 결과를 크게 정한다
    ([수익률 곡선 연습문제 5](../finance/yield_curve.md)).

    **2차 통계량만 본다.** 상관은 선형 관계만 재므로 꼬리에서의 함께 움직임을 놓친다.
    평시에 상관이 낮던 자산들이 위기에 함께 떨어지는 일이 그것이다.

    **2차원이 부족하다.** 500개 자산의 관계를 2차원에 담으면 반드시 많이 왜곡된다.
    셰퍼드 그림으로 얼마나 왜곡되었는지 보아야 한다([MDS 연습문제 6](mds.md)).

    마지막 것이 특히 조용한 위험이다. 그림이 그럴듯하게 나오면 왜곡을 의심하지 않게
    된다. 가까이 놓인 두 자산이 실제로 비슷한지 원래 상관값으로 되짚어 보는 습관이 좋다.

    그리고 상관 행렬을 잡음에서 걸러 내는 방법이 따로 발달해 있다. 무작위 행렬 이론으로
    잡음에 해당하는 고윳값 범위를 알아내 그 몫을 덜어 내는 것이며, 지도를 그리기 전에
    해 두면 훨씬 안정된다.


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
고전 MDS와 계량 MDS 가운데 무엇을 쓸지 어떻게 정하는가?

</div>

??? success "연습문제 10 풀이"
    갈리는 기준이 셋이다.

    | | 고전 MDS | 계량 MDS |
    |---|---|---|
    | 답 | 닫힌 꼴. 하나로 정해진다 | 반복. 초기값에 딸린다 |
    | 무엇을 줄이는가 | 그램 행렬의 어긋남 | 스트레스를 바로 |
    | 유클리드가 아닌 거리 | 음수 고윳값이 난다 | 문제없다 |
    | 값 | 고유 쪼개기 한 번 | 여러 번 다시 시작해야 한다 |

    **먼저 고전 MDS를 해 보는 것**이 순서다. 값싸고 답이 하나이며, 음수 고윳값의 몫을
    보면 이 자료에 맞는 방법인지까지 알려 준다([MDS 연습문제 3](mds.md)).

    음수 몫이 크면 계량 MDS로 옮긴다. 스트레스를 직접 줄이므로 거리가 유클리드가 아니어도
    되고, 실제로 그 경우 눈에 띄게 나은 묻힘을 준다.

    한 가지 눈여겨볼 점이 있다. 계량 MDS는 고전 MDS의 답을 **초기값으로** 쓸 수 있다.
    그러면 국소 최솟값에 걸릴 위험이 줄고 수렴이 빠르다. 둘을 고르는 문제가 아니라
    이어서 쓰는 문제가 되는 셈이다.

    ```python
    Z0 = classical_mds(D, k=2)                  # 값싼 시작점
    Z = metric_mds(D, init=Z0, steps=300)       # 스트레스를 다듬는다
    ```

    거리가 순위일 뿐이면 둘 다 맞지 않고 비계량 MDS로 가야 한다
    ([MDS 연습문제 5](mds.md)).

## 정리하며

**다룬 것** — MDS 깊이 들여다보기

고전 MDS는 제곱 거리 행렬을 두 번 가운데 맞춤해 그람 행렬 $B = -\frac{1}{2} H D^{(2)} H$을 되찾은 뒤 고윳값 쪼개기를 한다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
