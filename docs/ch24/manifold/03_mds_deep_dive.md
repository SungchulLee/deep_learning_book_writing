# MDS 구현 자세히 살펴보기

이 예제는 고전 MDS를 고유분해로 구현하고, 유클리드 거리에서 PCA 점수와 비교한 뒤 좌표를 Adam으로 최적화한다. 본 코드는 무작위 3차원 데이터 200개를 사용한다. MNIST 비교와 금융 상관 지도의 아이디어는 본 실행 범위와 구분해서 읽는다.

## 1. 예제 코드와 출력

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

```
PCA vs Classical MDS  max |difference| = 5.33e-15
```

## 2. 고전 MDS가 구하는 좌표

제곱 거리 D²를 H=I−11ᵀ/n으로 이중 중심화해 B=−HD²H/2를 만든다. 유클리드 거리이면 B=X_cX_cᵀ이고 상위 양의 고윳값·고유벡터로 좌표 Y=Q_kΛ_k^(1/2)를 구성한다.

B는 표본 공간의 n×n 행렬이다. 그 고유벡터 Q는 원래 피처 공간의 PCA 방향 V와 같은 배열이 아니다. X_c=USVᵀ일 때 Q가 U와 대응하고 MDS **좌표** U_kS_k가 PCA **점수**와 같다. 부호·중복 고유공간의 회전 자유를 고려해야 한다.

목표 차원이 부족할 때 고전 MDS는 그람 행렬 오차(strain)를 줄인다. 계량 MDS가 줄이는 원래 거리 잔차 제곱합(stress)과는 다르므로 유클리드 데이터라도 저차원 해가 달라질 수 있다.

## 3. 코드의 고윳값 처리

`classical_mds`는 상위 k개 고윳값만 반환한다. 음수는 좌표 계산에서 0으로 잘리지만 반환값에는 남는다. 전체 양의 고윳값 합이나 음수 비율을 구하려면 전체 스펙트럼이 필요하다. 일부 반환값만으로 분모를 계산하면 보존 비율을 잘못 계산하게 된다.

수치 허용 오차보다 큰 음수가 있으면 입력 거리를 어떤 차원의 실수 유클리드 좌표로도 정확히 실현할 수 없다는 뜻이다. stress 최적화는 이 경우에도 근사 배치를 구할 수 있지만 표현 불가능성 자체를 해결하지는 않는다. 원래 거리와 결과 사이 왜곡을 보고해야 한다.

## 4. PyTorch 최적화의 실제 동작

`metric_mds_torch`는 고전 MDS 좌표에서 시작해 상삼각 쌍의 원시 stress를 Adam으로 줄인다. 일반적인 SMACOF 구현은 아니고 고전 초기화가 항상 수렴을 빠르게 하거나 더 좋은 국소해를 준다는 보장도 없다. 초기값 후보로 비교할 수 있다.

좌표 Y가 `requires_grad=True`인 최적화 변수다. 좌표도 학습 파라미터 역할을 하지만, 새로운 입력을 좌표로 바꾸는 뉴럴 넷트웍 사상을 학습하는 것은 아니다. 현재 코드는 CPU float32 텐서를 만들며 지정 반복 300회와 러닝레이트 0.01의 수렴 여부를 따로 판정하지 않는다.

거리 계산의 `sqrt(sum(diff**2)+1e-12)`는 0 근처를 매끄럽게 만드는 수치 처리다. 비대각의 중복 좌표도 양의 작은 거리가 되어 원래 stress와 미세하게 달라진다. 마지막 `stress_hist` 값은 마지막 갱신 전 값이며 출력 좌표에서 다시 계산한 값과 같다고 보장할 수 없다.

## 5. 최적화·거리 품질의 구분

원시 stress, 입력 거리 기준 정규화와 적합 거리 기준 Stress-1은 각각 정의를 명시한다. `normalized_stress='auto'`가 계량 MDS를 언제나 정규화해 주는 설정은 아니다. scikit-learn의 반환 정의는 사용 버전과 옵션에 맞게 확인한다.

셰퍼드 다이어그램은 원래 거리와 임베딩 거리를 비교한다. 피어슨 상관이 1이어도 모든 거리가 두 배라면 크기를 보존하지 않는다. 상관만 보지 말고 y=x와의 잔차, 스케일과 거리 구간별 오차를 함께 본다. 비계량 MDS에서는 적합한 단조 변환과의 관계를 평가한다.

## 6. 규모와 금융 응용의 조건

n×n float64 거리 행렬은 n=60,000일 때 28.8 GB이고 복사·고유분해 공간이 추가된다. 고전 MDS와 계량 반복의 계산 복잡도는 다르며 장치·목표 차원·반복 수를 포함해 비용을 판단한다. 랜드마크 수를 고정한 근사와 표본 외 확장은 비용을 줄이는 대신 근사 오차와 대표성을 검증해야 한다. Isomap도 전체 그래프 경로를 계산하므로 단순히 먼 쌍을 버리는 저비용 MDS는 아니다.

상관 지도에서 d_ij=√(2(1−ρ_ij))는 공통 관측으로 중심화·단위 노름화한 수익률 벡터들의 유클리드 거리다. 상수 열·누락 처리·쌍별 기간 차이와 반올림을 확인해야 한다. 완전 상관의 서로 다른 자산은 거리가 0일 수 있고 무상관은 독립을 뜻하지 않는다. 지도는 표본 상관의 탐색 도구이며 미래 관계나 거래 기회를 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
아래 코드로 상위 두 고윳값의 전체 양의 스펙트럼 대비 비율을 계산할 수 있는가?

</div>

??? success "연습문제 1 풀이"

    현재 함수는 상위 두 값만 반환한다. 그 반환값의 양수 부분에서 두 값의 합을 같은 전체 합으로 나누면 보통 100%가 되어 원래 의도와 다르다. 전체 B의 고윳값을 구해 분모에 모든 유의한 양수를 포함해야 한다. 5차원 정규 표본의 두 성분이 반드시 90%를 설명하지도 않는다.

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


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
비유클리드 입력에서 고전 MDS의 음수 고윳값과 계량 최적화의 관계를 설명하라.

</div>

??? success "연습문제 2 풀이"

    상당한 음수는 D를 정확히 유클리드 좌표로 실현할 수 없음을 나타낸다. 양의 부분을 취하면 그람 행렬의 근사를, 계량 최적화는 거리 stress의 근사를 구한다. 후자가 모든 관계를 정확히 복원하는 것은 아니며 목적과 초기값에 따라 비교한다. 비유사도 이름만으로 유클리드성을 판정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 MNIST 비교 코드와 문제의 데이터·방법 조건이 일치하는지 설명하라.

</div>

??? success "연습문제 3 풀이"

    X_mnist는 본 코드에서 정의하지 않아 별도로 준비해야 한다. 조각은 500개를 사용하고 MDS는 고전 함수가 아닌 scikit-learn의 계량 최적화다. 따라서 2000개 고전 MDS 비교로 부를 수 없다. 피어슨 상관의 순위도 실제 확인해야 하며 거리 절대 오차나 스케일 보존과는 다른 지표다.

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이중 중심화가 제거하는 자유와 남기는 자유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    거리는 이동에 불변이고 이중 중심화는 원점을 표본 평균에 두는 그람 행렬을 정한다. 좌표의 회전·반사는 여전히 자유며 중복 스펙트럼의 축도 유일하지 않다. 중심화를 했다고 좌표가 하나로 정해지는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
아래 Adam 최적화 코드가 뉴럴 넷트웍 없이 무엇을 학습하는지와 실행에 필요한 전제를 설명하라.

</div>

??? success "연습문제 5 풀이"

    Z 자체를 최적화 변수로 두어 좌표를 갱신한다. n, steps와 torch 텐서 D, 상삼각 등의 mask를 같은 장치에 정의해야 하며 분모 거리 제곱합은 양수여야 한다. 이 조각만으로 새 입력 변환기를 학습하지는 않는다. 최적화가 전역해에 도달하는지도 별도 문제다.

    ```python
    Z = torch.randn(n, 2, requires_grad=True)
    opt = torch.optim.Adam([Z], lr=0.1)
    for _ in range(steps):
        d = torch.cdist(Z, Z)
        loss = ((d - D)**2)[mask].sum() / (D**2)[mask].sum()
        opt.zero_grad(); loss.backward(); opt.step()
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
0 거리의 수치 처리와 아래 두 대안의 차이를 설명하라.

</div>

??? success "연습문제 6 풀이"

    torch.cdist는 0 거리에서 별도 역전파 규칙을 사용하므로 반드시 NaN이 난다고 단정할 수 없다. 원시 sqrt 구현과 구분한다. cdist 결과에 ε를 더하는 첫 식은 이미 수행된 거리 연산의 미분 규칙을 바꾸지 않는다. 제곱 거리 clamp 후 sqrt는 하한 밖 그래디언트를 바꾸며 원래 목적도 변경한다. 서로 다른 표본의 좌표가 겹칠 수 있으므로 대각 마스크만으로 모든 퇴화를 제거하지는 못한다.

    ```python
    mask = ~torch.eye(n, dtype=bool)
    loss = ((d - D)**2)[mask].sum() / (D**2)[mask].sum()
    ```
    ```python
    d = torch.cdist(Z, Z) + 1e-8          # 또는
    d = ((Z[:,None]-Z[None])**2).sum(-1).clamp(min=1e-12).sqrt()
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
표본 수가 열 배일 때 전체 거리 행렬 크기는 어떻게 바뀌는가?

</div>

??? success "연습문제 7 풀이"

    원소 수가 n²이므로 100배다. float64에서는 n=1000에 8 MB, 10000에 800 MB, 60000에 28.8 GB다. 전체 작업 메모리는 더 크다. 랜드마크와 표본 축소를 사용할 때는 필요한 근사 품질과 희귀 표본의 보존도 확인한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
상관 거리 √(2(1−ρ))의 유클리드 해석과 필요한 조건을 설명하라.

</div>

??? success "연습문제 8 풀이"

    공통 기간의 중심화된 비상수 수익률 벡터를 단위 노름 u_i로 만들면 ρ_ij=u_iᵀu_j다. 따라서 ||u_i−u_j||²=2(1−ρ_ij)다. 서로 다른 자산에 0 거리가 가능해 자산 집합에서는 유사거리일 수 있다. 누락에 따른 쌍별 기간 차이와 범위 오차를 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
상관 지도의 위험 해석에서 확인할 항목을 설명하라.

</div>

??? success "연습문제 9 풀이"

    추정 기간·관측 수와 누락 처리, 상관 추정 오차와 국면 변화를 확인한다. 상관은 선형 2차 관계여서 꼬리 의존성을 충분히 설명하지 못할 수 있다. 2차원 왜곡과 원래 상관값을 함께 보고 정규화·축소 추정의 효과도 보류 구간에서 평가한다. 위기에서 모든 상관이 반드시 1이 된다는 법칙은 없다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
고전 MDS 결과를 계량 최적화 초기값으로 쓰려는 아래 코드의 미완성 부분과 비교 목적을 설명하라.

</div>

??? success "연습문제 10 풀이"

    고전 함수 인수는 k가 아니라 n_components이고 반환은 좌표와 고윳값의 튜플이다. metric_mds라는 함수와 init, steps 인터페이스도 본문에 정의되지 않았다. 실제 연결에는 반환 좌표를 추출하고 초기값을 받을 최적화 함수를 구성해야 한다. 이어서 계산한다고 항상 더 좋거나 빠른 결과가 보장되는 것은 아니다.

    ```python
    Z0 = classical_mds(D, k=2)                  # 값싼 시작점
    Z = metric_mds(D, init=Z0, steps=300)       # 스트레스를 다듬는다
    ```


## 정리하며

고전 MDS 좌표와 PCA 점수의 관계는 표본 공간과 피처 공간을 구분해 이해한다. 거리 stress 최적화는 다른 목적이며 구현의 초기화·분모·기록 시점을 확인해야 한다. 일부 고윳값만으로 전체 보존율을 계산하지 말고, 거리 상관과 절대 오차·표본 외 해석도 나누어 평가한다.
