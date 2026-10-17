# 다차원 척도법(MDS)

다차원 척도법은 점 사이의 비유사도나 거리에서 저차원 좌표를 구성한다. 고전 MDS의 그람 행렬 근사와 계량 MDS의 거리 stress 최소화는 서로 다른 목적이다. 입력 거리가 유클리드여도 목표 차원이 부족하면 두 방법이 같은 좌표를 줄 필요는 없다.

## 1. 입력과 목표

입력 D는 n×n 대칭 비유사도 행렬이다. 일반적으로 대각은 0, 비대각은 비음수이며 유한해야 한다. 비유사도가 반드시 원래 좌표의 유클리드 거리나 삼각부등식을 만족하는 거리는 아니다. 목표는 Y∈ℝⁿˣᵏ에서 δ_ij=||y_i−y_j||가 주어진 관계를 가능한 한 잘 표현하게 하는 것이다. 무엇을 “잘” 표현하는지는 방법별 목적 함수로 정한다.

## 2. 고전 MDS의 이중 중심화

D^{(2)}는 원소별 제곱 거리 행렬이며 H=I−11ᵀ/n으로 둔다.

$$B=-\frac12HD^{(2)}H.$$

중심화된 유클리드 좌표 X_c에서 d_ij²=||x_i||²+||x_j||²−2x_iᵀx_j다. H1=0이므로 양 끝의 노름 항이 사라져 B=X_cX_cᵀ가 된다. 따라서 B의 양의 고윳값 ρ_j와 단위 고유벡터 q_j로 좌표를 구성할 수 있다.

$$Y=Q_k\operatorname{diag}(\sqrt{\rho_1},\ldots,\sqrt{\rho_k}).$$

중심화된 B가 양의 준정부호이면 주어진 D는 유클리드 거리로 실현할 수 있다. 양의 고윳값 수만큼 차원을 남기면 거리를 정확히 복원한다. 더 작은 k에서는 그람 행렬의 저랭크 근사인 **strain**을 최소화하며, 원래 거리의 제곱 잔차 stress를 직접 최소화하는 것과 다르다.

## 3. 구현과 음의 고윳값

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

코드는 상위 `n_components`개를 택한 뒤 음수 고윳값을 0으로 잘라 좌표를 만든다. 양의 성분이 부족하면 0 좌표가 생기며 반환 고윳값 자체는 음수일 수 있다. 원하는 차원, 입력의 대칭성·비음수성과 수치 허용 오차를 별도로 검증해야 한다.

행렬 H를 명시적으로 곱하는 중심화는 밀집 행렬곱 비용이 든다. 행·열 평균을 사용하는 구현이라면 중심화 자체를 O(n²)로 할 수 있지만 이 코드가 그 효율적인 구현인 것은 아니다. 작은 음수는 반올림 오차일 수 있고 상당한 음수는 비유클리드 관계를 나타낸다. 음수 제거는 입력 관계를 바꾸는 근사이므로 버린 스펙트럼과 거리 왜곡을 함께 확인한다.

## 4. 유클리드 거리에서 PCA와의 관계

X_c=USVᵀ이면 B=US²Uᵀ다. 따라서 고전 MDS의 좌표 U_kS_k는 PCA 점수와 같다. 부호와 중복 고유공간의 회전 자유를 제외한 관계다.

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

```
Max difference: 1.55e-14
```

이 확인 코드는 100×10 무작위 데이터에서 두 축의 부호를 맞추어 비교한다. 보존된 작은 차이는 한 실행의 수치 예시다. 중복 고윳값에서는 축별 부호 정렬만으로 두 결과를 맞출 수 없으며 직교 정렬이나 투영 관계를 비교해야 한다. 원래 좌표가 있으면 PCA를 직접 계산하는 것이 전체 표본 거리 행렬을 만드는 것보다 경제적일 수 있다. 거리만 있는 경우에는 MDS의 입력 형식이 유용하다.

## 5. 계량 MDS와 stress

계량 MDS는 보통 다음 원시 stress를 최소화한다.

$$S(Y)=\sum_{i<j}w_{ij}(d_{ij}-\delta_{ij})^2.$$

이 절의 예제는 모든 쌍에 같은 웨이트를 사용한다. 크기를 정규화해 보고할 때는 분모를 명시해야 한다. 예를 들어 √(S/Σd_ij²)는 입력 거리 기준 상대 오차이고, Stress-1은 통상 √(S/Σδ_ij²)처럼 적합 거리 기준이다. 원시 합, 제곱근과 정규화 비율을 서로 바꾸어 읽지 않는다.

SMACOF는 stress를 상계하는 함수를 반복 최소화한다. 완전한 동일 웨이트의 중심화된 배치에서는 다음 Guttman 갱신을 쓸 수 있다.

$$Y^{(t+1)}=B(Y^{(t)})Y^{(t)}/n.$$

비대각 B_ij=−d_ij/δ_ij이고 대각은 행 합이 0이 되게 정한다. δ_ij=0인 경우의 처리와 누락·가중 관계에는 별도의 정의가 필요하다. 적절한 majorization 단계는 stress를 증가시키지 않지만 전역 최적해를 보장하지 않는다.

## 6. PyTorch의 직접 최적화

```python
import torch

def metric_mds_torch(D, n_components=2, n_iter=300, lr=0.01, seed=42):
    """
    스트레스에 그래디언트 내려가기를 써서 하는 계량 MDS.
    
    인수:
        D: 거리 행렬 [n, n](numpy 배열)
        n_components: 묻힘 차원
        n_iter: 가장 좋게 하기 되풀이 수
        lr: 러닝레이트
    
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

이 코드는 SMACOF가 아니라 Adam으로 좌표를 갱신한다. 고전 MDS 좌표로 초기화하고 상삼각 쌍의 **원시 제곱 잔차 합**을 기록한다. 거리 안의 10⁻¹² 때문에 정확한 유클리드 거리와는 작은 차이가 있으며 300회 뒤 수렴을 보장하지 않는다. 기록값도 각 갱신 전이다. 여러 초기값·러닝레이트와 최종 stress를 비교하고 n<2의 빈 쌍이나 퇴화 배치를 별도로 처리해야 한다.

## 7. 비계량 MDS와 라이브러리 인터페이스

비계량 MDS는 입력 비유사도에 단조 회귀를 적용해 얻은 disparity f(d_ij)와 임베딩 거리를 맞춘다. 순위 관계를 중시하지만 저차원에서 모든 엄격한 순서가 보존된다는 제약을 반드시 만족시키는 것은 아니다. 동점 처리와 스케일 정규화도 알고리즘의 일부다.

```python
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# --- 피처로 하는 계량 MDS ---
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

```
Stress: 9775.5425
```

이 코드의 `MDS(metric=True)`는 stress 기반 계량 MDS다. `dissimilarity='euclidean'`은 X에서 거리를 계산하고 `'precomputed'`는 D를 직접 받는다. `normalized_stress='auto'`가 계량 MDS에서 반드시 정규화된 값을 반환한다는 뜻은 아니다. 설정과 지원 버전에 따라 `stress_`의 정의를 확인해야 하며, 보존된 출력만으로 정규화 방식이나 반복 수를 추정하지 않는다. [scikit-learn MDS 문서](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)를 참고한다.

## 8. 셰퍼드 다이어그램과 품질 판단

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

원래 거리를 가로축, 임베딩 거리를 세로축에 둔다. 계량 거리 보존이 완벽하면 y=x 위에 놓인다. 같은 크기의 전체 stress라도 가까운 쌍과 먼 쌍에서 왜곡이 다를 수 있다. 비계량 MDS에서는 대각선보다 적합한 단조 관계와의 잔차를 살핀다.

stress의 0.05나 0.1 같은 값을 모든 데이터에 적용하는 합격 기준으로 쓰지 않는다. 표본 수·목표 차원·웨이트·정규화와 입력 구조에 따라 해석이 달라진다. 같은 목적과 데이터에서 차원·초기값·대안 방법을 비교한다. 같은 점들의 이름만 재배열하는 것은 새로운 무작위 기준선을 만들지 않는다.

## 9. 상관 거리로 그리는 자산 지도

공통 관측 구간에서 분산이 양수인 수익률 벡터들을 중심화·단위 노름화하면 상관 ρ_ij는 그 내적이다. 따라서 다음 거리는 정규화 벡터의 유클리드 거리다.

$$d_{ij}=\sqrt{2(1-\rho_{ij})}.$$

완전 상관이면 다른 자산이라도 거리가 0이므로 자산 이름 집합에서는 일반적으로 유사거리다. 서로 다른 관측 구간으로 쌍별 상관을 계산하면 양의 준정부호성이 깨져 이 유클리드 해석을 바로 적용할 수 없다.

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

위 데이터는 시장·업종 요인을 흉내 낸 합성 수익률이다. 자산 근접성은 해당 표본의 상관 구조를 보여 줄 뿐 인과관계·미래 상관이나 수익 기회를 증명하지 않는다. 상수 열은 상관이 정의되지 않고, 반올림으로 1−ρ가 조금 음수가 되면 제곱근에서 NaN이 생길 수 있다. 대칭성·대각과 유효 범위를 확인한 뒤 적절한 수치 처리가 필요하다.

## 10. 비용과 새 표본

거리 저장은 O(n²)이고 밀집 고전 MDS의 전체 고유분해는 O(n³)다. 고정 목표 차원 k에서 계량 갱신의 쌍별 계산은 반복당 대략 O(n²k)이며 반복 수와 초기화 횟수도 비용에 들어간다. 비계량 방식에는 순위·단조 회귀의 비용이 추가된다.

여기 구현들은 기존 표본의 좌표를 적합하며 새 점의 `transform`을 제공하지 않는다. 고전 MDS의 표본 외 확장이나 랜드마크 방법, 별도 파라메터 모델은 가능하므로 MDS에서 새 점을 배치하는 것이 원리적으로 불가능한 것은 아니다. Isomap은 이웃 그래프의 최단 경로를 거리로 사용해 매니폴드의 측지 구조를 근사하지만 이웃 그래프가 올바르게 연결된다는 전제가 중요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
고전 MDS와 stress 기반 계량 MDS가 같은 결과를 주는 충분조건을 설명하라.

</div>

??? success "풀이"

    입력이 유클리드 거리이고 목표 차원이 이를 실현하는 데 충분하면 고전 MDS가 stress=0인 좌표를 준다. 계량 MDS도 전역 최소 0을 찾으면 회전·반사·이동을 제외한 같은 거리 배치를 얻는다. 목표 차원이 부족하면 그람 근사와 거리 stress라는 목적 차이 때문에 결과가 달라질 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
거리 보존과 t-SNE 시각화의 목적 차이를 설명하라.

</div>

??? success "풀이"

    MDS는 선택한 거리·그람 목적에 따른 관계를 근사하고 t-SNE는 확률적 이웃 관계를 강조한다. t-SNE의 군집 면적과 군집 간 거리를 원래 분산·거리로 읽지 않는다. 어느 방법도 모든 국소·전역 관계나 실제 군집 발견을 보장하지 않으며 목적에 맞는 지표가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이중 중심화에서 고전 MDS 좌표를 유도하고 무엇을 최적화하는지 설명하라.

</div>

??? success "풀이"

    제곱 유클리드 거리의 노름 항은 H1=0으로 사라져 B=−HD²H/2=X_cX_cᵀ다. B의 상위 양의 고윳값·고유벡터로 Y=Q_kΛ_k^(1/2)를 만든다. 차원을 줄일 때 이는 그람 행렬 strain 최소화이며 원래 거리 stress 최소화와 같다고 할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
t-SNE에서 두꺼운 꼬리의 저차원 분포를 사용하는 이유를 설명하라.

</div>

??? success "풀이"

    저차원에서 이웃과 더 먼 관계를 모두 표현할 공간이 부족한 crowding 문제를 완화하려는 선택이다. 먼 점을 더 멀리 배치할 수 있게 하지만 군집의 완벽한 분리나 전체 거리 보존을 보장하지는 않는다. KL 정규화로 각 쌍의 영향도 서로 연결된다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
유클리드 거리의 고전 MDS와 PCA 점수의 관계를 설명하라.

</div>

??? success "풀이"

    X_c=USVᵀ이면 B=X_cX_cᵀ=US²Uᵀ다. MDS 좌표는 U_kS_k로 PCA 점수와 같다. 단순 고윳값에서는 부호를 맞추어 비교할 수 있고 중복 공간에서는 직교 정렬이 필요하다. 특정 표본의 소수점 차이를 보편적인 정확도 수치로 제시할 필요는 없다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
좌표 없이 거리만 있을 때 MDS를 적용하며 확인할 사항을 설명하라.

</div>

??? success "풀이"

    비유사도 정의와 대칭성·비음수성, 누락과 표본 대응을 확인한다. 고전 MDS에서는 중심화 그람 스펙트럼으로 유클리드 표현 가능성을 살핀다. 유클리드가 아니면 양의 부분의 근사가 가능하지만 왜곡을 보고해야 한다. 편집 거리 등도 상황에 따라 유클리드일 수 있으므로 이름만으로 판정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
음의 고윳값을 발견했을 때 수치 오차와 실제 비유클리드 관계를 어떻게 구분하는가?

</div>

??? success "풀이"

    전체 스펙트럼 크기와 계산 정밀도에 상대적인 허용 오차를 사용한다. 상당한 음수이면 원래 D를 유클리드 좌표로 정확히 표현할 수 없다는 신호다. 양의 부분만 남겼을 때 왜곡과 버린 몫을 확인한다. 임의로 거리에 상수를 더하거나 계량 최적화를 한다고 원래 관계의 정확한 표현이 보장되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
원시 stress와 정규화 stress를 구분하라. normalized_stress=auto의 해석도 설명하라.

</div>

??? success "풀이"

    원시 stress는 거리 잔차 제곱의 합이다. 정규화에서는 입력 거리 제곱합을 쓸지 적합 거리 제곱합을 쓸지와 제곱근 여부를 명시해야 한다. auto는 계량·비계량 설정에 따라 동작하므로 항상 정규화된다고 볼 수 없다. 설치 버전의 반환 정의를 확인하고 다른 정의의 값을 같은 기준으로 비교하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
비계량 MDS가 순위를 보존한다는 말을 정확하게 설명하라.

</div>

??? success "풀이"

    입력 비유사도의 단조 변환인 disparity와 임베딩 거리 사이의 오차를 줄인다는 뜻이다. 저차원 좌표가 모든 엄격한 순위를 반드시 만족하는 것은 아니다. 동점, 누락 처리와 정규화 규칙이 중요하며 계량 방식과 숫자가 낮다는 이유만으로 더 좋은 표현이라 결론 내리지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
셰퍼드 다이어그램에서 확인할 항목과 적용 한계를 설명하라.

</div>

??? success "풀이"

    같은 척도의 계량 비교에서는 대각선 주변의 흩어짐, 거리 구간별 압축·확장과 이상 쌍을 살핀다. 비계량은 적합한 단조 관계를 기준으로 본다. t-SNE의 원래 거리 보존을 주목적으로 평가하면 설계 목표와 어긋날 수 있다. 그림 한 장만으로 실제 군집이나 미래 관계를 검증할 수 없다.



## 정리하며

고전 MDS는 이중 중심화한 그람 행렬을 근사하고 계량 MDS는 거리 stress를 줄인다. PCA와의 동등성, 음의 고윳값과 정규화 정의를 구분하면 결과를 일관되게 해석할 수 있다. 거리의 출처와 표본 외 확장, 계산 비용도 적용 목적에 맞추어 확인한다.
