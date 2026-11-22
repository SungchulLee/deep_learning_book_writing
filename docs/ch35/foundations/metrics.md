# 그래프 생성의 평가 지표

그래프 생성 모델을 평가하려면 생성한 그래프의 분포 $\{G_i^{\text{gen}}\}$를 참조 분포 $\{G_i^{\text{ref}}\}$와 견주어야 한다. FID가 표준 지표를 주는 이미지 생성과 달리, 그래프 평가는 구조, 통계, 도메인 특유의 성질을 담는 서로 보완하는 지표 여럿을 요구한다. 지표 하나로는 모자란다. 생성기가 차수 분포는 맞지만 군집 패턴이 틀린 그래프를 낼 수도, 그 반대일 수도 있다.

---

## 1. 그래프 통계

가장 기본적인 평가 방법은 그래프 수준 성질의 분포 통계를 계산해 생성 집합과 참조 집합을 견주는 것이다.

### 차수 분포

노드 $v$의 차수는 $d_v = \sum_{u} A_{vu}$이다. 차수 분포 $P(d)$는 근본이 되는 연결 패턴을 담는다. 그래프 집합에서는 모든 그래프를 모아 경험적 차수 분포를 계산한다:

$$
\hat{P}(d) = \frac{1}{\sum_i |\mathcal{V}_i|} \sum_{i} \sum_{v \in \mathcal{V}_i} \mathbf{1}[d_v = d]
$$

### 군집 계수 분포

노드 $v$의 국소 군집 계수는 $v$를 지나는 가능한 삼각형 가운데 실제로 있는 몫을 잰다:

$$
C_v = \frac{2 |\{(u,w) : u,w \in \mathcal{N}(v), (u,w) \in \mathcal{E}\}|}{d_v(d_v - 1)}
$$

노드에 걸친 $C_v$의 분포는 그래프의 국소 구조 밀도와 전이성(이웃끼리도 이어져 있는 정도)을 담는다.

### 궤도 개수

그래프 궤도 개수(orbit count)는 작은 부분 그래프 패턴(그래플릿)이 나타나는 횟수를 센다. 노드 $v$의 궤도 개수 $o_k(v)$는 노드가 5개 이하인 그래플릿의 $k$번째 궤도에 $v$가 몇 번 놓이는지 센다. 73차원 궤도 개수 분포는 촘촘한 구조 지문을 준다.

### 스펙트럼 통계

정규화 라플라시안의 고윳값 $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$이 전역 그래프 구조를 담는다:

- **스펙트럼 갭** $\lambda_2$: 연결성과 확장 성질(expansion)을 잰다
- **스펙트럼 분포**: 고윳값의 히스토그램이 그래프 계열의 성질을 나타낸다

---

## 2. 분포 비교 지표

### 최대 평균 불일치(MMD)

MMD는 재생 커널 힐베르트 공간(RKHS)에서 두 분포의 평균 임베딩 사이의 거리를 잰다. 그래프 통계 표본 $\{s_i^{\text{ref}}\}$와 $\{s_j^{\text{gen}}\}$가 주어질 때:

$$
\text{MMD}^2 = \frac{1}{m^2}\sum_{i,j} k(s_i^{\text{ref}}, s_j^{\text{ref}}) + \frac{1}{m'^2}\sum_{i,j} k(s_i^{\text{gen}}, s_j^{\text{gen}}) - \frac{2}{mm'}\sum_{i,j} k(s_i^{\text{ref}}, s_j^{\text{gen}})
$$

여기서 $k$는 커널 함수(보통 가우스 RBF나 총변동 커널)다. MMD가 낮을수록 분포가 잘 맞는다.

일반적인 그래프 생성 지표는 차수, 군집, 궤도 분포에 대해 MMD를 계산한다:

$$
\text{MMD}_{\text{total}} = \text{MMD}_{\text{degree}} + \text{MMD}_{\text{clustering}} + \text{MMD}_{\text{orbit}}
$$

### 어스 무버 거리(바서슈타인)

그래프 통계 분포 사이의 1차 바서슈타인 거리는 값들이 서로 얼마나 떨어져 있는지까지 반영해 분포를 비교한다 — 한 분포를 다른 분포로 옮기는 데 드는 최소 운송 비용이다:

$$
W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \int \|x - y\| \, d\gamma(x,y)
$$

히스토그램으로 나타낸 이산 분포에서는 이것이 선형 계획 문제가 되고, 1차원 분포라면 누적 분포 함수 사이의 $L^1$ 거리를 계산하는 일로 줄어든다.

### 프레셰 그래프 거리(FGD)

이미지의 FID와 비슷하게 FGD는 사전 트레인된 그래프 뉴럴 네트워크 인코더 $f$로 그래프를 피처 공간에 임베딩한 뒤, 두 집합의 피처를 각각 가우스 분포로 근사해 그 사이의 프레셰 거리를 계산한다:

$$
\text{FGD} = \|\boldsymbol{\mu}_{\text{ref}} - \boldsymbol{\mu}_{\text{gen}}\|^2 + \text{Tr}\left(\boldsymbol{\Sigma}_{\text{ref}} + \boldsymbol{\Sigma}_{\text{gen}} - 2(\boldsymbol{\Sigma}_{\text{ref}}\boldsymbol{\Sigma}_{\text{gen}})^{1/2}\right)
$$

여기서 $\boldsymbol{\mu}$와 $\boldsymbol{\Sigma}$는 집합마다 계산한 $f(\mathcal{G})$의 평균과 공분산이다.

---

## 3. 유효성과 고유성

### 유효성 비율

도메인 제약을 만족하는 생성한 그래프의 몫:

$$
\text{Validity} = \frac{|\{G_i^{\text{gen}} : \text{valid}(G_i^{\text{gen}})\}|}{|\{G_i^{\text{gen}}\}|}
$$

도메인 특유의 유효성에는 다음이 있다:

- **분자**: 올바른 원자가, 전하 띤 조각 없음, 합성 가능함
- **금융**: 연결 요소, 차수 제한, 웨이트 제약

### 고유성과 참신성

$$
\text{Uniqueness} = \frac{|\text{unique}(\{G_i^{\text{gen}}\})|}{|\{G_i^{\text{gen}}\}|}
$$

$$
\text{Novelty} = \frac{|\{G_i^{\text{gen}} : G_i^{\text{gen}} \notin \{G_j^{\text{ref}}\}\}|}{|\{G_i^{\text{gen}}\}|}
$$

참신성을 재려면 그래프 동형 시험이 필요한데 이는 계산이 비싸다. 실제로는 해시 기반 근사(보기로 바이스파일러-레만 해시)를 쓴다.

---

## 4. 구현: 평가 꾸러미

```python
"""
그래프 만들기의 따지기 잣대.
"""
import torch
import numpy as np
from typing import Optional
from scipy.stats import wasserstein_distance
from collections import Counter

# 이 쪽이 싣는 수가 돌릴 때마다 달라지지 않게 씨앗을 박는다.
torch.manual_seed(0)
np.random.seed(0)


def degree_distribution(adj: torch.Tensor) -> np.ndarray:
    """차수 분포를 고르게 맞춘 막대 그림으로 셈한다."""
    degrees = adj.sum(dim=1).long().cpu().numpy()
    max_deg = max(degrees.max(), 1)
    hist = np.zeros(max_deg + 1)
    for d in degrees:
        hist[d] += 1
    return hist / hist.sum()

def clustering_coefficients(adj: torch.Tensor) -> np.ndarray:
    """모든 마디의 그 자리 뭉침 계수를 셈한다."""
    adj_np = adj.cpu().numpy()
    n = adj_np.shape[0]
    coeffs = np.zeros(n)

    for v in range(n):
        neighbors = np.where(adj_np[v] > 0)[0]
        k = len(neighbors)
        if k < 2:
            coeffs[v] = 0.0
            continue
        # 이웃 사이의 변을 센다
        subgraph = adj_np[np.ix_(neighbors, neighbors)]
        triangles = subgraph.sum() / 2  # 변마다 두 번 센다
        coeffs[v] = 2 * triangles / (k * (k - 1))

    return coeffs

def spectral_distribution(
    adj: torch.Tensor, num_bins: int = 50
) -> np.ndarray:
    """고르게 맞춘 라플라스의 고윳값 분포를 셈한다."""
    degree = adj.sum(dim=1)
    # 외톨이 절를 다룬다
    degree_inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    degree_inv_sqrt[mask] = 1.0 / torch.sqrt(degree[mask])

    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    L_norm = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt

    eigenvalues = torch.linalg.eigvalsh(L_norm).cpu().numpy()
    hist, _ = np.histogram(eigenvalues, bins=num_bins, range=(0, 2), density=True)
    return hist / (hist.sum() + 1e-10)

def gaussian_rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """막대 그림 벡터 둘 사이의 가우스 RBF 커널."""
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))

def compute_mmd(
    samples_ref: list[np.ndarray],
    samples_gen: list[np.ndarray],
    kernel_fn=None,
    sigma: float = 1.0,
) -> float:
    """
    분포 모임 둘 사이의 최대 평균 어긋남을 셈한다.
    
    인수:
        samples_ref: 잣대 그래프에서 온 막대 그림 벡터 목록
        samples_gen: 만든 그래프에서 온 막대 그림 벡터 목록
        kernel_fn: 커널 함수(기본값: 가우스 RBF)
        sigma: 커널 띠너비
        
    반환값:
        MMD^2 value
    """
    if kernel_fn is None:
        kernel_fn = lambda x, y: gaussian_rbf_kernel(x, y, sigma)

    # 막대 그림을 같은 길이로 채운다
    max_len = max(
        max(len(s) for s in samples_ref),
        max(len(s) for s in samples_gen),
    )
    ref = [np.pad(s, (0, max_len - len(s))) for s in samples_ref]
    gen = [np.pad(s, (0, max_len - len(s))) for s in samples_gen]

    m, mp = len(ref), len(gen)

    # K(잣대, 잣대)
    k_rr = sum(kernel_fn(ref[i], ref[j]) for i in range(m) for j in range(m)) / (m * m)
    # K(만든 것, 만든 것)
    k_gg = sum(kernel_fn(gen[i], gen[j]) for i in range(mp) for j in range(mp)) / (mp * mp)
    # K(잣대, 만든 것)
    k_rg = sum(kernel_fn(ref[i], gen[j]) for i in range(m) for j in range(mp)) / (m * mp)

    return float(k_rr + k_gg - 2 * k_rg)

def evaluate_generation(
    adj_ref: list[torch.Tensor],
    adj_gen: list[torch.Tensor],
) -> dict[str, float]:
    """
    만든 그래프를 잣대와 맞대어 두루 따진다.
    
    반환값:
        자 이름을 값으로 보내는 사전
    """
    results = {}

    # 차수 MMD
    deg_ref = [degree_distribution(a) for a in adj_ref]
    deg_gen = [degree_distribution(a) for a in adj_gen]
    results["mmd_degree"] = compute_mmd(deg_ref, deg_gen)

    # 뭉침 MMD
    clust_ref = [np.histogram(clustering_coefficients(a), bins=20, range=(0, 1), density=True)[0]
                 for a in adj_ref]
    clust_gen = [np.histogram(clustering_coefficients(a), bins=20, range=(0, 1), density=True)[0]
                 for a in adj_gen]
    results["mmd_clustering"] = compute_mmd(clust_ref, clust_gen)

    # 스펙트럼 MMD
    spec_ref = [spectral_distribution(a) for a in adj_ref]
    spec_gen = [spectral_distribution(a) for a in adj_gen]
    results["mmd_spectral"] = compute_mmd(spec_ref, spec_gen)

    # 기본 그래프 통계
    def graph_stats(adj_list):
        nodes = [a.size(0) for a in adj_list]
        edges = [a.sum().item() / 2 for a in adj_list]
        densities = [e / (n * (n - 1) / 2) if n > 1 else 0
                     for n, e in zip(nodes, edges)]
        return {
            "avg_nodes": np.mean(nodes),
            "avg_edges": np.mean(edges),
            "avg_density": np.mean(densities),
        }

    ref_stats = graph_stats(adj_ref)
    gen_stats = graph_stats(adj_gen)

    results["ref_avg_nodes"] = ref_stats["avg_nodes"]
    results["gen_avg_nodes"] = gen_stats["avg_nodes"]
    results["ref_avg_density"] = ref_stats["avg_density"]
    results["gen_avg_density"] = gen_stats["avg_density"]

    # 차수 분포의 바서슈타인 거리
    all_deg_ref = np.concatenate([a.sum(1).cpu().numpy() for a in adj_ref])
    all_deg_gen = np.concatenate([a.sum(1).cpu().numpy() for a in adj_gen])
    results["wasserstein_degree"] = wasserstein_distance(all_deg_ref, all_deg_gen)

    return results

if __name__ == "__main__":
    # 잣대 그래프를 만든다(에르되시-레니)
    n, p = 20, 0.15
    ref_graphs = []
    for _ in range(50):
        adj = (torch.rand(n, n) < p).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        ref_graphs.append(adj)

    # "만든" 그래프 — 비슷한 분포
    gen_good = []
    for _ in range(50):
        adj = (torch.rand(n, n) < p).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        gen_good.append(adj)

    # "만든" 그래프 — 다른 분포(더 빽빽함)
    gen_bad = []
    for _ in range(50):
        adj = (torch.rand(n, n) < 0.5).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        gen_bad.append(adj)

    print("=== Good Generator (same distribution) ===")
    results_good = evaluate_generation(ref_graphs, gen_good)
    for k, v in results_good.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== Bad Generator (different distribution) ===")
    results_bad = evaluate_generation(ref_graphs, gen_bad)
    for k, v in results_bad.items():
        print(f"  {k}: {v:.6f}")
```

**출력:**

```
=== Good Generator (same distribution) ===
  mmd_degree: 0.001281
  mmd_clustering: 0.042443
  mmd_spectral: 0.001371
  ref_avg_nodes: 20.000000
  gen_avg_nodes: 20.000000
  ref_avg_density: 0.154105
  gen_avg_density: 0.154632
  wasserstein_degree: 0.080000

=== Bad Generator (different distribution) ===
  mmd_degree: 0.273467
  mmd_clustering: 0.058539
  mmd_spectral: 0.024259
  ref_avg_nodes: 20.000000
  gen_avg_nodes: 20.000000
  ref_avg_density: 0.154105
  gen_avg_density: 0.508842
  wasserstein_degree: 6.740000
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
순열 불변성 문제 때문에 그래프 생성이 이미지 생성보다 왜 근본에서 더 어려운지 밝혀라. 레이블이 붙은 노드 $n$개의 그래프에는 같은 뜻의 인접 행렬 표현이 몇 개 있는가?

</div>

??? success "풀이"
    노드 $n$개의 그래프는 서로 다른 인접 행렬 최대 $n!$개로 나타낼 수 있다(노드 레이블의 순열마다 하나, 자기동형 사상이 있으면 그보다 적다). $n = 10$이면 같은 뜻의 표현이 최대 $10! = 3{,}628{,}800$개다. 생성 모델은 다음 가운데 하나를 해야 한다. (1) 이 겹침을 줄이려 표준 순서를 배운다, (2) 순열 불변인 로스 함수를 쓴다(보기로 그래프 매칭), (3) 본래 불변인 표현 위에서 작동한다(보기로 스펙트럼 피처). 이미지 생성은 픽셀에 고정된 위치 순서가 있어 이 문제를 겪지 않는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
확장성, 품질, 유효성 제약을 지키는 능력의 면에서 그래프 생성의 자기회귀 길과 원샷 생성 길을 비교하여라.

</div>

??? success "풀이"
    자기회귀 방법은 노드와 엣지를 차례로 만들어 스텝마다 국소 제약(보기로 원자가)을 자연스럽게 지키지만, 엣지마다 결정을 내려야 해서 스텝 수가 $O(n^2)$으로 커진다. 원샷 생성 방법은 인접 행렬 전체를 한꺼번에 만들어 병렬 계산으로 확장하기 쉽지만 이산적인 구조와 전역 제약에 애를 먹는다. 자기회귀 방법은 작은 크기($n < 100$)에서 대개 더 좋은 그래프를 내지만 큰 그래프에서는 느려진다. 순열 불변인 로스를 쓰는 원샷 생성 방법은 더 빠르지만 매칭 문제를 조심스레 다루어야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
자기회귀 인수 분해 $p(G) = \prod_{i=1}^n p(\text{node}_i | \text{nodes}_{<i}) \prod_{j<i} p(e_{ij} | \text{node}_i, \text{nodes}_{<i})$ 아래에서 그래프의 우도를 이끌어 내어라.

</div>

??? success "풀이"
    고정된 노드 순서 $\pi$ 아래에서 결합 확률은 $p(G | \pi) = \prod_{i=1}^n p(v_{\pi(i)} | G_{\pi(<i)}) \prod_{j < i} p(e_{\pi(i),\pi(j)} | v_{\pi(i)}, G_{\pi(<i)})$으로 인수 분해된다. 여기서 $G_{\pi(<i)}$은 앞서 만든 노드의 부분 그래프다. 주변 우도 $p(G) = \frac{1}{n!}\sum_\pi p(G | \pi)$은 모든 순서를 더하지만 다룰 수 없다. 실제로는 표준 순서(보기로 너비 우선)를 써서 우도를 그 순서에 의존하는 것으로 만든다. 순서의 품질이 생성 품질에 영향을 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
분포 지표를 넘어 화학적 유효성과 약물 유사성을 재는, 생성한 분자 그래프의 평가 규약을 내놓아라.

</div>

??? success "풀이"
    분포 지표(차수 분포, 군집 계수)를 넘어 다음을 평가한다. (1) 화학적 유효성 -- 원자가 제약을 만족하는 분자의 몫(RDKit으로 확인), (2) 고유성 -- 서로 다른 유효한 분자의 몫, (3) 참신성 -- 트레인 집합에 없는 몫, (4) 약물 유사성 -- QED 점수, 리핀스키의 다섯 규칙 준수, (5) 합성 용이성 -- 합성이 얼마나 쉬운지 나타내는 SA 점수, (6) 성질 최적화 -- 바란 타깃 성질과 얻은 성질의 상관. 여러 번 만들어 얻은 신뢰 구간과 함께 모든 지표를 알린다. $\square$

## 정리하며

이 절에서는 그래프 통계, 분포 비교 지표, 유효성과 고유성, 평가 꾸러미의 구현을 살펴보았다.
