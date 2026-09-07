# 그래프 만들기의 따지기 잣대
## 개요

그래프 만들기 모델을 따지려면 만들어 낸 그래프의 분포 $\{G_i^{\text{gen}}\}$을 잣대 분포 $\{G_i^{\text{ref}}\}$과 견주어야 한다. FID이 표준 잣대를 주는 그림 만들기와 달리, 그래프 따지기는 얼개, 통계, 마당 특유의 성질을 담는 서로 보완하는 잣대 여럿을 요구한다. 잣대 하나로는 모자란다. 만들개가 차수 분포는 맞지만 뭉침 무늬가 틀린 그래프를 낼 수도, 그 반대일 수도 있다.

## 그래프 통계

으뜸 따지기 길은 그래프 켜 성질의 분포 통계를 셈해 만든 모임과 잣대 모임을 견준다.

### 차수 분포

마디 $v$의 차수는 $d_v = \sum_{u} A_{vu}$이다. 차수 분포 $P(d)$은 근본이 되는 이어짐 무늬를 담는다. 그래프 모임에서는 모든 그래프를 모아 겪어 얻은 차수 분포를 셈한다:

$$
\hat{P}(d) = \frac{1}{\sum_i |\mathcal{V}_i|} \sum_{i} \sum_{v \in \mathcal{V}_i} \mathbf{1}[d_v = d]
$$

### 뭉침 계수 분포

마디 $v$의 그 자리 뭉침 계수는 $v$을 지나는 가능한 삼각형 가운데 실제로 있는 몫을 잰다:

$$
C_v = \frac{2 |\{(u,w) : u,w \in \mathcal{N}(v), (u,w) \in \mathcal{E}\}|}{d_v(d_v - 1)}
$$

마디에 걸친 $C_v$의 분포는 그래프의 그 자리 얼개 빽빽함과 옮김성을 담는다.

### 궤도 세기

(로바스 틀에서 온) 그래프 궤도는 작은 부분 그래프 무늬(그래플릿)가 나타나는 횟수를 센다. 마디 $v$에서 궤도 셈 $o_k(v)$은 마디가 5개 이하인 그래플릿의 $k$번째 궤도에 $v$이 몇 번 끼는지 센다. 73차원 궤도 셈 분포는 촘촘한 얼개 지문을 준다.

### 스펙트럼 통계

고르게 맞춘 라플라스의 고윳값 $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$이 온 자리 그래프 얼개를 담는다:

- **스펙트럼 틈** $\lambda_2$: 이어짐과 넓힘을 잰다
- **스펙트럼 분포**: 고윳값의 막대 그림이 그래프 무리의 성질을 나타낸다

## 분포 견주기 잣대

### 최대 평균 어긋남(MMD)

MMD은 되살리는 알맹이 힐베르트 공간에서 두 분포 사이의 거리를 잰다. 그래프 통계 표본 $\{s_i^{\text{ref}}\}$과 $\{s_j^{\text{gen}}\}$이 주어질 때:

$$
\text{MMD}^2 = \frac{1}{m^2}\sum_{i,j} k(s_i^{\text{ref}}, s_j^{\text{ref}}) + \frac{1}{m'^2}\sum_{i,j} k(s_i^{\text{gen}}, s_j^{\text{gen}}) - \frac{2}{mm'}\sum_{i,j} k(s_i^{\text{ref}}, s_j^{\text{gen}})
$$

여기서 $k$은 알맹이 함수(보통 가우스 RBF이나 온 흔들림)이다. MMD이 낮을수록 분포가 잘 맞는다.

여느 그래프 만들기 잣대는 차수, 뭉침, 궤도 분포에 대해 MMD을 셈한다:

$$
\text{MMD}_{\text{total}} = \text{MMD}_{\text{degree}} + \text{MMD}_{\text{clustering}} + \text{MMD}_{\text{orbit}}
$$

### 흙 나르기 거리(바서슈타인)

그래프 통계 분포 사이의 1차 바서슈타인 거리가 거리를 살피는 견주기를 준다:

$$
W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \int \|x - y\| \, d\gamma(x,y)
$$

막대 그림으로 나타낸 띄엄띄엄한 분포에서는 이것이 선형 계획을 푸는 일, 또는 같은 뜻으로 누적 분포 함수 사이의 $L^1$ 거리를 셈하는 일로 줄어든다.

### 프레셰 그래프 거리(FGD)

그림의 FID과 비슷하게 FGD은 미리 익힌 그래프 신경망 담개 $f$으로 그래프를 특징 자리에 박아 넣은 뒤 가우스 어림 사이의 프레셰 거리를 셈한다:

$$
\text{FGD} = \|\boldsymbol{\mu}_{\text{ref}} - \boldsymbol{\mu}_{\text{gen}}\|^2 + \text{Tr}\left(\boldsymbol{\Sigma}_{\text{ref}} + \boldsymbol{\Sigma}_{\text{gen}} - 2(\boldsymbol{\Sigma}_{\text{ref}}\boldsymbol{\Sigma}_{\text{gen}})^{1/2}\right)
$$

여기서 $\boldsymbol{\mu}$과 $\boldsymbol{\Sigma}$은 모임마다 셈한 $f(\mathcal{G})$의 평균과 함께 흩어짐이다.

## 올바름과 하나뿐임

### 올바름 비율

마당 매임을 만족하는 만들어 낸 그래프의 몫:

$$
\text{Validity} = \frac{|\{G_i^{\text{gen}} : \text{valid}(G_i^{\text{gen}})\}|}{|\{G_i^{\text{gen}}\}|}
$$

마당 특유의 올바름에는 다음이 있다:

- **분자**: 올바른 원자가, 전하 띤 조각 없음, 합성 가능함
- **금융**: 이어진 조각, 차수 가둠, 무게 매임

### 하나뿐임과 새로움

$$
\text{Uniqueness} = \frac{|\text{unique}(\{G_i^{\text{gen}}\})|}{|\{G_i^{\text{gen}}\}|}
$$

$$
\text{Novelty} = \frac{|\{G_i^{\text{gen}} : G_i^{\text{gen}} \notin \{G_j^{\text{ref}}\}\}|}{|\{G_i^{\text{gen}}\}|}
$$

새로움을 재려면 그래프 같은 꼴 시험이 필요한데 이는 셈이 비싸다. 실제로는 흩기 바탕 어림(보기로 바이스파일러-레만 흩기값)을 쓴다.

## 짜기: 따지기 꾸러미

```python
"""
그래프 만들기의 따지기 잣대.
"""
import torch
import numpy as np
from typing import Optional
from scipy.stats import wasserstein_distance
from collections import Counter


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
    # 외톨이 마디를 다룬다
    degree_inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    degree_inv_sqrt[mask] = 1.0 / torch.sqrt(degree[mask])

    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    L_norm = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt

    eigenvalues = torch.linalg.eigvalsh(L_norm).cpu().numpy()
    hist, _ = np.histogram(eigenvalues, bins=num_bins, range=(0, 2), density=True)
    return hist / (hist.sum() + 1e-10)


def gaussian_rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """막대 그림 벡터 둘 사이의 가우스 RBF 알맹이."""
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
        kernel_fn: 알맹이 함수(기본값: 가우스 RBF)
        sigma: 알맹이 띠너비
        
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

## 연습문제

**연습문제 1.**
자리바꿈에 안 바뀜 문제 때문에 그래프 만들기가 그림 만들기보다 왜 근본에서 더 어려운지 밝혀라. 이름표가 붙은 마디 $n$개의 그래프에는 같은 뜻의 이웃 행렬 나타냄이 몇 개 있는가?

??? success "연습문제 1 풀이"
    마디 $n$개의 그래프는 서로 다른 이웃 행렬 $n!$개로 나타낼 수 있다(마디 이름표의 자리바꿈마다 하나). $n = 10$이면 같은 뜻의 나타냄이 $10! = 3{,}628{,}800$개다. 짓는 모델은 다음 가운데 하나를 해야 한다. (1) 이 겹침을 줄이려 표준 차례를 배운다, (2) 자리바꿈에 안 바뀌는 손실 함수를 쓴다(보기로 그래프 짝짓기), (3) 본디 안 바뀌는 나타냄에서 돈다(보기로 스펙트럼 특징). 그림 만들기는 픽셀에 붙박인 자리 차례가 있어 이 문제를 겪지 않는다. $\square$

---

**연습문제 2.**
커지기, 품질, 올바름 매임을 지킬 수 있음의 면에서 그래프 만들기의 자기 되돌이 길과 한 번에 만들기 길을 견주어라.

??? success "연습문제 2 풀이"
    자기 되돌이 방법은 마디와 변을 차례로 만들어 걸음마다 그 자리 매임(보기로 원자가)을 자연스럽게 지키지만, 이웃 결정이 늘어나 $O(n^2)$으로 커진다. 한 번에 만들기 방법은 이웃 행렬 전체를 한꺼번에 만들어 나란한 셈에는 더 잘 커지지만 띄엄띄엄한 얼개와 온 자리 매임에 애를 먹는다. 자기 되돌이 방법은 작은 크기($n < 100$)에서 대개 더 좋은 그래프를 내지만 큰 그래프에서는 느려진다. 자리바꿈에 안 바뀌는 손실을 쓰는 한 번에 만들기 방법은 더 빠르지만 짝짓기 문제를 조심스레 다루어야 한다. $\square$

---

**연습문제 3.**
자기 되돌이 인수 분해 $p(G) = \prod_{i=1}^n p(\text{node}_i | \text{nodes}_{<i}) \prod_{j<i} p(e_{ij} | \text{node}_i, \text{nodes}_{<i})$ 아래에서 그래프의 가능도를 이끌어 내어라.

??? success "연습문제 3 풀이"
    붙박인 마디 차례 $\pi$ 아래에서 함께 확률은 $p(G | \pi) = \prod_{i=1}^n p(v_{\pi(i)} | G_{\pi(<i)}) \prod_{j < i} p(e_{\pi(i),\pi(j)} | v_{\pi(i)}, G_{\pi(<i)})$으로 인수 분해된다. 여기서 $G_{\pi(<i)}$은 앞서 만든 마디의 부분 그래프다. 주변 가능도 $p(G) = \frac{1}{n!}\sum_\pi p(G | \pi)$은 모든 차례를 더하지만 다룰 수 없다. 실제로는 표준 차례(보기로 너비 우선)를 써서 가능도를 그 차례에 매인 것으로 만든다. 차례의 품질이 만들기 품질에 영향을 준다. $\square$

---

**연습문제 4.**
분포 잣대를 넘어 화학의 올바름과 약다움을 재는, 만들어 낸 분자 그래프의 따지기 규약을 내놓아라.

??? success "연습문제 4 풀이"
    분포 잣대(차수 분포, 뭉침 계수)를 넘어 다음을 따진다. (1) 화학의 올바름 -- 원자가 매임을 만족하는 분자의 몫(RDKit으로 확인), (2) 하나뿐임 -- 서로 다른 올바른 분자의 몫, (3) 새로움 -- 익히기 모임에 없는 몫, (4) 약다움 -- QED 점수, 리핀스키의 다섯 규칙 지킴, (5) 만들기 쉬움 -- 합성이 얼마나 쉬운지 나타내는 SA 점수, (6) 성질 가장 좋게 하기 -- 바란 과녁 성질과 얻은 성질의 얽힘. 여러 번 만들어 얻은 믿음 구간과 함께 모든 잣대를 알린다. $\square$
