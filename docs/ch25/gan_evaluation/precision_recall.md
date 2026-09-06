# 만들어 내는 모델의 정밀도와 재현율
## 개요

가르기에서 만들어 내는 모델로 옮겨 온 정밀도와 재현율 잣대는 FID만으로는 담을 수 없는 만들어 내기 품질의 결을 알려 준다. 이 잣대는 **충실함**(만든 표본이 그럴듯한가?)과 **다양함**(모델이 자료 분포의 봉우리를 모두 덮는가?)을 따로 잰다.

!!! info "배움 목표"
    이 절을 마치면 다음을 할 수 있게 된다.
    
    - FID 너머로 정밀도와 재현율이 필요한 까닭을 이해한다
    - k번째 가장 가까운 이웃에 바탕한 정밀도와 재현율 셈하기를 짠다
    - 만들어 내는 모델에서 정밀도와 재현율의 맞바꿈을 풀이한다
    - 나아진 정밀도와 재현율 잣대(IPR)를 쓴다
    - 이 잣대로 봉우리 무너짐과 품질 문제를 짚어낸다

## 까닭: 왜 FID 너머인가?

### 숫자 하나짜리 잣대의 한계

FID은 서로 다른 두 가지 잘못됨을 뒤섞는다.

| 잘못됨 | 설명 | FID의 반응 |
|--------------|-------------|--------------|
| **낮은 충실함** | 만든 표본이 가짜처럼 보인다 | FID이 는다 |
| **낮은 다양함** | 모델이 봉우리의 일부만 만든다 | FID이 는다 |

FID 값 하나로는 이 바탕부터 다른 문제를 가려낼 수 없다.

```
Model A: FID = 50 (high-quality but mode collapsed)
Model B: FID = 50 (diverse but low-quality)

FID은 같지만 문제는 아주 다르다!
```

### 정밀도와 재현율로 나누기

**정밀도(충실함)**: 만든 표본 가운데 그럴듯한 것의 몫은 얼마인가?

$$
\text{Precision} = \frac{\text{Generated samples that look real}}{\text{All generated samples}}
$$

**재현율(덮기)**: 모델이 실제 자료 봉우리의 얼마를 덮는가?

$$
\text{Recall} = \frac{\text{Real data covered by generator}}{\text{All real data}}
$$

## 수학적 바탕

### 다양체에 바탕한 풀이

실제 자료와 만든 자료는 특징 공간의 다양체 위에 놓인다.

- **실제 다양체** $\mathcal{M}_r$: "참" 자료 분포의 받침
- **만든 다양체** $\mathcal{M}_g$: 만들개가 내놓는 분포의 받침

**정밀도**가 재는 것: $P(\mathcal{M}_g \cap \mathcal{M}_r | \mathcal{M}_g)$

**재현율**이 재는 것: $P(\mathcal{M}_g \cap \mathcal{M}_r | \mathcal{M}_r)$

### k번째 가장 가까운 이웃 방식

참 다양체에 다다를 수 없으므로 k번째 가장 가까운 이웃으로 어림한다.

**핵심 생각**: 점 $x$은 그 다양체 안의 $k$번째 가장 가까운 이웃이 "충분히 가까우면" 그 다양체에 든다.

다음이 주어졌다고 하자.

- 실제 특징: $X_r = \{x_r^1, ..., x_r^{N_r}\}$
- 만든 특징: $X_g = \{x_g^1, ..., x_g^{N_g}\}$

점마다 다음을 셈한다.

$$
d_k(x, X) = \|x - \text{NN}_k(x, X)\|_2
$$

여기서 $\text{NN}_k(x, X)$은 모임 $X$ 안에서 $x$의 $k$번째 가장 가까운 이웃이다.

### 정밀도 뜻매김

만든 표본 $x_g$은 어떤 실제 표본의 k번째 가장 가까운 이웃 공 안에 들면 "실제 같다"고 본다.

$$
\text{Precision} = \frac{1}{N_g} \sum_{i=1}^{N_g} \mathbb{1}\left[d_k(x_g^i, X_r) \leq d_k(\text{NN}_k(x_g^i, X_r), X_r)\right]
$$

**간추리면**: 가짜 표본이 실제 표본에 가까우면 그럴듯하다.

### 재현율 뜻매김

마찬가지로 재현율은 실제 표본 가운데 가까이에 만든 표본이 있는 것이 얼마나 되는지 잰다.

$$
\text{Recall} = \frac{1}{N_r} \sum_{i=1}^{N_r} \mathbb{1}\left[d_k(x_r^i, X_g) \leq d_k(\text{NN}_k(x_r^i, X_g), X_g)\right]
$$

**간추리면**: 재현율은 실제 자료 봉우리가 만든 표본으로 덮이는지를 잰다.

## PyTorch 구현

### 기본 구현

```python
import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist


class PrecisionRecallCalculator:
    """
    만들어 내는 모델의 정밀도와 재현율 잣대.
    
    Based on Sajjadi et al. (2018) and Kynkäänniemi et al. (2019).
    
    정밀도는 충실함을 잰다. 곧 만든 표본이 그럴듯한가?
    재현율은 다양함을 잰다. 곧 모델이 모든 봉우리를 덮는가?
    """
    
    def __init__(self,
                 k: int = 3,
                 row_batch_size: int = 10000,
                 col_batch_size: int = 10000):
        """
        셈개를 첫자리매김한다.
        
        인수:
            k: k번째 가장 가까운 이웃의 이웃 수
            row_batch_size: Batch size for distance computation (rows)
            col_batch_size: Batch size for distance computation (cols)
        """
        self.k = k
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
    
    def _batch_pairwise_distances(self,
                                  X: np.ndarray,
                                  Y: np.ndarray) -> np.ndarray:
        """
        묶음 단위로 짝마다 유클리드 거리를 셈한다.
        
        인수:
            X: First set of points [N, D]
            Y: Second set of points [M, D]
            
        반환값:
            Distance matrix [N, M]
        """
        n = len(X)
        m = len(Y)
        distances = np.zeros((n, m), dtype=np.float32)
        
        for i in range(0, n, self.row_batch_size):
            end_i = min(i + self.row_batch_size, n)
            
            for j in range(0, m, self.col_batch_size):
                end_j = min(j + self.col_batch_size, m)
                
                distances[i:end_i, j:end_j] = cdist(
                    X[i:end_i], Y[j:end_j], metric='euclidean'
                )
        
        return distances
    
    def _compute_knn_distances(self,
                              X: np.ndarray,
                              Y: np.ndarray) -> np.ndarray:
        """
        X에서 Y까지 k번째 가장 가까운 이웃 거리를 셈한다.
        
        인수:
            X: Query points [N, D]
            Y: Reference points [M, D]
            
        반환값:
            k-th NN distances for each point in X [N]
        """
        distances = self._batch_pairwise_distances(X, Y)
        
        # 정렬해 k번째로 작은 거리를 얻는다
        # 참고: X==Y이면 k=0은 그 점 자신이다
        kth_distances = np.partition(distances, self.k, axis=1)[:, self.k]
        
        return kth_distances
    
    def compute_precision_recall(self,
                                real_features: np.ndarray,
                                generated_features: np.ndarray) -> Tuple[float, float]:
        """
        정밀도와 재현율을 셈한다.
        
        알고리즘:
        1. 실제 자료의 k번째 이웃 공을 셈한다
        2. 정밀도: 실제 k번째 이웃 공에 드는 만든 표본의 몫
        3. 만든 자료의 k번째 이웃 공을 셈한다
        4. 재현율: 만든 k번째 이웃 공에 드는 실제 표본의 몫
        
        인수:
            real_features: Features from real images [N_r, D]
            generated_features: Features from generated images [N_g, D]
            
        반환값:
            Tuple of (precision, recall)
        """
        print(f"Computing precision/recall with k={self.k}")
        print(f"  Real samples: {len(real_features)}")
        print(f"  Generated samples: {len(generated_features)}")
        
        # 실제 자료의 다양체 반지름을 셈한다
        # (실제 자료 안에서 k번째 가장 가까운 이웃까지의 거리)
        real_nn_distances = self._compute_knn_distances(real_features, real_features)
        
        # 만든 것에서 실제까지의 거리를 셈한다
        gen_to_real_distances = self._compute_knn_distances(generated_features, real_features)
        
        # 정밀도: 실제 다양체 안의 만든 표본
        # 만든 표본마다 가장 가까운 실제 이웃이
        # 그 실제 표본의 k번째 이웃 공 안에 있는지
        distances_gen_to_real = self._batch_pairwise_distances(
            generated_features, real_features
        )
        nearest_real_idx = np.argmin(distances_gen_to_real, axis=1)
        nearest_real_dist = distances_gen_to_real[np.arange(len(generated_features)), nearest_real_idx]
        
        # 다양체 안에 있는지 살핀다
        precision_mask = nearest_real_dist <= real_nn_distances[nearest_real_idx]
        precision = np.mean(precision_mask)
        
        # 만든 자료의 다양체 반지름을 셈한다
        gen_nn_distances = self._compute_knn_distances(generated_features, generated_features)
        
        # 재현율: 만든 다양체 안의 실제 표본
        distances_real_to_gen = self._batch_pairwise_distances(
            real_features, generated_features
        )
        nearest_gen_idx = np.argmin(distances_real_to_gen, axis=1)
        nearest_gen_dist = distances_real_to_gen[np.arange(len(real_features)), nearest_gen_idx]
        
        # 다양체 안에 있는지 살핀다
        recall_mask = nearest_gen_dist <= gen_nn_distances[nearest_gen_idx]
        recall = np.mean(recall_mask)
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return float(precision), float(recall)


class ImprovedPrecisionRecall:
    """
    Improved Precision and Recall (IPR) from Kynkäänniemi et al. (2019).
    
    본디 것보다 나아진 핵심:
    1. 초구면에 바탕한 다양체 어림을 쓴다
    2. 동떨어진 값에 더 튼튼하다
    3. 성긴 자리를 더 잘 다룬다
    """
    
    def __init__(self,
                 k: int = 3,
                 row_batch_size: int = 10000,
                 col_batch_size: int = 10000):
        """
        나아진 정밀도와 재현율 셈개를 첫자리매김한다.
        
        인수:
            k: 다양체 어림의 이웃 수
            row_batch_size: 거리 셈하기의 묶음 크기
            col_batch_size: 거리 셈하기의 묶음 크기
        """
        self.k = k
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
    
    def _batch_pairwise_distances(self,
                                  X: np.ndarray,
                                  Y: np.ndarray) -> np.ndarray:
        """묶음 단위로 짝마다 거리를 셈한다."""
        n = len(X)
        m = len(Y)
        distances = np.zeros((n, m), dtype=np.float32)
        
        for i in range(0, n, self.row_batch_size):
            end_i = min(i + self.row_batch_size, n)
            
            for j in range(0, m, self.col_batch_size):
                end_j = min(j + self.col_batch_size, m)
                
                distances[i:end_i, j:end_j] = cdist(
                    X[i:end_i], Y[j:end_j], metric='euclidean'
                )
        
        return distances
    
    def _compute_manifold(self, features: np.ndarray) -> np.ndarray:
        """
        k번째 가장 가까운 이웃으로 다양체 반지름을 셈한다.
        
        점마다 다양체 반지름은 그 k번째 가장 가까운 이웃까지의 거리이다
        nearest neighbor (excluding itself).
        
        인수:
            features: Feature vectors [N, D]
            
        반환값:
            Manifold radii [N]
        """
        distances = self._batch_pairwise_distances(features, features)
        
        # 대각선을 무한으로 둔다(자신을 뺀다)
        np.fill_diagonal(distances, np.inf)
        
        # 점마다 k번째로 작은 거리를 얻는다
        radii = np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]
        
        return radii
    
    def compute_improved_precision_recall(self,
                                         real_features: np.ndarray,
                                         generated_features: np.ndarray) -> Tuple[float, float]:
        """
        나아진 정밀도와 재현율을 셈한다.
        
        나아진 정밀도와 재현율은 초구면에 바탕한 다양체 어림을 쓴다.
        - 실제 다양체: 실제 표본 둘레 초구면의 합집합
        - 만든 다양체: 만든 표본 둘레 초구면의 합집합
        
        인수:
            real_features: Real image features [N_r, D]
            generated_features: Generated image features [N_g, D]
            
        반환값:
            Tuple of (precision, recall)
        """
        print(f"Computing Improved Precision/Recall (k={self.k})")
        
        # 다양체 반지름을 셈한다
        real_radii = self._compute_manifold(real_features)
        gen_radii = self._compute_manifold(generated_features)
        
        # 실제와 만든 것 사이의 거리를 셈한다
        dist_gen_to_real = self._batch_pairwise_distances(generated_features, real_features)
        dist_real_to_gen = self._batch_pairwise_distances(real_features, generated_features)
        
        # 정밀도: 만든 표본마다 어떤 실제 표본이
        # 그것을 제 다양체 안에 담는지 살핀다(곧 거리 <= 실제 반지름)
        # 꼴: [N_g, N_r] - real_radii [N_r]
        in_real_manifold = dist_gen_to_real <= real_radii[np.newaxis, :]
        precision = np.mean(np.any(in_real_manifold, axis=1))
        
        # 재현율: 실제 표본마다 어떤 만든 표본이
        # 그것을 제 다양체 안에 담는지 살핀다
        in_gen_manifold = dist_real_to_gen <= gen_radii[np.newaxis, :]
        recall = np.mean(np.any(in_gen_manifold, axis=1))
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return float(precision), float(recall)


def compute_density_coverage(real_features: np.ndarray,
                            generated_features: np.ndarray,
                            k: int = 5) -> Tuple[float, float]:
    """
    Compute Density and Coverage metrics (Naeem et al., 2020).
    
    이는 정밀도와 재현율을 달리 적은 것이다.
    - 밀도: 만든 표본의 실제 이웃 평균 수
    - 덮기: 만든 이웃이 적어도 하나 있는 실제 표본의 몫
    
    인수:
        real_features: Real features [N_r, D]
        generated_features: Generated features [N_g, D]
        k: 이웃 수
        
    반환값:
        Tuple of (density, coverage)
    """
    from scipy.spatial.distance import cdist
    
    # 실제 자료의 k번째 이웃 반지름을 셈한다
    real_to_real = cdist(real_features, real_features, 'euclidean')
    np.fill_diagonal(real_to_real, np.inf)
    real_radii = np.partition(real_to_real, k-1, axis=1)[:, k-1]
    
    # 만든 것에서 실제까지의 거리
    gen_to_real = cdist(generated_features, real_features, 'euclidean')
    
    # 밀도: 만든 표본마다 그 이웃 안에 드는 실제 표본의 평균 수
    # 이웃(실제 다양체로 정한다)
    in_ball = gen_to_real <= real_radii[np.newaxis, :]
    density = np.mean(np.sum(in_ball, axis=1)) / k
    
    # 덮기: 만든 이웃이 적어도 하나 있는 실제 표본의 몫
    # 그 다양체 안에
    real_to_gen = cdist(real_features, generated_features, 'euclidean')
    covered = np.any(real_to_gen <= real_radii[:, np.newaxis], axis=1)
    coverage = np.mean(covered)
    
    return float(density), float(coverage)
```

### 보여 주기

```python
def demonstrate_precision_recall():
    """
    다스린 상황으로 정밀도와 재현율을 보인다.
    """
    print("=" * 70)
    print("Precision and Recall Demonstration")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 5000
    feature_dim = 128
    
    # 실제 자료: 정규 분포 무리 4개 섞기
    cluster_centers = np.array([
        [-2, -2],
        [-2, 2],
        [2, -2],
        [2, 2]
    ])
    
    real_features = []
    for center in cluster_centers:
        cluster = np.random.randn(n_samples // 4, 2) * 0.5 + center
        real_features.append(cluster)
    real_features = np.concatenate(real_features, axis=0)
    
    # 더 높은 차원으로 채운다(인셉션 특징을 흉내 낸다)
    real_features = np.concatenate([
        real_features,
        np.zeros((len(real_features), feature_dim - 2))
    ], axis=1)
    
    calculator = ImprovedPrecisionRecall(k=3)
    
    # 상황 1: 가장 좋은 만들어 내기(품질을 지키며 모든 봉우리를 덮는다)
    print("\n📊 Scenario 1: Ideal Generation")
    print("-" * 50)
    
    gen_ideal = []
    for center in cluster_centers:
        cluster = np.random.randn(n_samples // 4, 2) * 0.5 + center
        gen_ideal.append(cluster)
    gen_ideal = np.concatenate(gen_ideal, axis=0)
    gen_ideal = np.concatenate([
        gen_ideal,
        np.zeros((len(gen_ideal), feature_dim - 2))
    ], axis=1)
    
    p1, r1 = calculator.compute_improved_precision_recall(real_features, gen_ideal)
    
    # 상황 2: 봉우리 무너짐(정밀도 높고 재현율 낮음)
    print("\n📊 Scenario 2: Mode Collapse (only 1 cluster)")
    print("-" * 50)
    
    gen_collapse = np.random.randn(n_samples, 2) * 0.5 + cluster_centers[0]
    gen_collapse = np.concatenate([
        gen_collapse,
        np.zeros((len(gen_collapse), feature_dim - 2))
    ], axis=1)
    
    p2, r2 = calculator.compute_improved_precision_recall(real_features, gen_collapse)
    print("Note: High precision (realistic), low recall (missing modes)")
    
    # 상황 3: 낮은 품질(정밀도 낮고 재현율 높음)
    print("\n📊 Scenario 3: Low Quality (noisy but diverse)")
    print("-" * 50)
    
    gen_noisy = []
    for center in cluster_centers:
        cluster = np.random.randn(n_samples // 4, 2) * 2.0 + center  # 높은 잡음
        gen_noisy.append(cluster)
    gen_noisy = np.concatenate(gen_noisy, axis=0)
    gen_noisy = np.concatenate([
        gen_noisy,
        np.zeros((len(gen_noisy), feature_dim - 2))
    ], axis=1)
    
    p3, r3 = calculator.compute_improved_precision_recall(real_features, gen_noisy)
    print("Note: Low precision (unrealistic), high recall (covers modes)")
    
    # 요약
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Precision':>12} {'Recall':>12}")
    print("-" * 50)
    print(f"{'Ideal':<25} {p1:>12.4f} {r1:>12.4f}")
    print(f"{'Mode Collapse':<25} {p2:>12.4f} {r2:>12.4f}")
    print(f"{'Low Quality':<25} {p3:>12.4f} {r3:>12.4f}")
    
    return {
        'ideal': (p1, r1),
        'collapse': (p2, r2),
        'noisy': (p3, r3)
    }


demonstrate_precision_recall()
```

## 정밀도와 재현율 풀이하기

### 정밀도와 재현율의 맞바꿈

만들어 내는 모델은 흔히 맞바꿈을 보인다.

```
Precision ↑  ←→  Recall ↓  (Conservative generation)
Precision ↓  ←→  Recall ↑  (Diverse but noisy)
```

### 짚어내기 표

| 정밀도 | 재현율 | 짚어내기 |
|-----------|--------|-----------|
| 높음 | 높음 | **가장 좋음**: 그럴듯하고 다양하다 |
| 높음 | 낮음 | **봉우리 무너짐**: 품질은 좋으나 봉우리를 빠뜨린다 |
| 낮음 | 높음 | **낮은 충실함**: 자료를 덮지만 품질이 나쁘다 |
| 낮음 | 낮음 | **실패**: 그럴듯하지도 다양하지도 않다 |

### 절충을 눈으로 보기

```python
import matplotlib.pyplot as plt


def plot_precision_recall_tradeoff(models_results: dict):
    """
    모델 여럿의 정밀도와 재현율을 그린다.
    
    인수:
        models_results: Dict mapping model name to (precision, recall)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for name, (precision, recall) in models_results.items():
        ax.scatter(recall, precision, s=100, label=name)
        ax.annotate(name, (recall + 0.02, precision + 0.02))
    
    # 기준선
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 사분면 이름표
    ax.text(0.25, 0.75, 'Mode Collapse\n(good quality, low diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.75, 'Ideal\n(good quality, high diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.25, 0.25, 'Failure\n(poor quality, low diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.25, 'Low Fidelity\n(poor quality, high diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    
    ax.set_xlabel('Recall (Coverage)', fontsize=12)
    ax.set_ylabel('Precision (Fidelity)', fontsize=12)
    ax.set_title('Precision-Recall Tradeoff', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
```

## FID과의 관계

### 정밀도와 재현율이 FID을 어떻게 메우는가

```python
def compare_fid_and_pr():
    """
    정밀도와 재현율이 FID이 주지 못하는 통찰을 어떻게 주는지 보인다.
    """
    # FID은 같지만 정밀도와 재현율이 다른 모델 둘
    
    # 모델 A: 봉우리 무너짐(정밀도 높고 재현율 낮다)
    # 모델 B: 낮은 품질(정밀도 낮고 재현율 높다)
    
    print("Two models with similar FID:")
    print("-" * 40)
    print("Model A (Mode Collapse):")
    print("  FID = 25, Precision = 0.95, Recall = 0.30")
    print("\nModel B (Low Quality):")
    print("  FID = 25, Precision = 0.30, Recall = 0.95")
    print("\n→ FID alone cannot distinguish these cases!")
    print("→ P&R reveals the specific failure mode.")
```

### 두 잣대 함께 쓰기

**권하는 따지기 방식:**

1. **FID**: 빠른 견줌을 위한 전체 품질 점수
2. **정밀도**: 품질이 떨어지는 것을 알아낸다
3. **재현율**: 봉우리 무너짐을 알아낸다

```python
def comprehensive_evaluation(real_features, gen_features):
    """
    잣대 여럿으로 두루 따지기.
    """
    from fid_calculator import FIDCalculator
    
    # FID
    fid_calc = FIDCalculator()
    mu_r, sigma_r = fid_calc.compute_statistics(real_features)
    mu_g, sigma_g = fid_calc.compute_statistics(gen_features)
    fid = fid_calc.calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    
    # 정밀도와 재현율
    pr_calc = ImprovedPrecisionRecall(k=3)
    precision, recall = pr_calc.compute_improved_precision_recall(
        real_features, gen_features
    )
    
    # 알리기
    print("Comprehensive Evaluation:")
    print(f"  FID: {fid:.2f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # 진단
    if fid < 20 and precision > 0.7 and recall > 0.7:
        print("  Diagnosis: Excellent generation")
    elif precision > 0.7 and recall < 0.5:
        print("  Diagnosis: Mode collapse detected")
    elif precision < 0.5 and recall > 0.7:
        print("  Diagnosis: Low quality but good coverage")
    else:
        print("  Diagnosis: Needs improvement")
    
    return fid, precision, recall
```

## 나아가서: F1 점수와 F-베타

### 정밀도와 재현율 아우르기

숫자 하나가 필요하면 F 점수를 쓴다.

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

**흔한 고르기:**

- $F_1$: 정밀도와 재현율에 같은 무게
- $F_{0.5}$: 정밀도를 무겁게 본다(품질 중심)
- $F_2$: 재현율을 무겁게 본다(덮기 중심)

```python
def compute_f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    F-베타 점수를 셈한다.
    
    인수:
        precision: 정밀도 값
        recall: 재현율 값
        beta: Weight parameter (beta > 1 favors recall)
        
    반환값:
        F-베타 점수
    """
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    return f_score


# 사용 예
precision, recall = 0.8, 0.6

f1 = compute_f_score(precision, recall, beta=1.0)
f05 = compute_f_score(precision, recall, beta=0.5)
f2 = compute_f_score(precision, recall, beta=2.0)

print(f"F1 (balanced): {f1:.4f}")
print(f"F0.5 (precision-focused): {f05:.4f}")
print(f"F2 (recall-focused): {f2:.4f}")
```

## 모범 사례

### 1. k을 꼼꼼히 고르라

```python
def analyze_k_sensitivity(real_features, gen_features, k_values=[1, 3, 5, 10, 20]):
    """
    매개변수 k에 대한 민감함을 살핀다.
    """
    results = []
    
    for k in k_values:
        calc = ImprovedPrecisionRecall(k=k)
        p, r = calc.compute_improved_precision_recall(real_features, gen_features)
        results.append({'k': k, 'precision': p, 'recall': r})
        print(f"k={k:2d}: P={p:.4f}, R={r:.4f}")
    
    return results
```

**권하는 바:**

- 기본: k=3(튼튼한 고르기)
- 성긴 자료: k=1(다만 잡음이 많다)
- 빽빽한 자료: k=5-10(더 매끄럽다)

### 2. 표본 크기에 대한 살핌

- 가장 적게: 안정된 어림을 위해 표본 5,000개
- 권함: 표본 10,000개 이상
- 결과를 믿음 구간과 함께 알려라

### 3. 특징 공간 고르기

```python
# 여느 것: InceptionV3(FID과 같다)
from torchvision.models import inception_v3

# 특정 마당의 대안:
# - 글에서 그림으로에는 CLIP 특징
# - 의료나 위성 그림에는 마당에 맞춘 신경망
```

## 요약

!!! success "핵심 간추리기"
    
    1. **정밀도는 충실함을 잰다**: 만든 표본이 그럴듯한가?
    
    2. **재현율은 다양함을 잰다**: 모델이 자료 봉우리를 모두 덮는가?
    
    3. **핵심 짚어내기**:
       - 정밀도 높고 재현율 낮음 → 봉우리 무너짐
       - 정밀도 낮고 재현율 높음 → 낮은 품질
       - 둘 다 높음 → 뛰어난 만들어 내기
    
    4. **FID과 함께 쓰라**: 정밀도와 재현율은 FID이 주지 못하는 통찰을 준다
    
    5. **권하는 자리매김**: k=3, 표본 10,000개 이상, InceptionV3 특징

## 참고 문헌

1. Sajjadi, M.S.M., et al. (2018). "Assessing Generative Models via Precision and Recall." *NeurIPS*.

2. Kynkäänniemi, T., et al. (2019). "Improved Precision and Recall Metric for Assessing Generative Models." *NeurIPS*.

3. Naeem, M.F., et al. (2020). "Reliable Fidelity and Diversity Metrics for Generative Models." *ICML*.

4. Simon, L., et al. (2019). "Revisiting Precision and Recall Definition for Generative Model Evaluation." *ICML*.

## 연습문제

**연습문제 1.**
프레셰 인셉션 거리(FID)를 뜻매김하고 맞겨루기 만들개를 따질 때 인셉션 점수보다 이를 더 낫게 여기는 까닭을 설명하라.

??? success "연습문제 1 풀이"
    FID은 실제 그림과 만든 그림의 Inception-v3 특징 분포를 여러 변수 정규 분포 $\mathcal{N}(\mu_r, \Sigma_r)$과 $\mathcal{N}(\mu_g, \Sigma_g)$으로 나타낸 뒤 다음을 셈한다.

    $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

    FID을 더 낫게 여기는 까닭은 이렇다. (1) 만든 그림을 실제 그림과 견준다(인셉션 점수는 만든 그림만 따진다). (2) 봉우리 무너짐을 알아낸다(평균과 함께 흩어짐이 달라진다). (3) 사람의 판단과 더 잘 맞는다. (4) FID이 낮을수록 느낌의 품질과 더 잘 이어진다.

---

**연습문제 2.**
인셉션 점수의 한계는 무엇인가? 표본이 나쁜데도 인셉션 점수가 높을 수 있는가?

??? success "연습문제 2 풀이"
    인셉션 점수 = $\exp(\mathbb{E}_x [D_{\text{KL}}(p(y|x) \| p(y))])$이며 $p(y|x)$은 만든 그림 $x$에 대한 인셉션 가름개의 헤아림이다. 한계는 이렇다. (1) 품질(또렷하고 가를 수 있음)과 다양함(갈래에 고루 퍼짐)만 재고 익히기 자료에 얼마나 충실한지는 재지 않는다. (2) 갈래마다 완벽한 그림 하나씩만 만드는 모델도 점수가 높지만 갈래 안의 다양함은 무시한다. (3) 인셉션 모델의 치우침에 민감하다. (4) 갈래 안의 결이나 결결이 품질을 담지 못한다. 그렇다. ImageNet 갈래마다 대표 그림 하나씩을 외우면 인셉션 점수가 높아질 수 있다.

---

**연습문제 3.**
만들어 내는 모델의 정밀도와 재현율 잣대가 가르기에서의 그것과 어떻게 다른지 설명하라.

??? success "연습문제 3 풀이"
    만들어 내는 자리에서는(Kynkaanniemi 외, 2019) **정밀도**가 만든 표본 가운데 실제 자료 분포의 받침 안에 드는 몫을 잰다(품질/충실함). **재현율**은 실제 자료 가운데 만든 분포의 받침 안에 드는 몫을 잰다(다양함/덮기). 정밀도가 높고 재현율이 낮으면 봉우리 무너짐이다(봉우리는 적지만 그럴듯하다). 정밀도가 낮고 재현율이 높으면 품질은 나쁘지만 다양하다. 띄엄띄엄한 맞음을 세는 가르기의 정밀도/재현율과 달리 만들어 내기의 정밀도/재현율은 특징 공간에서 $k$번째 가장 가까운 이웃 거리로 분포의 받침을 어림한다.

---

**연습문제 4.**
만들어 내는 모델을 따질 때 여러 잣대를 함께 써야 하는 까닭은 무엇인가?

??? success "연습문제 4 풀이"
    어느 잣대 하나도 만들어 내기 품질의 모든 면을 담지 못한다. **FID**은 전체 분포의 닮음을 재지만 품질과 다양함을 뒤섞는다. **인셉션 점수**는 품질과 다양함을 담지만 익히기 자료에 대한 충실함은 무시한다. **정밀도/재현율**은 품질과 다양함을 갈라내지만 특징 뽑개와 $k$을 어떻게 고르느냐에 매인다. **느낌 잣대**(LPIPS)는 그림 수준 품질을 재지만 다양함은 재지 않는다. 잣대를 함께 쓰면 온전한 그림이 보인다. 곧 FID이 낮고 정밀도가 높으며 재현율이 낮은 모델은 봉우리가 무너진 것이고, 재현율이 높고 정밀도가 낮은 모델은 다양하지만 품질 낮은 표본을 낸다. 마지막 판단에는 사람이 따지는 것이 여전히 으뜸 기준이다.
