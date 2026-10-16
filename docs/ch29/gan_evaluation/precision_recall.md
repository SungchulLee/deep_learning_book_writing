# 생성 모델의 정밀도와 재현율

생성 정밀도·재현율은 생성 샘플이 실제 데이터와 얼마나 가까운지, 실제 데이터가 생성 샘플에 얼마나 포괄되는지를 나누어 살펴보는 지표다. 여기서는 [Kynkäänniemi 등의 방법](https://arxiv.org/abs/1904.06991)처럼 피처 공간의 최근접 이웃 반지름으로 영역을 근사한다. 정답 레이블로 참양성·거짓양성을 세는 분류 평가와는 계산 대상이 다르다.

## 1. 하나의 거리에서 두 측면으로

FID가 비슷한 모델도 표본의 결함과 분포 포괄성은 다를 수 있다. 다음 수치는 그런 상황을 설명하기 위한 가상 예시이며 실제 실험 결과가 아니다.

```
Model A: FID = 50 (high-quality but mode collapsed)
Model B: FID = 50 (diverse but low-quality)

FID는 같지만 문제는 아주 다르다!
```

정밀도는 실제 피처로 만든 영역 안에 들어오는 생성 피처의 비율, 재현율은 생성 피처로 만든 영역 안에 들어오는 실제 피처의 비율로 정의한다. 이 비율을 시각적 품질과 다양성의 단서로 사용하지만 두 개념을 정확히 측정한다고 보장하지는 않는다.

## 2. k번째 최근접 이웃 구들의 합집합

실제 피처를 $r_1,\ldots,r_{N_r}$, 생성 피처를 $g_1,\ldots,g_{N_g}$라 하자. 각 집합에서 자기 자신을 제외한 k번째 최근접 이웃까지의 거리를 각각 $\rho_i^r,\rho_j^g$로 둔다. 추정 영역은 다음과 같다.

$$\widehat M_r=\bigcup_{i=1}^{N_r}\{v:\|v-r_i\|\le\rho_i^r\},\qquad
\widehat M_g=\bigcup_{j=1}^{N_g}\{v:\|v-g_j\|\le\rho_j^g\}.$$

$$\widehat P=\frac1{N_g}\sum_j\mathbf1[g_j\in\widehat M_r],\qquad
\widehat R=\frac1{N_r}\sum_i\mathbf1[r_i\in\widehat M_g].$$

한 점이 **어느 하나의 구**에 포함되면 포함으로 센다. 최근접 중심의 구에만 들어오는지 검사하는 것과 다르다. 가까운 중심의 반지름이 작아 실패해도 더 먼 중심의 큰 구에는 들어갈 수 있다. 두 집합 모두 N>k≥1이어야 한다. 유한 표본의 구 합집합은 참 지지집합이나 실제 매니폴드 자체가 아니다.

## 3. 두 구현의 차이와 계산비용

```python
import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist

class PrecisionRecallCalculator:
    """
    만들어 내는 모델의 정밀도와 재현율 잣대.
    
    Sajjadi 외(2018)와 Kynkäänniemi 외(2019)를 바탕으로 한다.
    
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
            row_batch_size: 거리 셈에 쓰는 배치 크기(줄)
            col_batch_size: 거리 셈에 쓰는 배치 크기(열)
        """
        self.k = k
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
    
    def _batch_pairwise_distances(self,
                                  X: np.ndarray,
                                  Y: np.ndarray) -> np.ndarray:
        """
        배치 단위로 짝마다 유클리드 거리를 셈한다.
        
        인수:
            X: 첫째 점 모임 [N, D]
            Y: 둘째 점 모임 [M, D]
            
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
            X의 점마다 k번째 가장 가까운 이웃까지의 거리 [N]
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
            real_features: 참 그림에서 뽑은 피처 [N_r, D]
            generated_features: 만들어 낸 그림에서 뽑은 피처 [N_g, D]
            
        반환값:
            (정밀도, 재현율) 튜플
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
    Kynkäänniemi 외(2019)의 나아진 정밀도와 재현율(IPR).
    
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
            row_batch_size: 거리 셈하기의 배치 크기
            col_batch_size: 거리 셈하기의 배치 크기
        """
        self.k = k
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
    
    def _batch_pairwise_distances(self,
                                  X: np.ndarray,
                                  Y: np.ndarray) -> np.ndarray:
        """배치 단위로 짝마다 거리를 셈한다."""
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
        가장 가까운 이웃(제 자신은 뺀다).
        
        인수:
            features: 피처 벡터 [N, D]
            
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
            real_features: 참 그림의 피처 [N_r, D]
            generated_features: 만들어 낸 그림의 피처 [N_g, D]
            
        반환값:
            (정밀도, 재현율) 튜플
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
    밀도와 덮음 자를 셈한다(Naeem 외, 2020).
    
    이는 정밀도와 재현율을 달리 적은 것이다.
    - 밀도: 만든 표본의 실제 이웃 평균 수
    - 덮기: 만든 이웃이 적어도 하나 있는 실제 표본의 몫
    
    인수:
        real_features: 참 피처 [N_r, D]
        generated_features: 만들어 낸 피처 [N_g, D]
        k: 이웃 수
        
    반환값:
        (밀도, 덮음) 튜플
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

`PrecisionRecallCalculator`는 최근접 중심 하나의 반지름과만 비교하므로 위 합집합 정의와 다른 근사다. 반면 `ImprovedPrecisionRecall`은 모든 구와의 포함 관계에서 any를 사용해 위 정의를 계산한다. 같은 반지름과 거리라면 최근접 중심 근사는 합집합 방식이 포함하는 점 중 일부를 놓칠 수 있다. 첫 클래스의 사용되지 않는 `gen_to_real_distances`도 추가 계산을 만든다.

`_compute_manifold`는 자기 거리를 ∞로 바꾸고 0부터 세는 k−1번째 값을 선택한다. 첫 클래스는 자기 거리 0이 포함된 행에서 인덱스 k를 선택한다. 중복 피처는 다른 표본이라면 거리 0인 이웃이 될 수 있어, k개 이상의 중복이 있는 위치에서는 반지름이 0이 된다.

거리 계산을 행·열 블록으로 나누지만 결과 **전체 N×M 행렬을 미리 할당한다**. 따라서 메모리가 블록 크기만큼으로 제한되는 구현은 아니다. 각 집합 내부와 집합 사이의 거리 행렬을 여러 번 만들며, 교차 거리의 전치도 다시 계산한다. d차원 유클리드 거리의 대표 연산량은 O((N_r²+N_g²+N_rN_g)d)이며, float32 거리 저장도 N=M=10,000이면 행렬 하나당 약 400 MB다. cdist의 임시 배열과 포함 관계 배열도 추가된다.

## 4. 인공 피처의 평가 예시

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
    
    # 더 높은 차원으로 채운다(인셉션 피처를 흉내 낸다)
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

```
======================================================================
Precision and Recall Demonstration
======================================================================

📊 Scenario 1: Ideal Generation
--------------------------------------------------
Computing Improved Precision/Recall (k=3)
  Precision: 0.9790
  Recall: 0.9778

📊 Scenario 2: Mode Collapse (only 1 cluster)
--------------------------------------------------
Computing Improved Precision/Recall (k=3)
  Precision: 0.9772
  Recall: 0.2466
Note: High precision (realistic), low recall (missing modes)

📊 Scenario 3: Low Quality (noisy but diverse)
--------------------------------------------------
Computing Improved Precision/Recall (k=3)
  Precision: 0.5018
  Recall: 0.9870
Note: Low precision (unrealistic), high recall (covers modes)

======================================================================
Summary
======================================================================
Scenario                     Precision       Recall
--------------------------------------------------
Ideal                           0.9790       0.9778
Mode Collapse                   0.9772       0.2466
Low Quality                     0.5018       0.9870
```

실제 이미지 대신 네 개의 2차원 가우시안 군집을 만들고 나머지 126좌표에는 0을 붙인다. 따라서 유효한 기하 구조는 2차원이며 일반적인 128차원 이미지 피처의 난이도를 재현하지 않는다. 같은 방식의 네 군집, 한 군집만 생성, 넓게 퍼진 네 군집을 비교한다. 보존된 출력은 이 난수 예시의 결과이며 새로 실행한 결과가 아니다.

가우시안 군집은 수학적으로 전체 평면에서 양의 밀도를 가지므로, 이 실험의 '모드 포괄'은 주요 고밀도 영역의 유한 표본 포괄을 뜻한다. 참 지지집합을 정확히 비교하는 실험으로 해석하지 않는다. 같은 모집단의 독립 표본에서도 추정 정밀도·재현율이 반드시 1이 되는 것은 아니다.

## 5. 결과의 해석

```
Precision ↑  ←→  Recall ↓  (Conservative generation)
Precision ↓  ←→  Recall ↑  (Diverse but noisy)
```

높은 정밀도와 낮은 재현율은 실제 영역 근처의 일부 패턴만 생성하는 상황을 시사할 수 있다. 반대 조합은 넓게 생성하지만 실제 영역 밖 피처가 많은 상황을 시사할 수 있다. 다만 두 지표 사이에 필연적인 트레이드오프가 있는 것은 아니며 모델 개선으로 둘 다 높아질 수도 있다.

```python
import matplotlib.pyplot as plt

def plot_precision_recall_tradeoff(models_results: dict):
    """
    모델 여럿의 정밀도와 재현율을 그린다.
    
    인수:
        models_results: 모델 이름을 (정밀도, 재현율)로 보내는 사전
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

위 그림 함수의 0.5 경계와 영역 이름은 설명용이다. 모든 데이터·피처에 공통인 품질 판정 기준이 아니다. 높은 두 값도 피처가 놓치는 결함이나 학습 데이터 암기를 배제하지 않는다.

```python
def compare_fid_and_pr():
    """
    정밀도와 재현율이 FID가 주지 못하는 통찰을 어떻게 주는지 보인다.
    """
    # FID는 같지만 정밀도와 재현율이 다른 모델 둘
    
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

이 함수도 고정 숫자를 출력하는 가상 비교이며 실제 모델을 평가하지 않는다.

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

종합 평가 예시는 외부 `fid_calculator` 모듈을 전제로 한다. FID 계산 클래스와 import 경로가 준비되어야 하며, if문의 고정 임계값은 검증된 자동 진단 기준이 아니다. 수치를 근거로 문제를 조사하되 모드 붕괴나 시각적 실패를 확정하지 않는다.

## 6. density, coverage와 F 점수

첫 코드 블록의 `compute_density_coverage`는 실제 피처 구를 기준으로 두 값을 계산한다. density는 생성 피처가 포함되는 실제 구의 수를 평균 내어 k로 나눈 값이며 1보다 클 수 있다. coverage는 각 실제 중심의 반지름 안에 생성 피처가 하나라도 있는 실제 표본의 비율이다. 생성 구로 실제 표본을 덮는 위 재현율과 다르다. [density와 coverage 논문](https://proceedings.mlr.press/v119/naeem20a.html)은 이웃 기반 정밀도·재현율의 이상값 민감성 등을 분석한다.

정밀도와 재현율을 하나로 요약하려면 β>0에서 다음 F 점수를 사용할 수 있다.

$$F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}.$$

```python
def compute_f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    F-베타 점수를 셈한다.
    
    인수:
        precision: 정밀도 값
        recall: 재현율 값
        beta: 무게 파라메터(beta > 1이면 재현율을 더 친다)
        
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

```
F1 (balanced): 0.6857
F0.5 (precision-focused): 0.7500
F2 (recall-focused): 0.6316
```

F₁은 두 값을 대칭적으로 결합하고 β<1은 정밀도, β>1은 재현율을 상대적으로 강조한다. 하나로 합치면 진단 정보가 줄어드므로 원래 두 값도 보고한다. 함수는 β의 유효성을 검사하지 않아 β=0, R=0, P>0 같은 입력에서는 분모가 0이 될 수 있다.

## 7. 설정과 보고

```python
def analyze_k_sensitivity(real_features, gen_features, k_values=[1, 3, 5, 10, 20]):
    """
    파라메터 k에 대한 민감함을 살핀다.
    """
    results = []
    
    for k in k_values:
        calc = ImprovedPrecisionRecall(k=k)
        p, r = calc.compute_improved_precision_recall(real_features, gen_features)
        results.append({'k': k, 'precision': p, 'recall': r})
        print(f"k={k:2d}: P={p:.4f}, R={r:.4f}")
    
    return results
```

고정된 피처 집합에서 k를 늘리면 각 반지름이 커지거나 같아져 합집합 방식의 정밀도·재현율은 감소하지 않는다. 하지만 최대 허용 k에서도 영역은 유한하므로 모든 점이 반드시 포함되는 것은 아니다. 외딴 실제 표본의 큰 구가 정밀도를 부풀릴 수 있고, 생성 집합의 이상값도 재현율에 영향을 준다. 데이터 밀도만 보고 최적 k를 단정하지 말고 미리 정한 설정과 민감도 분석을 보고한다.

```python
# 여느 것: InceptionV3(FID와 같다)
from torchvision.models import inception_v3

# 특정 마당의 대안:
# - 글에서 그림으로에는 CLIP 피처
# - 의료나 위성 그림에는 마당에 맞춘 뉴럴 넷트웍
```

피처 모델을 바꾸면 거리의 의미와 값도 바뀐다. 모델·체크포인트·전처리, 각 집합의 표본 수, k, 거리 정의와 정확·근사 이웃 계산법을 명시한다. 특정 표본 수만 넘으면 안정적이라는 보장은 없으며 근사 이웃 탐색도 포함 판정을 바꿀 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
FID와 생성 정밀도·재현율은 무엇을 다르게 요약하는가?

</div>

??? success "연습문제 1 풀이"

    FID는 피처 평균·공분산의 차이를 한 수치로 요약한다. 이 페이지의 정밀도·재현율은 각각 실제 구 영역 안의 생성 표본 비율과 생성 구 영역 안의 실제 표본 비율이다. 어느 것도 원래 이미지 분포의 모든 차이를 측정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
클래스별 원핫 예측이 나오는 이미지 하나씩을 균등 반복하면 IS가 높을 수 있는 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"

    주변 클래스 엔트로피는 높고 조건부 엔트로피는 낮기 때문이다. 클래스 내부에 고유 이미지가 하나뿐인 문제는 이 두 예측 통계에서 드러나지 않을 수 있다. 정밀도·재현율도 표본과 피처에 따라 반복을 놓칠 수 있어 중복 분석이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최근접 중심 하나의 구만 확인하는 방식과 구 합집합 방식의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"

    최근접 중심의 반지름이 작으면 포함되지 않더라도 조금 더 먼 중심의 큰 구에는 포함될 수 있다. 합집합 방식은 모든 중심 중 하나라도 포함하는지 검사한다. 따라서 첫 클래스의 근사는 ImprovedPrecisionRecall과 동일한 알고리즘이 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
LPIPS를 단일 생성 이미지의 절대 품질로 해석하면 안 되는 이유는 무엇인가?

</div>

??? success "연습문제 4 풀이"

    LPIPS는 선택한 두 이미지 사이의 지각적 거리를 계산한다. 대응할 기준 이미지나 비교 목적 없이 낮을수록 생성 품질이 좋다고 말할 수 없다. 분포 지표와 이미지 쌍 지표의 평가 단위를 구분한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
정밀도와 재현율의 분모 및 영역을 만드는 집합을 각각 설명하라.

</div>

??? success "연습문제 5 풀이"

    정밀도는 생성 표본 수를 분모로 하고 실제 표본의 구 합집합에 포함되는지 센다. 재현율은 실제 표본 수를 분모로 하고 생성 표본의 구 합집합을 사용한다. 두 영역은 유한 피처 표본의 근사이며 참 매니폴드와 같다고 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
같은 FID에서 정밀도·재현율이 다른 모델이 있을 수 있는가?

</div>

??? success "연습문제 6 풀이"

    가능하다. FID는 모멘트 차이를 하나로 합치므로 다른 피처 구조가 같은 값을 줄 수 있다. 이때 정밀도·재현율이 다른 정보를 제공할 수 있지만 GAN과 VAE라는 이름만으로 특정 패턴을 예측할 수는 없다. 관측 모형, 구조, 학습·샘플링 조건을 실제로 평가해야 한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
같은 피처 집합에서 k를 늘리면 합집합 기반 정밀도·재현율은 어떻게 되는가?

</div>

??? success "연습문제 7 풀이"

    각 중심의 k번째 이웃 거리는 감소하지 않으므로 구 합집합이 줄어들지 않는다. 따라서 같은 평가 점에 대한 포함 비율도 감소하지 않는다. 이는 표본 수나 피처를 바꾸는 경우의 단조성을 뜻하지 않는다. 이상값의 큰 반지름과 중복으로 인한 0 반지름도 확인한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
하나의 학습 피처를 반복 생성하면 두 지표는 어떻게 될 수 있는가?

</div>

??? success "연습문제 8 풀이"

    기준 집합에 그 피처가 있고 k 조건이 유효하면 반복 생성 피처는 실제 영역 안이므로 정밀도는 1이다. 생성 피처끼리의 반지름은 0이므로 재현율은 그 위치와 정확히 같은 실제 피처의 비율이다. 거의 0일 수 있지만 항상 0은 아니다. 전체 학습 피처를 재생하면 같은 기준 집합에 두 값이 1일 수 있어 암기를 별도로 조사해야 한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
F 점수 하나만 보고할 때 잃는 정보는 무엇인가?

</div>

??? success "연습문제 9 풀이"

    정밀도와 재현율의 다른 조합이 같은 F 점수를 만들 수 있어 실패 원인을 구분하기 어렵다. β와 결합값을 보고하더라도 원래 P와 R을 함께 제시한다. 순위를 FID로만 정해야 한다는 보편적 규칙도 없으며 평가 목적에 따라 기준을 정한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
블록별 cdist 호출만으로 메모리가 O(블록 크기)가 되는가?

</div>

??? success "연습문제 10 풀이"

    아니다. 이 구현은 전체 거리 행렬을 먼저 할당하므로 N×M 메모리를 사용한다. 블록화는 각 호출의 임시 계산을 나눌 뿐이다. 실제 메모리를 줄이려면 k번째 거리와 포함 판정에 필요한 값만 누적하는 별도 알고리즘이 필요하다. 근사 이웃을 사용하면 방법과 오차 영향을 보고한다.



## 정리하며

생성 정밀도·재현율은 피처 공간에서 품질과 포괄성을 나누어 살펴보는 유한 표본 지표다. 모든 구의 합집합과 최근접 중심 근사를 구분하고, k·표본 수·피처·메모리 비용을 명시한다. 높은 점수도 시각적 품질이나 비암기를 완전히 보장하지 않는다.

**참고 문헌**

1. Sajjadi, M.S.M., et al. (2018). "Assessing Generative Models via Precision and Recall." *NeurIPS*.

2. Kynkäänniemi, T., et al. (2019). "Improved Precision and Recall Metric for Assessing Generative Models." *NeurIPS*.

3. Naeem, M.F., et al. (2020). "Reliable Fidelity and Diversity Metrics for Generative Models." *ICML*.

4. Simon, L., et al. (2019). "Revisiting Precision and Recall Definition for Generative Model Evaluation." *ICML*.
