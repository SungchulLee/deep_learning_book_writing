# 가우시안 혼합 EM

각 관측이 어느 가우스 성분에서 나왔는지 모르는 상황에서 혼합비, 평균과 공분산을 추정한다. 이 예제는 경사 하강법 대신 E 단계와 M 단계를 반복한다.

## 1. 코드

??? note "코드 (483줄)"

    ```python
    #!/usr/bin/env python3
    """
    ================================================================================
    가우스 혼합 최대가능도 — 기댓값-최대화 알고리즘
    ================================================================================

    난이도: ⭐⭐⭐ 심화(3단계)

    학습 목표:
    - 잠재변수 모델을 이해한다
    - 기댓값-최대화(EM) 알고리즘을 배운다
    - 소프트 군집화를 구현한다
    - 최대가능도가 결측 데이터와 잠재 데이터를 어떻게 다루는지 본다

    문제: 서로 다른 가우스 분포 K개에서 나온 데이터를 보지만 어떤 점이 어느
    분포에서 왔는지는 모른다. 파라메터를 모두 추정하여라!

    MODEL:
    - 평균이 μₖ, 공분산이 Σₖ인 가우스 성분 K개
    - 혼합 웨이트 πₖ(성분 k일 확률)
    - 잠재변수 zᵢ가 xᵢ를 생성한 성분을 가리킨다

    가능도(잠재변수 포함):
    P(x, z | θ) = ∏ [πₖ N(xᵢ | μₖ, Σₖ)]^{z_ik}

    그런데 z은 볼 수 없다! 그래서 EM 알고리즘을 쓴다.

    E 단계: 사후확률(책임도)을 계산한다
    γᵢₖ = P(z_ik = 1 | xᵢ, θ)

    M 단계: 완전 데이터 로그 가능도의 기댓값을 최대화하도록 파라메터를 갱신한다
    πₖ = (1/N) Σ γᵢₖ
    μₖ = (Σ γᵢₖ xᵢ) / (Σ γᵢₖ)
    Σₖ = (Σ γᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ) / (Σ γᵢₖ)

    APPLICATIONS:
    - 군집화(소프트 k-평균)
    - 이상 탐지
    - 영상 분할
    - 음성 인식
    - 토픽 모델링

    지은이: PyTorch 최대가능도 학습
    DATE: 2025
    ================================================================================
    """

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from typing import Tuple, List

    # ========================================================================
    # 메인
    # ========================================================================


    def generate_gmm_data(n_samples: int = 300, n_components: int = 3, seed: int = 42):
        """가우스 혼합 모델에서 데이터를 생성한다"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 참된 파라메터
        true_means = [
            torch.tensor([-3.0, -3.0]),
            torch.tensor([3.0, 3.0]),
            torch.tensor([0.0, 5.0])
        ]

        true_covs = [
            torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
            torch.tensor([[1.5, -0.3], [-0.3, 1.5]]),
            torch.tensor([[0.8, 0.0], [0.0, 0.8]])
        ]

        true_weights = torch.tensor([0.3, 0.4, 0.3])

        # 데이터를 생성한다
        X_list = []
        labels_list = []

        for k in range(n_components):
            n_k = int(n_samples * true_weights[k])

            # 다변량 정규분포에서 표본을 뽑는다
            mean = true_means[k].numpy()
            cov = true_covs[k].numpy()
            X_k = np.random.multivariate_normal(mean, cov, n_k)

            X_list.append(torch.tensor(X_k, dtype=torch.float32))
            labels_list.append(torch.full((n_k,), k, dtype=torch.long))

        X = torch.cat(X_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # 뒤섞는다
        perm = torch.randperm(len(X))
        X, labels = X[perm], labels[perm]

        return X, labels, true_means, true_covs, true_weights


    class GaussianMixture:
        """EM 알고리즘을 쓰는 가우스 혼합 모델"""

        def __init__(self, n_components=3, n_iterations=50):
            self.n_components = n_components
            self.n_iterations = n_iterations

            self.means = None
            self.covs = None
            self.weights = None
            self.history = []

        def initialize_parameters(self, X):
            """무작위로 고른 데이터 점으로 파라메터를 초기화한다"""
            n_samples, n_features = X.shape

            # 무작위로 고른 데이터 점을 초기 평균으로 쓴다
            indices = torch.randperm(n_samples)[:self.n_components]
            self.means = X[indices].clone()

            # 공분산을 단위행렬로 초기화한다
            self.covs = [torch.eye(n_features) for _ in range(self.n_components)]

            # 웨이트를 균등하게 초기화한다
            self.weights = torch.ones(self.n_components) / self.n_components

        def gaussian_pdf(self, X, mean, cov):
            """다변량 정규분포의 확률밀도를 계산한다"""
            n_features = X.shape[1]

            # 수치적 안정성을 위해 대각에 작은 값을 더한다
            cov = cov + torch.eye(n_features) * 1e-6

            # 확률을 계산한다
            diff = X - mean
            cov_inv = torch.inverse(cov)

            exponent = -0.5 * torch.sum(diff @ cov_inv * diff, dim=1)
            normalization = 1.0 / torch.sqrt((2 * np.pi) ** n_features * torch.det(cov))

            return normalization * torch.exp(exponent)

        def e_step(self, X):
            """
            E 단계: 책임도(사후확률)를 계산한다.

            γᵢₖ = πₖ N(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ | μⱼ, Σⱼ)
            """
            n_samples = X.shape[0]
            responsibilities = torch.zeros(n_samples, self.n_components)

            # 성분마다 가중 확률을 계산한다
            for k in range(self.n_components):
                prob = self.gaussian_pdf(X, self.means[k], self.covs[k])
                responsibilities[:, k] = self.weights[k] * prob

            # 정규화하여 사후확률을 얻는다
            responsibilities = responsibilities / (responsibilities.sum(dim=1, keepdim=True) + 1e-10)

            return responsibilities

        def m_step(self, X, responsibilities):
            """
            M 단계: 로그 가능도의 기댓값을 최대화하도록 파라메터를 갱신한다.
            """
            n_samples, n_features = X.shape

            # 각 성분에 배정된 실효 점 개수
            N_k = responsibilities.sum(dim=0)

            # 웨이트를 갱신한다
            self.weights = N_k / n_samples

            # 평균을 갱신한다
            for k in range(self.n_components):
                self.means[k] = (responsibilities[:, k:k+1] * X).sum(dim=0) / N_k[k]

            # 공분산을 갱신한다
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covs[k] = (weighted_diff.T @ diff) / N_k[k]

                # 양의 정부호임을 보장한다
                self.covs[k] = self.covs[k] + torch.eye(n_features) * 1e-6

        def compute_log_likelihood(self, X):
            """데이터의 로그가능도를 계산한다"""
            n_samples = X.shape[0]
            log_likelihood = 0.0

            for i in range(n_samples):
                # 이 점에 대한 혼합 확률
                prob_sum = 0.0
                for k in range(self.n_components):
                    prob = self.gaussian_pdf(X[i:i+1], self.means[k], self.covs[k])
                    prob_sum += self.weights[k] * prob

                log_likelihood += torch.log(prob_sum + 1e-10)

            return log_likelihood.item()

        def fit(self, X):
            """EM 알고리즘으로 GMM을 적합시킨다"""
            self.initialize_parameters(X)

            print(f"   Running EM algorithm for {self.n_iterations} iterations...")

            for iteration in range(self.n_iterations):
                # E 단계: 책임도를 계산한다
                responsibilities = self.e_step(X)

                # M 단계: 파라메터를 갱신한다
                self.m_step(X, responsibilities)

                # 로그가능도를 계산한다
                log_lik = self.compute_log_likelihood(X)
                self.history.append(log_lik)

                if (iteration + 1) % 10 == 0:
                    print(f"   Iteration {iteration+1}/{self.n_iterations}, Log-Likelihood: {log_lik:.2f}")

            return self

        def predict(self, X):
            """군집 배정을 예측한다 (하드 군집화)"""
            responsibilities = self.e_step(X)
            return torch.argmax(responsibilities, dim=1)

        def predict_proba(self, X):
            """군집 확률을 예측한다 (소프트 군집화)"""
            return self.e_step(X)


    def plot_gmm_results(X, labels, gmm, true_means):
        """종합적인 시각화를 만든다"""

        fig = plt.figure(figsize=(18, 12))

        # ================================================================
        # 그림 1: 참 군집 (라벨이 있는 경우)
        # ================================================================
        ax1 = plt.subplot(2, 3, 1)

        colors = ['red', 'blue', 'green']
        for k in range(gmm.n_components):
            mask = labels == k
            ax1.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'True Cluster {k}')

        # 참 평균을 그린다
        for k, mean in enumerate(true_means):
            ax1.scatter(mean[0], mean[1], c=colors[k], marker='*', s=500, 
                       edgecolors='black', linewidths=2, label=f'True μ_{k}')

        ax1.set_xlabel('Feature 1', fontsize=12)
        ax1.set_ylabel('Feature 2', fontsize=12)
        ax1.set_title('True Clusters', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ================================================================
        # 그림 2: 예측된 군집 (EM)
        # ================================================================
        ax2 = plt.subplot(2, 3, 2)

        predicted_labels = gmm.predict(X)

        for k in range(gmm.n_components):
            mask = predicted_labels == k
            ax2.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30, label=f'Cluster {k}')

        # 추정된 평균과 공분산을 그린다
        for k in range(gmm.n_components):
            mean = gmm.means[k]
            cov = gmm.covs[k]

            # 평균을 그린다
            ax2.scatter(mean[0], mean[1], c=colors[k], marker='X', s=500,
                       edgecolors='black', linewidths=2, label=f'Est μ_{k}')

            # 공분산 타원을 그린다
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item()))
            width, height = 2 * torch.sqrt(eigenvalues)

            ellipse = Ellipse(mean.numpy(), width.item(), height.item(), angle=angle,
                             facecolor='none', edgecolor=colors[k], linewidth=2, linestyle='--')
            ax2.add_patch(ellipse)

        ax2.set_xlabel('Feature 1', fontsize=12)
        ax2.set_ylabel('Feature 2', fontsize=12)
        ax2.set_title('EM Clustering Results', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ================================================================
        # 그림 3: 소프트 군집 (책임도)
        # ================================================================
        ax3 = plt.subplot(2, 3, 3)

        responsibilities = gmm.predict_proba(X)

        # 책임도에 따라 RGB 색을 만든다
        rgb_colors = torch.zeros(len(X), 3)
        for k in range(min(3, gmm.n_components)):
            if k == 0:
                rgb_colors[:, 0] = responsibilities[:, k]  # Red
            elif k == 1:
                rgb_colors[:, 2] = responsibilities[:, k]  # Blue
            elif k == 2:
                rgb_colors[:, 1] = responsibilities[:, k]  # Green

        ax3.scatter(X[:, 0], X[:, 1], c=rgb_colors.numpy(), s=50, alpha=0.7, edgecolors='black', linewidths=0.5)

        for k in range(gmm.n_components):
            mean = gmm.means[k]
            ax3.scatter(mean[0], mean[1], c=colors[k], marker='X', s=500,
                       edgecolors='black', linewidths=2)

        ax3.set_xlabel('Feature 1', fontsize=12)
        ax3.set_ylabel('Feature 2', fontsize=12)
        ax3.set_title('Soft Clustering (Color = Responsibility)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # ================================================================
        # 그림 4: 로그가능도의 수렴
        # ================================================================
        ax4 = plt.subplot(2, 3, 4)

        ax4.plot(gmm.history, 'b-', linewidth=2)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Log-Likelihood', fontsize=12)
        ax4.set_title('EM Convergence', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # ================================================================
        # 그림 5: 성분 웨이트
        # ================================================================
        ax5 = plt.subplot(2, 3, 5)

        components = [f'Comp {k}' for k in range(gmm.n_components)]
        bars = ax5.bar(components, gmm.weights.numpy(), color=colors[:gmm.n_components], 
                       alpha=0.7, edgecolor='black', linewidth=2)

        # 값 라벨을 추가한다
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax5.set_ylabel('Weight (π_k)', fontsize=12)
        ax5.set_title('Mixture Weights', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim([0, 1.0])

        # ================================================================
        # 그림 6: 요약 통계
        # ================================================================
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        # 지표를 계산한다
        ari = adjusted_rand_score(labels.numpy(), predicted_labels.numpy())
        nmi = normalized_mutual_info_score(labels.numpy(), predicted_labels.numpy())

        # 요약 표
        table_data = [
            ['Metric', 'Value'],
            ['Components (K)', f'{gmm.n_components}'],
            ['Data points (N)', f'{len(X)}'],
            ['Final Log-Likelihood', f'{gmm.history[-1]:.2f}'],
            ['Adjusted Rand Index', f'{ari:.3f}'],
            ['Normalized Mutual Info', f'{nmi:.3f}'],
        ]

        table = ax6.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)

        # 머리글 서식
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('mixture_gaussians_em_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 Figure saved as 'mixture_gaussians_em_results.png'")
        plt.show()


    def main():
        print("=" * 80)
        print("MIXTURE OF GAUSSIANS - EM Algorithm")
        print("=" * 80)

        # 데이터를 생성한다
        print("\n🎲 Generating data from Gaussian Mixture...")
        X, labels, true_means, true_covs, true_weights = generate_gmm_data(n_samples=300, n_components=3)

        print(f"   • Generated {len(X)} data points")
        print(f"   • Number of true components: {len(true_means)}")
        print(f"   • True mixture weights: {true_weights.numpy()}")

        # GMM을 적합시킨다
        print("\n🔄 Fitting Gaussian Mixture Model...")
        print("-" * 80)

        gmm = GaussianMixture(n_components=3, n_iterations=50)
        gmm.fit(X)

        # 결과
        print("\n📊 Results:")
        print("-" * 80)
        print(f"   Final log-likelihood: {gmm.history[-1]:.2f}")
        print(f"   Estimated mixture weights: {gmm.weights.numpy()}")
        print("\n   Estimated means:")
        for k in range(gmm.n_components):
            print(f"      Component {k}: {gmm.means[k].numpy()}")

        # 군집화 평가
        predicted_labels = gmm.predict(X)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(labels.numpy(), predicted_labels.numpy())
        print(f"\n   Clustering quality (ARI): {ari:.3f}")

        # 시각화한다
        print("\n📊 Creating visualizations...")
        plot_gmm_results(X, labels, gmm, true_means)

        print("\n" + "=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print("\n💡 KEY TAKEAWAYS:")
        print("   1. EM algorithm handles latent (hidden) variables")
        print("   2. E-step: compute responsibilities (soft assignments)")
        print("   3. M-step: update parameters using weighted MLE")
        print("   4. Converges to local maximum (not necessarily global)")
        print("   5. Foundation for many ML algorithms (k-means, HMM, etc.)")
        print("\n" + "=" * 80)


    """
    🎓 EXERCISES:

    1. 보통: 자동 모델 선택
       - 성분 수(K)를 바꾸어 본다
       - 모델 선택에 BIC나 AIC를 쓴다
       - K에 따른 BIC를 그린다

    2. 보통: 여러 공분산 구조
       - Spherical: Σₖ = σₖ²I
       - 대각: Σₖ = diag(σₖ₁², ..., σₖₚ²)
       - 완전(full): Σₖ(현재 구현)
       - 성능과 속도를 견준다

    3. 어려움: 초기화 전략
       - 무작위 초기화
       - k-means++ 초기화
       - 여러 번의 무작위 재시작
       - 수렴 양상을 견준다

    4. 어려움: 베이즈 가우스 혼합 모델
       - 혼합 웨이트에 디리클레 사전분포를 둔다
       - 평균과 공분산에 가우스-위샤트 사전분포를 둔다
       - 변분 추론을 구현한다

    5. 어려움: 응용
       - 가우스 혼합 모델로 하는 영상 분할
       - 이상 탐지(확률이 낮은 점)
       - 밀도 추정과 샘플링
    """


    if __name__ == "__main__":
        main()
    ```


**출력:**

```
================================================================================
MIXTURE OF GAUSSIANS - EM Algorithm
================================================================================

🎲 Generating data from Gaussian Mixture...
   • Generated 300 data points
   • Number of true components: 3
   • True mixture weights: [0.3 0.4 0.3]

🔄 Fitting Gaussian Mixture Model...
--------------------------------------------------------------------------------
   Running EM algorithm for 50 iterations...
   Iteration 10/50, Log-Likelihood: -1134.90
   Iteration 20/50, Log-Likelihood: -1134.21
   Iteration 30/50, Log-Likelihood: -1134.17
   Iteration 40/50, Log-Likelihood: -1134.17
   Iteration 50/50, Log-Likelihood: -1134.17

📊 Results:
--------------------------------------------------------------------------------
   Final log-likelihood: -1134.17
   Estimated mixture weights: [0.3090367  0.39096326 0.29999995]

   Estimated means:
      Component 0: [-0.05871385  4.791123  ]
      Component 1: [3.0191047 3.1185782]
      Component 2: [-2.938697 -2.875988]

   Clustering quality (ARI): 0.881

📊 Creating visualizations...

📊 Figure saved as 'mixture_gaussians_em_results.png'

================================================================================
✅ COMPLETE!
================================================================================

💡 KEY TAKEAWAYS:
   1. EM algorithm handles latent (hidden) variables
   2. E-step: compute responsibilities (soft assignments)
   3. M-step: update parameters using weighted MLE
   4. Converges to local maximum (not necessarily global)
   5. Foundation for many ML algorithms (k-means, HMM, etc.)

================================================================================
```

## 2. 논의

### 잠재 성분과 책임도

혼합분포는 $p(x)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)$이다. E 단계는 현재 파라메터로 관측별 성분의 사후확률인 책임도를 계산한다. M 단계는 책임도로 가중한 표본 수, 평균과 공분산을 사용해 파라메터를 다시 추정한다. 두 단계를 반복하는 것이 EM이다.

### 수렴 보장의 범위

정확한 E·M 단계는 관측 로그가능도(log-likelihood; 가능도는 우도라고도 한다)를 감소시키지 않지만 전역 최댓값을 보장하지 않는다. 초기화가 다른 해를 만들 수 있으며, 가우스 혼합의 비제약 가능도는 한 성분의 공분산이 표본에 붕괴하며 무한히 커질 수 있다. 예제의 대각선 보정과 작은 분모 상수는 수치적 안정화이며 이상적인 EM의 정확한 단조성 보장을 그대로 적용하는 근거는 아니다.

**「초기화가 다른 해를 만들 수 있다」가 이 데이터에서 얼마나 자주 일어나는가.** 위 코드는 한 번만 돌린다. 같은 데이터에 초기화만 바꾸어 서른 번 돌려 보면 여덟 번이 다른 곳에 멈춘다.

![칸이 둘인 그림. 두 산점도 모두 같은 300개의 점과 데이터를 만든 세 중심을 회색 가위표로 보인다. 왼쪽은 쪽이 찍은 결과로 세 타원이 세 군집을 하나씩 덮고 웨이트가 0.309, 0.391, 0.300이며 로그가능도는 마이너스 1134.17, ARI는 0.881이다. 오른쪽은 초기화만 바꾼 결과로 커다란 타원 하나가 위쪽 두 군집을 통째로 삼켜 웨이트 0.700을 가지고, 나머지 두 성분이 아래쪽 한 군집을 0.164와 0.136으로 나누어 가진다. 로그가능도는 마이너스 1152.17, ARI는 0.443이다](../figures/em_local_maxima.svg)

| | 로그가능도 | ARI |
|---|---|---|
| 서른 번 가운데 **22번** | $-1134.17$ | 0.881 |
| 나머지 **8번** | $-1155.44 \sim -1151.86$ | 0.440 ~ 0.455 |

**실패한 쪽의 징후가 분명하다.** 어느 성분 하나가 웨이트 **0.700**을 가진다. 참 혼합 웨이트가 $(0.3, 0.4, 0.3)$이므로 0.700은 **두 군집을 합친 값**이다. 초기 평균 세 개 가운데 둘이 같은 군집에 떨어지면, 그 둘이 한 군집을 나누어 갖고 남은 하나가 나머지 두 군집을 통째로 맡는다. 이 배치에서 E 단계와 M 단계를 아무리 반복해도 빠져나오지 못한다. 로그가능도는 매 단계 올라가되 **올라갈 곳이 거기까지**다.

ARI가 0.881에서 0.44로 반 토막 나는데도 **로그가능도는 2%가 채 안 되게 떨어진다.** 로그가능도만 보고 있으면 이 실패를 알아채기 어렵다는 뜻이기도 하다.

실무에서는 그래서 **여러 초기화로 돌려 로그가능도가 가장 높은 것을 고른다**(scikit-learn의 `GaussianMixture`에는 `n_init` 인자가 있다). 위 코드처럼 한 번만 돌린 결과는, 그 한 번이 좋은 자리에 떨어졌는지 알 수 없다.

### 성분 번호와 평가

성분 번호를 순열로 바꾸어도 같은 혼합분포다. 따라서 생성 라벨과 예측 번호의 단순 일치율보다 ARI처럼 순열에 영향을 받지 않는 지표가 적절하다. 책임도 합, 아주 작은 성분의 유효 표본 수와 공분산의 상태를 함께 살펴본다. 이 구현에는 옵티마이저나 역전파가 필요하지 않으므로 일반 뉴럴 네트워크 학습 루프와 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 EM 구현에 `optimizer.zero_grad()`나 `loss.backward()`가 필요하지 않은 이유를 설명하라.

</div>

??? success "풀이"
    E 단계는 책임도를 계산하고 M 단계는 그 책임도로 가중한 닫힌 형태의 파라메터 업데이트를 수행한다. 그래디언트를 계산해 옵티마이저로 이동하는 학습 루프가 아니므로 그래디언트 초기화도 필요하지 않다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
EM을 여러 초기값에서 실행할 때 어떤 기준으로 결과를 비교해야 하는가?

</div>

??? success "풀이"
    같은 데이터와 성분 수에서 최종 관측 로그가능도와 수치적 안정성을 비교한다. 초기값에 따른 국소해 차이가 있으므로 여러 번 재시작할 수 있다. 정답 성분이 있는 합성 실험에서는 라벨 순열에 불변인 ARI 같은 지표도 함께 본다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
가우스 혼합에서 성분 붕괴와 수치 언더플로를 어떻게 진단하는가?

</div>

??? success "풀이"
    아주 작은 성분 유효 표본 수, 공분산의 작은 고윳값과 비정상적인 로그가능도 증가를 확인한다. 성분 밀도를 직접 더할 때의 언더플로에는 로그 공간 계산이 도움이 된다. 공분산 제약이나 재초기화는 목적과 보장 조건을 함께 고려해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료를 구현하라. 매 에포크 후 검증 로스를 추적하고, 10 에포크 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 웨이트를 저장하고 복원하라.

</div>

??? success "풀이"
    인내 횟수 카운터와 최저 로스 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    검증 지표가 더 나아지지 않을 때 중단하고 가장 좋은 체크포인트를 복원하면 과적합을 줄이는 데 도움이 될 수 있다. 검증셋을 반복적인 선택에 사용한 결과와 독립 시험 성능은 구분한다.

## 정리하며

EM은 잠재 성분에 대한 추정과 파라메터 업데이트를 번갈아 수행한다. 초기화, 수치 안정화와 성분 붕괴를 점검하고 성분 번호 자체에는 의미를 부여하지 않는다.
