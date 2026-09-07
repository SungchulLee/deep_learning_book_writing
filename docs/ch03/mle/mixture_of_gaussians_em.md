# 가우스 혼합 EM

가우스 혼합 MLE - 기댓값 최대화 알고리즘. 문제: K개의 서로 다른 정규분포에서 나온 데이터를 관측하지만,

이 튜토리얼은 PyTorch에서 최대가능도 추정에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
#!/usr/bin/env python3
"""
================================================================================
가우스 섞음 최대가능도 — 기댓값-최대화 알고리즘
================================================================================

어려움: ⭐⭐⭐ 앞선(3단계)

학습 목표:
- 숨은 변수 모델을 이해한다
- 기댓값-최대화(EM) 알고리즘을 배운다
- 부드러운 무리 짓기를 짠다
- 최대가능도가 빠진 데이터와 숨은 데이터를 어떻게 다루는지 본다

문제: 서로 다른 가우스 분포 K개에서 나온 데이터를 보지만 어떤 점이 어느
분포에서 왔는지는 모른다. 매개변수를 모두 어림하여라!

MODEL:
- 평균이 μₖ, 공분산이 Σₖ인 가우스 조각 K개
- 섞음 가중치 πₖ(조각 k일 확률)
- 숨은 변수 zᵢ이 xᵢ을 만든 조각을 가리킨다

가능도(숨은 변수를 넣어):
P(x, z | θ) = ∏ [πₖ N(xᵢ | μₖ, Σₖ)]^{z_ik}

그런데 z은 볼 수 없다! 그래서 EM 알고리즘을 쓴다.

E 걸음: 뒤확률(맡은 몫)을 계산한다
γᵢₖ = P(z_ik = 1 | xᵢ, θ)

M 걸음: 온 데이터 로그 가능도의 기댓값을 가장 크게 하도록 매개변수를 고친다
πₖ = (1/N) Σ γᵢₖ
μₖ = (Σ γᵢₖ xᵢ) / (Σ γᵢₖ)
Σₖ = (Σ γᵢₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ) / (Σ γᵢₖ)

APPLICATIONS:
- 무리 짓기(부드러운 k 평균)
- 이상 알아내기
- 그림 나누기
- 말소리 알아듣기
- 주제 모델 짓기

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
    
    # 참된 매개변수
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
        """k-means++로 매개변수를 초기화한다"""
        n_samples, n_features = X.shape
        
        # k-means++ 방식으로 평균을 초기화한다
        indices = torch.randperm(n_samples)[:self.n_components]
        self.means = X[indices].clone()
        
        # 공분산을 단위행렬로 초기화한다
        self.covs = [torch.eye(n_features) for _ in range(self.n_components)]
        
        # 가중치를 균등하게 초기화한다
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
        E 걸음: 맡은 몫(뒤확률)을 계산한다.
        
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
        M 걸음: 로그 가능도의 기댓값을 가장 크게 하도록 매개변수를 고친다.
        """
        n_samples, n_features = X.shape
        
        # 각 성분에 배정된 실효 점 개수
        N_k = responsibilities.sum(dim=0)
        
        # 가중치를 갱신한다
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
            
            # M 단계: 매개변수를 갱신한다
            self.m_step(X, responsibilities)
            
            # 로그가능도를 계산한다
            log_lik = self.compute_log_likelihood(X)
            self.history.append(log_lik)
            
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration+1}/{self.n_iterations}, Log-Likelihood: {log_lik:.2f}")
        
        return self
    
    def predict(self, X):
        """군집 배정을 예측한다 (딱딱한 군집화)"""
        responsibilities = self.e_step(X)
        return torch.argmax(responsibilities, dim=1)
    
    def predict_proba(self, X):
        """군집 확률을 예측한다 (부드러운 군집화)"""
        return self.e_step(X)


def plot_gmm_results(X, labels, gmm, true_means):
    """종합적인 시각화를 만든다"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # ================================================================
    # 그림 1: 참 군집 (레이블가 있는 경우)
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
    # 그림 3: 부드러운 군집 (책임도)
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
    # 그림 5: 성분 가중치
    # ================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    components = [f'Comp {k}' for k in range(gmm.n_components)]
    bars = ax5.bar(components, gmm.weights.numpy(), color=colors[:gmm.n_components], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    
    # 값 레이블를 추가한다
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax5.set_ylabel('Weight (π_k)', fontsize=12)
    ax5.set_title('Mixture Weights', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1.0])
    
    # ================================================================
    # 그림 6: 혼동 행렬
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

1. 보통: 모델을 절로 고르기
   - 조각 수(K)를 바꾸어 본다
   - 모델 고르기에 BIC이나 AIC를 쓴다
   - BIC와 K을 견주어 그린다

2. 보통: 여러 공분산 짜임
   - Spherical: Σₖ = σₖ²I
   - 대각: Σₖ = diag(σₖ₁², ..., σₖₚ²)
   - 온전한 것: Σₖ(이제 짜보기)
   - 성능과 빠르기를 견준다

3. 어려움: 첫자리 잡는 꾀
   - 마구잡이 첫자리
   - K 평균++ 첫자리
   - 여러 번 마구잡이로 다시 비롯하기
   - 모여드는 모습을 견준다

4. 어려움: 베이즈 가우스 섞음 모델
   - 섞음 가중치에 디리클레 앞확률을 둔다
   - 평균과 공분산에 가우스-위샤트 앞확률을 둔다
   - 변이 추론을 짠다

5. 어려움: 쓰임새
   - 가우스 섞음 모델으로 하는 그림 나누기
   - 이상 알아내기(확률이 낮은 점)
   - 밀도 어림과 뽑기
"""


if __name__ == "__main__":
    main()```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 통계적 추론 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
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
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
