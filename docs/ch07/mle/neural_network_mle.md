# 뉴럴 넷트웍 MLE

평균만 예측하는 회귀 뉴럴 넷트웍과 평균·분산을 함께 예측하는 이분산 뉴럴 넷트웍을 비교한다. 두 모델은 결합된 부품이 아니라 서로 다른 조건부 분포 가정을 비교하는 대안이다.

## 1. 코드

??? note "코드 (484줄)"

    ```python
    #!/usr/bin/env python3
    """
    ================================================================================
    뉴럴 넷트웍 최대가능도 — 최대가능도로 하는 깊은 학습
    ================================================================================

    어려움: ⭐⭐⭐ 앞선(3단계)

    학습 목표:
    - 뉴럴 넷트웍이 최대가능도를 어떻게 쓰는지 이해한다
    - 로스 함수와 가능도 사이의 이음을 본다
    - 최대가능도 바탕 맞춤 로스를 짠다
    - 이분산 회귀(불확실성 예측)를 배운다

    핵심 통찰: 뉴럴 넷트웍 학습이 곧 최대가능도 어림이다!

    여느 회귀:
    - 망이 예측한다: ŷ = f(x; θ)
    - 가정: σ이 붙박인 y ~ N(ŷ, σ²)
    - 최대가능도 목표: Σ(y - ŷ)²을 가장 작게 한다(MSE 로스)

    이분산 회귀:
    - 망이 평균과 분산을 모두 예측한다: (μ̂, σ̂²) = f(x; θ)
    - 모델: σ이 달라지는 y ~ N(μ̂, σ̂²)
    - 최대가능도 목표: Σ log N(y | μ̂, σ̂²)을 가장 크게 한다
                 = Σ [log(σ̂²) + (y - μ̂)²/σ̂²]을 가장 작게 한다

    그러면 망이 제 예측의 불확실성을 나타낼 수 있다!

    APPLICATIONS:
    - 불확실성을 수로 나타내는 회귀
    - 든든한 회귀(튄값 다루기)
    - 앞장선 학습(불확실성이 큰 점을 묻는다)
    - 무릅씀을 살피는 판단

    지은이: PyTorch 최대가능도 학습
    DATE: 2025
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Tuple

    # ========================================================================
    # 메인
    # ========================================================================


    def generate_heteroscedastic_data(n_samples: int = 300, seed: int = 42):
        """
        x에 따라 잡음이 달라지는 데이터를 만든다(이분산).

        y = sin(x) + ε, where ε ~ N(0, σ(x)²) and σ(x) increases with |x|
        """
        torch.manual_seed(seed)

        # x 값을 생성한다
        x = torch.rand(n_samples, 1) * 10 - 5  # Range: [-5, 5]

        # 참 함수: 사인파
        y_true = torch.sin(x)

        # 이분산 잡음: σ(x) = 0.1 + 0.1 * |x|
        sigma_x = 0.1 + 0.1 * torch.abs(x)
        noise = torch.randn_like(x) * sigma_x

        y = y_true + noise

        return x, y, sigma_x


    class StandardNN(nn.Module):
        """평균만 예측하는 표준 뉴럴 넷트웍"""

        def __init__(self, hidden_size=50):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            return self.network(x)


    class HeteroscedasticNN(nn.Module):
        """
        평균과 분산을 함께 예측하는 뉴럴 넷트웍.

        이것이 이분산 회귀의 최대가능도 길이다!
        """

        def __init__(self, hidden_size=50):
            super().__init__()

            # 공유되는 은닉 레이어
            self.shared = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )

            # 평균 클래스
            self.mean_head = nn.Linear(hidden_size, 1)

            # 로그 분산 클래스 (σ² > 0을 보장하려고 log(σ²)을 예측한다)
            self.logvar_head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            """
            Returns:
                mean: 예측한 평균
                logvar: 예측한 로그 분산(log(σ²))
            """
            features = self.shared(x)
            mean = self.mean_head(features)
            logvar = self.logvar_head(features)
            return mean, logvar


    def gaussian_nll_loss(y_true, y_pred_mean, y_pred_logvar):
        """
        가우스 음의 로그 가능도 로스.

        이것이 이분산 회귀의 최대가능도 목표다!

        NLL = -log N(y | μ, σ²)
            = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]

        상수를 셈에서 빼면
        NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
            = 0.5 * [log_var + (y - μ)² / exp(log_var)]
        """
        # 음의 로그가능도를 계산한다
        variance = torch.exp(y_pred_logvar)
        loss = 0.5 * (y_pred_logvar + (y_true - y_pred_mean) ** 2 / variance)

        return loss.mean()


    def train_standard_nn(x_train, y_train, epochs=1000, lr=0.01):
        """MSE 로스로 표준 뉴럴 넷트웍을 학습시킨다"""

        model = StandardNN(hidden_size=50)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = []

        for epoch in range(epochs):
            # 순전파
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            if (epoch + 1) % 200 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        return model, history


    def train_heteroscedastic_nn(x_train, y_train, epochs=1000, lr=0.01):
        """사용자 정의 MLE 로스로 이분산 뉴럴 넷트웍을 학습시킨다"""

        model = HeteroscedasticNN(hidden_size=50)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []

        for epoch in range(epochs):
            # 순전파
            y_pred_mean, y_pred_logvar = model(x_train)

            # 음의 로그가능도를 계산한다 (우리의 MLE 목표이다!)
            loss = gaussian_nll_loss(y_train, y_pred_mean, y_pred_logvar)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            if (epoch + 1) % 200 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, NLL Loss: {loss.item():.4f}")

        return model, history


    def visualize_results(x, y, x_test, model_standard, model_hetero):
        """종합적인 시각화를 만든다"""

        fig = plt.figure(figsize=(18, 12))

        # 예측을 얻는다
        with torch.no_grad():
            # 표준 모델
            y_pred_standard = model_standard(x_test)

            # 이분산 모델
            y_pred_mean, y_pred_logvar = model_hetero(x_test)
            y_pred_std = torch.sqrt(torch.exp(y_pred_logvar))

        x_np = x.detach().numpy().flatten()
        y_np = y.detach().numpy().flatten()
        x_test_np = x_test.detach().numpy().flatten()

        # ================================================================
        # 그림 1: 표준 뉴럴 넷트웍의 예측
        # ================================================================
        ax1 = plt.subplot(2, 3, 1)

        # 그림을 그리기 위해 정렬한다
        sort_idx = torch.argsort(x_test.flatten())
        x_sorted = x_test_np[sort_idx]
        y_pred_sorted = y_pred_standard.detach().numpy().flatten()[sort_idx]

        ax1.scatter(x_np, y_np, alpha=0.5, s=20, label='Data', color='blue')
        ax1.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='Standard NN')
        ax1.plot(x_sorted, np.sin(x_sorted), 'g--', linewidth=2, label='True function')

        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Standard NN (MSE Loss)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ================================================================
        # 그림 2: 불확실성을 함께 낸 이분산 뉴럴 넷트웍
        # ================================================================
        ax2 = plt.subplot(2, 3, 2)

        y_mean_sorted = y_pred_mean.detach().numpy().flatten()[sort_idx]
        y_std_sorted = y_pred_std.detach().numpy().flatten()[sort_idx]

        ax2.scatter(x_np, y_np, alpha=0.5, s=20, label='Data', color='blue')
        ax2.plot(x_sorted, y_mean_sorted, 'r-', linewidth=2, label='Predicted mean')
        ax2.plot(x_sorted, np.sin(x_sorted), 'g--', linewidth=2, label='True function')

        # 불확실성 띠를 그린다 (±1σ, ±2σ)
        ax2.fill_between(x_sorted, 
                         y_mean_sorted - 2*y_std_sorted,
                         y_mean_sorted + 2*y_std_sorted,
                         alpha=0.2, color='red', label='±2σ (95% CI)')
        ax2.fill_between(x_sorted,
                         y_mean_sorted - y_std_sorted,
                         y_mean_sorted + y_std_sorted,
                         alpha=0.3, color='red', label='±1σ (68% CI)')

        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('Heteroscedastic NN (MLE Loss)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ================================================================
        # 그림 3: 예측 불확실성과 참 불확실성
        # ================================================================
        ax3 = plt.subplot(2, 3, 3)

        # sigma(x) = 0.1 + 0.1|x| 는 자료를 만들 때 쓴 참 함수이다.
        # true_sigma는 학습점 300개에서의 값이므로, 시험 격자 200개에
        # 맞추어 여기서 다시 계산한다.
        true_sigma_test = (0.1 + 0.1 * x_test.abs()).detach().numpy().flatten()
        true_sigma_sorted = true_sigma_test[sort_idx]

        ax3.plot(x_sorted, true_sigma_sorted, 'g-', linewidth=3, label='True σ(x)')
        ax3.plot(x_sorted, y_std_sorted, 'r-', linewidth=3, label='Predicted σ(x)')
        ax3.fill_between(x_sorted, 0, true_sigma_sorted, alpha=0.2, color='green')

        ax3.set_xlabel('x', fontsize=12)
        ax3.set_ylabel('σ (Standard Deviation)', fontsize=12)
        ax3.set_title('Uncertainty Estimation', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ================================================================
        # 그림 4: 잔차 비교
        # ================================================================
        ax4 = plt.subplot(2, 3, 4)

        residuals_standard = (y - model_standard(x)).detach().numpy().flatten()
        # 잔차와 보정은 학습점에서 재야 하므로 x_test가 아니라 x로 예측한다.
        # 아래 그림 5도 이 값을 쓴다.
        with torch.no_grad():
            mu_train, logvar_train = model_hetero(x)
            std_train = torch.sqrt(torch.exp(logvar_train))
        residuals_hetero = (y - mu_train).detach().numpy().flatten()

        ax4.scatter(x_np, residuals_standard, alpha=0.5, s=20, label='Standard NN', color='blue')
        ax4.scatter(x_np, residuals_hetero, alpha=0.5, s=20, label='Heteroscedastic NN', color='red')
        ax4.axhline(0, color='black', linestyle='--', linewidth=2)

        ax4.set_xlabel('x', fontsize=12)
        ax4.set_ylabel('Residuals', fontsize=12)
        ax4.set_title('Residual Analysis', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # ================================================================
        # 그림 5: 보정 그림
        # ================================================================
        ax5 = plt.subplot(2, 3, 5)

        # 이분산 모델의 정규화된 잔차를 계산한다
        residuals = residuals_hetero
        predicted_stds = std_train.detach().numpy().flatten()
        normalized_residuals = residuals / predicted_stds

        # 정규화된 잔차의 히스토그램 (잘 보정되었다면 N(0,1)이어야 한다)
        ax5.hist(normalized_residuals, bins=30, density=True, alpha=0.7, 
                edgecolor='black', label='Normalized Residuals')

        # N(0,1) 분포를 겹쳐 그린다
        x_range = np.linspace(-4, 4, 100)
        from scipy.stats import norm
        ax5.plot(x_range, norm.pdf(x_range), 'r-', linewidth=2, label='N(0,1)')

        ax5.set_xlabel('Normalized Residuals', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.set_title('Calibration Check', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # ================================================================
        # 그림 6: 로그가능도 비교
        # ================================================================
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # 로그가능도들을 계산한다
        with torch.no_grad():
            # 표준 모델 (σ = 1로 고정되어 있다고 가정)
            residuals_std = y - model_standard(x)
            nll_standard = 0.5 * (np.log(2 * np.pi) + torch.mean(residuals_std ** 2)).item()

            # 이분산 모델
            mean_het, logvar_het = model_hetero(x)
            nll_hetero = gaussian_nll_loss(y, mean_het, logvar_het).item()

        # 비교 표
        table_data = [
            ['Model', 'Negative Log-Likelihood'],
            ['Standard NN', f'{nll_standard:.4f}'],
            ['Heteroscedastic NN', f'{nll_hetero:.4f}'],
            ['Improvement', f'{nll_standard - nll_hetero:.4f}'],
        ]

        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)

        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 개선된 부분을 강조한다
        table[(3, 0)].set_facecolor('#FFF9C4')
        table[(3, 1)].set_facecolor('#FFF9C4')

        ax6.set_title('Model Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('neural_network_mle_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 Figure saved as 'neural_network_mle_results.png'")
        plt.show()


    def main():
        print("=" * 80)
        print("NEURAL NETWORK MLE - Deep Learning with Uncertainty")
        print("=" * 80)

        # 데이터를 생성한다
        print("\n🎲 Generating heteroscedastic data...")
        x_train, y_train, true_sigma = generate_heteroscedastic_data(n_samples=300)

        # 매끄러운 예측을 위한 시험 데이터
        x_test = torch.linspace(-5, 5, 200).unsqueeze(1)

        print(f"   • Training samples: {len(x_train)}")
        print(f"   • Noise varies with x (heteroscedastic)")

        # 표준 뉴럴 넷트웍을 학습시킨다
        print("\n🔵 Training Standard NN (MSE Loss)...")
        print("-" * 80)
        model_standard, history_standard = train_standard_nn(x_train, y_train, epochs=1000, lr=0.01)

        # 이분산 뉴럴 넷트웍을 학습시킨다
        print("\n🔴 Training Heteroscedastic NN (MLE Loss)...")
        print("-" * 80)
        model_hetero, history_hetero = train_heteroscedastic_nn(x_train, y_train, epochs=1000, lr=0.01)

        # 평가
        print("\n📊 Evaluation:")
        print("-" * 80)

        with torch.no_grad():
            # 표준 모델
            y_pred_std = model_standard(x_train)
            mse_std = torch.mean((y_train - y_pred_std) ** 2).item()

            # 이분산 모델
            y_pred_mean, y_pred_logvar = model_hetero(x_train)
            mse_het = torch.mean((y_train - y_pred_mean) ** 2).item()
            nll_het = gaussian_nll_loss(y_train, y_pred_mean, y_pred_logvar).item()

        print(f"   Standard NN:")
        print(f"      MSE: {mse_std:.4f}")

        print(f"\n   Heteroscedastic NN:")
        print(f"      MSE: {mse_het:.4f}")
        print(f"      NLL: {nll_het:.4f}")

        # 시각화한다
        print("\n📊 Creating visualizations...")
        visualize_results(x_train, y_train, x_test, model_standard, model_hetero)

        print("\n" + "=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print("\n💡 KEY TAKEAWAYS:")
        print("   1. Neural networks ARE MLE when trained with appropriate losses")
        print("   2. MSE = MLE with Gaussian assumption and fixed variance")
        print("   3. Heteroscedastic networks predict uncertainty!")
        print("   4. Custom loss functions = Custom probabilistic assumptions")
        print("   5. This enables uncertainty-aware deep learning")
        print("\n   🎯 Applications:")
        print("      • Medical diagnosis (quantify confidence)")
        print("      • Autonomous vehicles (safety-critical decisions)")
        print("      • Financial modeling (risk assessment)")
        print("      • Active learning (query uncertain points)")
        print("\n" + "=" * 80)


    """
    🎓 EXERCISES:

    1. 보통: 불확실성을 곁들인 분류
       - 분류 일로 넓힌다
       - 클래스 확률을 예측한다(소프트맥스)
       - 음의 로그 가능도(교차 엔트로피)를 쓴다
       - 예측의 자신도를 그림으로 본다

    2. 보통: 여러 잡음 모델
       - 라플라스 잡음: 제곱 대신 절대 오차를 쓴다
       - 스튜던트 t 잡음: 튄값에 든든하다
       - 가능도 함수를 견준다

    3. 어려움: 베이즈 뉴럴 넷트웍
       - 불확실성 어림을 위해 드롭아웃을 더한다
       - 몬테카를로 드롭아웃: 순전파를 여러 번 한다
       - 앎의 불확실성과 타고난 불확실성을 견준다

    4. 어려움: 출력이 여럿인 회귀
       - 공분산을 곁들여 벡터 출력을 예측한다
       - 온 공분산 행렬과 대각 행렬을 견준다
       - 다변량 가우스 가능도

    5. 어려움: 앞장선 학습
       - 불확실성으로 알려 주는 바가 큰 표본을 고른다
       - 작은 데이터셋으로 익힌다
       - 불확실성이 큰 점을 거듭 묻는다
       - 학습 굽이가 더 빨리 나아짐을 보인다
    """


    if __name__ == "__main__":
        main()
    ```


**출력:**

```
================================================================================
NEURAL NETWORK MLE - Deep Learning with Uncertainty
================================================================================

🎲 Generating heteroscedastic data...
   • Training samples: 300
   • Noise varies with x (heteroscedastic)

🔵 Training Standard NN (MSE Loss)...
--------------------------------------------------------------------------------
   Epoch 200/1000, Loss: 0.1387
   Epoch 400/1000, Loss: 0.1328
   Epoch 600/1000, Loss: 0.1311
   Epoch 800/1000, Loss: 0.1302
   Epoch 1000/1000, Loss: 0.1307

🔴 Training Heteroscedastic NN (MLE Loss)...
--------------------------------------------------------------------------------
   Epoch 200/1000, NLL Loss: -0.6561
   Epoch 400/1000, NLL Loss: -0.6560
   Epoch 600/1000, NLL Loss: -0.6850
   Epoch 800/1000, NLL Loss: -0.6934
   Epoch 1000/1000, NLL Loss: -0.7022

📊 Evaluation:
--------------------------------------------------------------------------------
   Standard NN:
      MSE: 0.1301

   Heteroscedastic NN:
      MSE: 0.1308
      NLL: -0.7024

📊 Creating visualizations...

📊 Figure saved as 'neural_network_mle_results.png'

================================================================================
✅ COMPLETE!
================================================================================

💡 KEY TAKEAWAYS:
   1. Neural networks ARE MLE when trained with appropriate losses
   2. MSE = MLE with Gaussian assumption and fixed variance
   3. Heteroscedastic networks predict uncertainty!
   4. Custom loss functions = Custom probabilistic assumptions
   5. This enables uncertainty-aware deep learning

   🎯 Applications:
      • Medical diagnosis (quantify confidence)
      • Autonomous vehicles (safety-critical decisions)
      • Financial modeling (risk assessment)
      • Active learning (query uncertain points)

================================================================================
```

## 2. 논의

### 서로 다른 출력과 목적

`StandardNN`은 평균 하나를 출력하고 MSE를 최소화한다. `HeteroscedasticNN`은 공유 피처에서 평균과 로그 분산을 각각 출력하며 가우스 NLL을 최소화한다. 입력에 따라 잡음이 커지는 합성 데이터에서 두 모델의 차이를 살펴본다.

### 로그 분산의 역할

로그 분산 $s$를 사용하면 분산은 $e^s>0$이다. 상수를 제외한 NLL은 $\frac12[s+(y-\mu)^2e^{-s}]$로, 오차를 줄이는 일과 분산을 조정하는 일을 함께 평가한다. 정확하게 적합한 관측의 분산을 0으로 보내는 과적합이나 극단적인 지수값도 발생할 수 있으므로 분산의 하한과 검증 성능을 점검해야 한다.

### 그림과 지표의 비교 조건

평균 주위의 표준편차 띠는 가정한 데이터 잡음의 크기를 나타낸다. 이는 평균 추정값의 신뢰구간이나 모든 모델 불확실성을 포함한 구간과 다르다. 학습 데이터의 표준화 잔차만으로 새 데이터의 보정을 확정할 수는 없다.

제공 코드의 표준 모델 NLL은 상수 항과 분산 1을 사용하고, 이분산 모델의 로스는 같은 상수 항을 생략한다. 두 숫자를 그대로 비교해 우열을 결론내리면 안 된다. 동일한 상수, 감소 방식과 평가 데이터에서 NLL을 맞춰 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 `hidden_size=50`인 `StandardNN`의 학습 파라미터를 계층별로 계산하라.

</div>

??? success "연습문제 1 풀이"
    선형 계층은 $1\to50$, $50\to50$, $50\to1$이다. 각각 $50+50=100$, $2500+50=2550$, $50+1=51$개이므로 총 2701개다. ReLU에는 파라미터가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
평균만 예측하는 모델과 이분산 모델의 NLL을 비교할 때 어떤 항과 조건을 맞춰야 하는가?

</div>

??? success "연습문제 2 풀이"
    두 모델의 정규분포 상수 항과 합·평균 방식, 분산 가정을 일치시키고 같은 독립 검증 데이터에서 계산한다. 서로 다른 상수를 생략한 로스 숫자나 학습 로스만으로 모델 우열을 판단하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
예측한 표준편차 띠를 평균 예측의 신뢰구간으로 부르면 안 되는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    출력 분산은 가정한 조건부 관측 잡음의 크기다. 평균 함수의 추정 불확실성과는 다르며, 모델 오지정이나 분포 변화도 자동으로 포함하지 않는다. 새 데이터에서 구간의 포함률과 보정을 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
레이어나 블록의 개수를 설정할 수 있도록 `StandardNN`를 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하라.

</div>

??? success "연습문제 4 풀이"
    고정되어 있던 레이어들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = StandardNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

뉴럴 넷트웍 MLE의 해석은 출력 분포와 로스의 정의에 달려 있다. 예측 분산의 의미를 확인하고, 지표를 비교할 때는 상수 항과 평가 조건을 일치시킨다.
