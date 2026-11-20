# 낮은 신호 대 잡음비

금융 시장은 신호 대 잡음비(signal-to-noise ratio, SNR)가 아주 낮아 일간 샤프 비율이 흔히 0.02~0.04쯤이다. 진짜 예측 신호가 엄청난 잡음에 파묻혀 있다는 뜻이며, 작은 우위라도 뽑아내어 검증하려면 앙상블, 데이터 증강, 엄밀한 통계 검증 같은 특별한 기법이 있어야 한다.

## 1. 코드

```python
"""
35.5.3장: 낮은 신호 대 잡음
=====================================
잡음 많은 금융 자료에서 신호를 뽑아내는 재주.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


@dataclass
class SNRConfig:
    num_ensemble: int = 5
    noise_std: float = 0.001
    bootstrap_samples: int = 100


class SNRAnalyzer:
    """신호 대 잡음 비를 살피고 잰다."""

    @staticmethod
    def compute_snr(returns: np.ndarray) -> Dict[str, float]:
        mean = np.mean(returns)
        std = np.std(returns) + 1e-8
        daily_snr = abs(mean) / std
        annual_sharpe = mean / std * np.sqrt(252)
        return {
            "daily_snr": float(daily_snr),
            "annual_sharpe": float(annual_sharpe),
            "mean": float(mean),
            "std": float(std),
            "required_days_for_significance": int((2.0 / daily_snr) ** 2) if daily_snr > 0 else 999999,
        }


class EnsembleAgent:
    """잡음을 줄이려 여러 클래스 에이전트를 무리 짓는다."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int = 5):
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, action_dim), nn.Tanh(),
            ) for _ in range(num_agents)
        ])

    def predict(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            predictions = torch.stack([agent(state) for agent in self.agents])
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            return {"mean": mean, "std": std, "individual": predictions}


class DataAugmenter:
    """금융 때 열을 위한 자료 늘리기."""

    @staticmethod
    def add_noise(data: np.ndarray, noise_std: float = 0.001) -> np.ndarray:
        return data + np.random.randn(*data.shape) * noise_std

    @staticmethod
    def bootstrap_sample(data: np.ndarray, block_size: int = 20) -> np.ndarray:
        T = len(data)
        n_blocks = T // block_size + 1
        indices = np.random.randint(0, T - block_size, n_blocks)
        blocks = [data[i:i + block_size] for i in indices]
        return np.concatenate(blocks)[:T]

    @staticmethod
    def time_reversal(returns: np.ndarray) -> np.ndarray:
        return returns[::-1].copy()


def demo_snr():
    """신호 대 잡음 살피기와 누그러뜨리기를 보인다."""
    print("=" * 70)
    print("Low Signal-to-Noise Ratio Analysis")
    print("=" * 70)

    np.random.seed(42)
    # 참에 가까운 금융 돌아옴(낮은 신호 대 잡음)
    T = 1000
    signal = 0.0003  # 해마다 약 7.5% 돌아옴
    noise = 0.015    # 해마다 약 24% 흔들림
    returns = np.random.randn(T) * noise + signal

    analyzer = SNRAnalyzer()
    snr = analyzer.compute_snr(returns)
    print(f"\n--- SNR Analysis ---")
    print(f"Daily SNR: {snr['daily_snr']:.4f}")
    print(f"Annual Sharpe: {snr['annual_sharpe']:.4f}")
    print(f"Days needed for significance: {snr['required_days_for_significance']}")

    # 무리 짓기
    print("\n--- Ensemble Agent ---")
    ensemble = EnsembleAgent(state_dim=10, action_dim=5, num_agents=5)
    state = torch.randn(1, 10)
    result = ensemble.predict(state)
    print(f"Ensemble mean: {result['mean'][0].numpy()}")
    print(f"Ensemble std:  {result['std'][0].numpy()}")
    print(f"Disagreement:  {result['std'].mean().item():.4f}")

    # 자료 늘리기
    print("\n--- Data Augmentation ---")
    aug = DataAugmenter()
    noisy = aug.add_noise(returns, 0.001)
    boot = aug.bootstrap_sample(returns, block_size=20)
    rev = aug.time_reversal(returns)
    print(f"Original mean: {np.mean(returns):.6f}")
    print(f"Noisy mean:    {np.mean(noisy):.6f}")
    print(f"Bootstrap mean:{np.mean(boot):.6f}")
    print(f"Reversed mean: {np.mean(rev):.6f}")


if __name__ == "__main__":
    demo_snr()
```

**출력:**

```
======================================================================
Low Signal-to-Noise Ratio Analysis
======================================================================

--- SNR Analysis ---
Daily SNR: 0.0402
Annual Sharpe: 0.6379
Days needed for significance: 2476

--- Ensemble Agent ---
Ensemble mean: [ 0.05910273  0.00556063  0.03426272  0.04864367 -0.08268502]
Ensemble std:  [0.15709348 0.16061704 0.09600415 0.05470599 0.11995234]
Disagreement:  0.1177

--- Data Augmentation ---
Original mean: 0.000590
Noisy mean:    0.000661
Bootstrap mean:0.000089
Reversed mean: 0.000590
```

## 2. 논의

SNR 분석은 금융 예측이 얼마나 어려운지를 수치로 보여 준다. 일간 SNR이 0.02(연율 샤프 비율 약 0.3에 해당)이면 통계적 유의성을 확립하는 데 독립 관측이 수천 개 있어야 한다. 필요한 날수 식 $T = (z / \text{SNR})^2$에 따르면 95% 신뢰 수준에서 연율 샤프 0.5를 탐지하려면 일간 데이터가 15년 넘게(약 3,900 거래일) 있어야 하는데, 이는 대부분의 전략이 가진 이력보다 훨씬 길다. 위 코드는 $z = 2$를 쓰며, 출력의 연율 샤프 0.6379에 대해 2,476일, 곧 10년쯤이 나온다.

앙상블 에이전트는 여러 모델을 묶어 예측의 분산을 줄인다. 이 코드의 에이전트들은 구조는 같고 초기값만 달라, 어느 정도 서로 다른 추정을 낸다. 앙상블 크기를 $N$이라 할 때 오차가 서로 독립이면 앙상블 평균의 표준편차는 $1/\sqrt{N}$배로(분산은 $1/N$배로) 작아지므로, 묶은 예측의 SNR이 실제로 오른다. 앙상블의 표준편차는 포지션 크기를 정하는 데 쓸 수 있는 불확실성 추정도 덤으로 준다.

금융 시계열의 데이터 증강에는 이 분야에 맞는 기법이 있어야 한다. 작은 가우스 잡음을 더하면 전체 분포를 거의 유지하면서 새 학습 샘플을 만든다. 블록 부트스트랩 재샘플링은 짧은 시간 규모의 상관 구조를 유지하는 합성 경로를 만든다. 시간 뒤집기는 (수익률 과정이 시간을 뒤집어도 통계적으로 같다고 볼 수 있을 때) 타당한 증강 수익률 열을 만들어 데이터셋 크기를 두 배로 늘린다. 출력에서 부트스트랩 표본의 평균(0.000089)이 원래 평균(0.000590)과 크게 다른 것도 SNR이 낮다는 증거다. 1,000일치 평균의 표준 오차가 $0.015/\sqrt{1000} \approx 0.00047$로, 평균 자체와 크기가 비슷하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기대 수익률이 연 8%이고 변동성이 연 25%인 전략에 대해 일간 SNR과, 99% 신뢰 수준에서 통계적으로 유의하려면 필요한 최소 거래일 수를 계산하여라.

</div>

??? success "풀이"
    일간 기대 수익률: $\mu = 0.08/252 \approx 0.000317$. 일간 변동성: $\sigma = 0.25/\sqrt{252} \approx 0.01575$.

    일간 SNR: $\mu/\sigma \approx 0.0201$.

    99% 신뢰 수준에서 $z = 2.576$이다.

    최소 날수: $T = (z/\text{SNR})^2 = (2.576/0.0201)^2 \approx 16{,}425$일 $\approx 65$년.

    이것이 근본적인 어려움을 보여 준다. 연율 샤프 비율 0.32라는 그럴듯한 전략조차 99% 신뢰 수준에서 유의성을 확립하려면 데이터가 65년 넘게 있어야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
앙상블의 불일치(앙상블 예측 사이의 표준편차)가 SNR이 낮은 환경에서 포지션 크기를 정하는 데 쓸모 있는 신호인 까닭을 설명하여라.

</div>

??? success "풀이"
    앙상블 에이전트들의 예측이 크게 엇갈리면(표준편차가 크면) 예측을 믿기 어렵다. 모델마다 잡음 속에서 서로 다른 패턴을 보고 있는 것이다. 예측이 일치하면(표준편차가 작으면) 그 신호가 진짜일 확률이 더 크다.

    불일치의 역수에 비례해 포지션 크기를 정하면(예측이 일치하면 포지션을 키우고 엇갈리면 줄이면) 사실상 확신도에 따라 가중치를 주는 셈이 된다. SNR이 낮은 환경에서 이 필터는 잡음 많은 예측에 크게 걸지 않으면서 확신도가 높은 신호에 자본을 모아, 실현 샤프 비율을 높인다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
원래 데이터의 주변 분포와 자기상관 구조를 함께 유지하는 합성 수익률 열을 만드는 블록 부트스트랩 데이터 증강 함수를 작성하여라.

</div>

??? success "풀이"
    ```python
    def block_bootstrap_augment(returns, n_synthetic=5, block_size=20):
        T = len(returns)
        synthetic_series = []

        for _ in range(n_synthetic):
            n_blocks = T // block_size + 1
            block_starts = np.random.randint(0, T - block_size, n_blocks)
            blocks = [returns[s:s+block_size] for s in block_starts]
            synthetic = np.concatenate(blocks)[:T]
            synthetic_series.append(synthetic)

        return synthetic_series
    ```
    블록 부트스트랩은 블록 안의 시간 의존성(자기상관, 변동성 군집)을 지키면서 블록을 무작위 순서로 이어 붙인다. 블록 크기는 상관의 시간 스케일에 맞추어야 한다. 너무 작으면 블록 경계마다 시간 의존성이 끊기고, 너무 크면 서로 다른 블록의 개수가 줄어 합성 경로의 다양성이 떨어진다.

## 정리하며

**다룬 것** — 낮은 신호 대 잡음비

SNR 분석은 금융 예측이 얼마나 어려운지를 수치로 보여 준다.

핵심 클래스는 `SNRConfig`, `SNRAnalyzer`, `EnsembleAgent`, `DataAugmenter`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
