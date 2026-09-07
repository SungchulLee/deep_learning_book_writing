# 낮은 신호 대 잡음 비

금융 저자는 신호 대 잡음 비가 아주 낮아 하루 샤프 비가 흔히 0.02~0.04쯤이다. 참된 미리 보기 신호가 어마어마한 잡음에 파묻혀 있다는 뜻이며, 얼마간의 앞섬을 뽑아내어 살피려면 무리 짓기, 자료 늘리기, 꼼꼼한 통계 살피기 같은 남다른 재주가 있어야 한다.

## 코드

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
    """잡음을 줄이려 여러 갈래 부림꾼을 무리 짓는다."""

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

## 논의

신호 대 잡음 살피기는 금융 미리 보기가 얼마나 어려운지를 수로 보인다. 하루 신호 대 잡음이 0.02(해마다 샤프 약 0.3에 맞선다)이면 통계로 뜻있음을 세우는 데 서로 매이지 않은 눈금이 수천 개 있어야 한다. 필요한 날수 식 $T = (z / \text{SNR})^2$은 95% 믿음에서 샤프 0.5을 알아내려면 하루 자료가 40년 넘게 있어야 함을 드러내는데, 이는 거의 모든 이가 손에 쥔 것보다 훨씬 많다.

무리 부림꾼은 여러 갈래 모형을 엮어 미리 보기의 흩어짐을 줄인다. 무리 안의 부림꾼마다 얼개나 첫 값이 달라 서로 매이지 않은 어림을 낸다. 무리 크기를 $N$이라 할 때 무리 평균의 흩어짐은 $1/\sqrt{N}$만큼 작아지므로 엮은 미리 보기의 신호 대 잡음이 실제로 오른다. 무리의 표준편차는 자리 크기를 잡는 데 쓸 수 있는 헤아릴 수 없음 어림도 절로 준다.

금융 때 열의 자료 늘리기에는 이 마당에 맞는 재주가 있어야 한다. 작은 가우스 잡음을 더하면 온 분포를 지키면서 새 익힘 보기를 만든다. 덩이 부트스트랩 다시 뽑기는 짧은 때의 얽힘 짜임을 지키는 가짜 자취를 지어낸다. 때 뒤집기는 (로그 돌아옴이 때를 뒤집어도 같다고 여길 때) 옳은 늘린 돌아옴 열을 만들어 실제 자료 뭉치 크기를 곱절로 늘린다.

## 연습문제

**연습문제 1.**
해마다 어림 돌아옴이 8%이고 해마다 흔들림이 25%인 꾀에 대해 하루 신호 대 잡음과, 99% 믿음에서 통계로 뜻있으려면 있어야 하는 가장 적은 거래 날수를 셈하여라.

??? success "연습문제 1 풀이"
    하루 어림 돌아옴: $\mu = 0.08/252 \approx 0.000317$. 하루 흔들림: $\sigma = 0.25/\sqrt{252} \approx 0.01575$.

    하루 신호 대 잡음: $\mu/\sigma \approx 0.0201$.

    99% 믿음에서 $z = 2.576$이다.

    가장 적은 날수: $T = (z/\text{SNR})^2 = (2.576/0.0201)^2 \approx 16{,}425$일 $\approx 65$년.

    이것이 근본 어려움을 보여 준다. 해마다 샤프 0.32이라는 그럴듯한 꾀조차 99% 믿음에서 뜻있음을 굳히려면 자료가 65년 넘게 있어야 한다.

---


**연습문제 2.**
무리의 엇갈림(무리 미리 보기 사이의 표준편차)이 신호 대 잡음이 낮은 자리에서 자리 크기를 잡는 데 쓸모 있는 신호인 까닭을 풀어라.

??? success "연습문제 2 풀이"
    무리 부림꾼이 크게 엇갈리면(표준편차가 크면) 미리 보기가 헤아릴 수 없다. 모형마다 잡음 속에서 다른 결을 보고 있는 것이다. 뜻이 맞으면(표준편차가 작으면) 그 신호가 참일 낌새가 더 크다.

    엇갈림의 거꾸로에 견주어 자리 크기를 잡으면(무리의 뜻이 맞으면 자리를 키우고 엇갈리면 줄이면) 사실상 믿음에 맞춰 무게를 주는 셈이 된다. 신호 대 잡음이 낮은 자리에서 이 거르기 장치는 잡음 많은 미리 보기에 크게 걸지 않으면서 믿음이 높은 신호에 밑천을 모아, 실제로 얻는 샤프 비를 높인다.

---


**연습문제 3.**
처음 자료의 가장자리 분포와 스스로 얽힘 짜임을 함께 지키는 가짜 돌아옴 열을 지어내는 덩이 부트스트랩 자료 늘리기 함수를 만들어라.

??? success "연습문제 3 풀이"
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
    덩이 부트스트랩은 덩이 안의 때 매임(스스로 얽힘, 흔들림 뭉침)을 지키면서 덩이의 차례만 아무렇게나 섞는다. 덩이 크기는 얽힘의 때 잣대에 맞추어야 한다. 너무 작으면 잇단 매임이 무너지고, 너무 크면 서로 다른 덩이의 개수가 줄어 지어낸 자취의 여러 갈래가 좁아진다.
