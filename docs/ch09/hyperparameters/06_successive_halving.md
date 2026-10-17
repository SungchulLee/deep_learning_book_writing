# 연속 반감법

연속 반감법은 적은 자원으로 많은 후보를 평가하고 일부에 더 큰 자원을 배분한다. Hyperband는 초기 예산과 후보 수를 달리한 여러 브래킷을 실행하는 방식이다.

## 1. 코드

```python
"""
초파라메터 최적화를 위한 연속 반감법과 하이퍼밴드

이 모듈은 요즘의 초파라메터 최적화 알고리즘 두 가지를 구현한다:
1. 연속 반감법: 적은 예산으로 많은 설정을 학습시키고 나쁜 것부터 차례로
   떨어뜨리며 살아남은 설정에 예산을 늘려 준다.
2. 하이퍼밴드: 절충 파라메터를 달리하며 연속 반감법을 돌려 탐색과 활용의
   균형을 맞춘다.

이 알고리즘들은 가망 없는 설정을 일찍 버리므로 무작위 탐색이나 격자 탐색보다
효율적이다.

교육 목적: 10장 - 학습 파이프라인 (초파라메터 조율)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple
import random

# ========================================================================
# 메인
# ========================================================================


class SimpleMLPConfig:
    """간단한 MLP 모델의 설정."""
    def __init__(self, learning_rate: float, batch_size: int, hidden_dim: int = 64):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.id = random.randint(0, 1000000)

    def __repr__(self):
        return f"Config(lr={self.learning_rate:.2e}, bs={self.batch_size}, id={self.id})"


class SimpleMLP(nn.Module):
    """분류를 위한 간단한 2레이어 MLP."""
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def create_synthetic_data(n_samples: int = 1000, input_dim: int = 20, num_classes: int = 2):
    """
    합성 분류 데이터셋을 만든다.

    인수:
        n_samples: 표본의 총수
        input_dim: 피처의 차원
        num_classes: 출력 클래스의 수

    반환값:
        학습 집합과 검증 집합의 DataLoader
    """
    # 무작위 피처와 레이블 생성
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))

    # 학습(70%)과 검증(30%)으로 나누기
    train_size = int(0.7 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, y_train, X_val, y_val


def train_config_for_budget(
    config: SimpleMLPConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    device: str = 'cpu'
) -> float:
    """
    주어진 설정으로 정해진 에포크 수만큼 모델을 학습시킨다.

    인수:
        config: 러닝레이트, 배치 크기 등을 담은 설정
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        epochs: 학습할 에포크 수
        device: 학습에 쓸 장치

    반환값:
        최종 검증 정확도
    """
    # 모델과 최적화기 만들기
    model = SimpleMLP(hidden_dim=config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 설정의 배치 크기로 데이터 로더 만들기
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # 학습 루프
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # 검증 집합에서 평가
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        val_preds = val_logits.argmax(dim=1)
        val_accuracy = (val_preds == y_val.to(device)).float().mean().item()

    return val_accuracy


class SuccessiveHalvingScheduler:
    """
    연속 반감법: 적은 예산으로 많은 설정을 시작하여 상위 1/eta만 남기고
    예산을 eta배로 늘리기를 되풀이한다.

    알고리즘:
    1. 설정 n개를 뽑는다
    2. 각각에 자원 예산 r_min을 준다
    3. 모두 평가하여 상위 ceil(n / eta)개를 남긴다
    4. 예산을 늘린다: r_min *= eta
    5. 설정이 하나 남을 때까지 되풀이한다

    인수:
        eta: 줄이고 늘리는 인수 (보통 2나 3)
        r_min: 최소 예산 (예: 에포크)
        r_max: 최대 예산
    """

    def __init__(self, eta: float = 2.0, r_min: int = 1, r_max: int = 32):
        self.eta = eta
        self.r_min = r_min
        self.r_max = r_max
        self.s_max = math.floor(math.log(r_max / r_min) / math.log(eta))

    def run(
        self,
        configs: List[SimpleMLPConfig],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: str = 'cpu',
        verbose: bool = True
    ) -> Tuple[SimpleMLPConfig, float]:
        """
        주어진 설정들에 연속 반감법을 돌린다.

        인수:
            configs: 탐색할 설정의 목록
            X_train, y_train, X_val, y_val: 학습 데이터와 검증 데이터
            device: 학습에 쓸 장치
            verbose: 진행 상황을 출력할지 여부

        반환값:
            가장 좋은 설정과 그 검증 정확도
        """
        remaining_configs = configs[:]
        budget = self.r_min

        stage = 0
        while len(remaining_configs) > 1:
            if verbose:
                print(f"\nStage {stage}: {len(remaining_configs)} configs, budget={budget} epochs")

            # 현재 예산으로 남은 설정을 모두 평가
            scores = []
            for config in remaining_configs:
                acc = train_config_for_budget(
                    config, X_train, y_train, X_val, y_val,
                    epochs=budget, device=device
                )
                scores.append((config, acc))
                if verbose:
                    print(f"  {config} -> accuracy={acc:.4f}")

            # 점수로 정렬 (내림차순)
            scores.sort(key=lambda x: x[1], reverse=True)

            # 상위 1/eta 설정만 남기기
            num_keep = max(1, int(len(remaining_configs) / self.eta))
            remaining_configs = [config for config, _ in scores[:num_keep]]

            if verbose:
                print(f"  Keeping top {num_keep} config(s)")

            # 다음 단계를 위해 예산 늘리기
            budget = int(budget * self.eta)
            budget = min(budget, self.r_max)
            stage += 1

        # 가장 좋은 설정의 최종 평가
        best_config = remaining_configs[0]
        final_acc = train_config_for_budget(
            best_config, X_train, y_train, X_val, y_val,
            epochs=self.r_max, device=device
        )

        if verbose:
            print(f"\nBest config: {best_config}")
            print(f"Final accuracy at r_max={self.r_max}: {final_acc:.4f}")

        return best_config, final_acc


class Hyperband:
    """
    하이퍼밴드: 절충 파라메터를 달리하며 연속 반감법을 돌린다.

    하이퍼밴드는 R(최대 예산)과 n(처음 설정의 수)의 쌍을 달리한 여러 브래킷에서
    연속 반감법을 돌린다. 이로써 다음의 균형을 맞춘다:
    - 적은 예산으로 많은 설정 (넓은 탐색)
    - 많은 예산으로 적은 설정 (깊은 탐색)

    인수:
        eta: 연속 반감법의 인수 (보통 2나 3)
        R: 설정마다의 최대 자원 (전체 예산)
        B: 쓸 전체 예산 (에포크 × 설정 수)
    """

    def __init__(self, eta: float = 2.0, R: int = 81, B: int = 5):
        self.eta = eta
        self.R = R
        self.B = B  # 브래킷당 예산
        self.s_max = math.floor(math.log(R) / math.log(eta))

    def run(
        self,
        configs: List[SimpleMLPConfig],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: str = 'cpu',
        verbose: bool = True
    ) -> Tuple[SimpleMLPConfig, float]:
        """
        하이퍼밴드를 돌린다 (연속 반감법 브래킷 여러 개).

        인수:
            configs: 뽑아 쓸 설정의 모음
            X_train, y_train, X_val, y_val: 학습 데이터와 검증 데이터
            device: 학습에 쓸 장치
            verbose: 진행 상황을 출력할지 여부

        반환값:
            찾아낸 가장 좋은 설정과 그 정확도
        """
        best_config = None
        best_acc = 0.0

        # 연속 반감법의 브래킷을 훑기 (s 파라메터)
        for s in range(self.s_max, -1, -1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Hyperband Bracket s={s}")
                print(f"{'='*60}")

            # 브래킷별 파라메터 계산
            n = math.ceil((self.B / self.R) * (self.s_max + 1) / (s + 1))
            r_min = self.R / (self.eta ** s)

            # 이 브래킷을 위해 설정 n개 뽑기
            bracket_configs = random.sample(configs, min(int(n), len(configs)))

            # 이 브래킷에 연속 반감법 실행
            scheduler = SuccessiveHalvingScheduler(
                eta=self.eta,
                r_min=max(1, int(r_min)),
                r_max=self.R
            )
            bracket_best_config, bracket_best_acc = scheduler.run(
                bracket_configs, X_train, y_train, X_val, y_val,
                device=device, verbose=verbose
            )

            # 전역 최고 기록 갱신
            if bracket_best_acc > best_acc:
                best_acc = bracket_best_acc
                best_config = bracket_best_config

        return best_config, best_acc


def main():
    """
    시연: 합성 데이터에 연속 반감법과 하이퍼밴드를 돌린다.
    이 알고리즘들이 좋은 초파라메터를 효율적으로 찾는 모습을 보인다.
    """
    print("Deep Learning Hyperparameter Optimization: Successive Halving & Hyperband")
    print("=" * 70)

    # 재현성을 위해 난수 씨앗 고정
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 합성 데이터셋 만들기
    print("\nGenerating synthetic data...")
    X_train, y_train, X_val, y_val = create_synthetic_data(n_samples=2000, input_dim=20)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # 탐색할 무작위 설정 생성
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [16, 32, 64, 128]
    configs = [
        SimpleMLPConfig(learning_rate=lr, batch_size=bs)
        for lr in learning_rates
        for bs in batch_sizes
    ]
    print(f"\nSearching over {len(configs)} configurations")

    # 연속 반감법 실행
    print("\n" + "="*70)
    print("SUCCESSIVE HALVING")
    print("="*70)
    sh_scheduler = SuccessiveHalvingScheduler(eta=2.0, r_min=1, r_max=16)
    sh_best_config, sh_best_acc = sh_scheduler.run(
        configs, X_train, y_train, X_val, y_val, verbose=True
    )

    # 하이퍼밴드 실행
    print("\n" + "="*70)
    print("HYPERBAND")
    print("="*70)
    hyperband = Hyperband(eta=2.0, R=16, B=5)
    hb_best_config, hb_best_acc = hyperband.run(
        configs, X_train, y_train, X_val, y_val, verbose=False
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successive Halving best: {sh_best_config} with accuracy {sh_best_acc:.4f}")
    print(f"Hyperband best: {hb_best_config} with accuracy {hb_best_acc:.4f}")


if __name__ == "__main__":
    main()
```

**출력:**

```
Deep Learning Hyperparameter Optimization: Successive Halving & Hyperband
======================================================================

Generating synthetic data...
Training set: torch.Size([1400, 20]), Validation set: torch.Size([600, 20])

Searching over 16 configurations

======================================================================
SUCCESSIVE HALVING
======================================================================

Stage 0: 16 configs, budget=1 epochs
  Config(lr=1.00e-04, bs=16, id=670487) -> accuracy=0.5450
  Config(lr=1.00e-04, bs=32, id=116739) -> accuracy=0.4733
  Config(lr=1.00e-04, bs=64, id=26225) -> accuracy=0.4983
  Config(lr=1.00e-04, bs=128, id=777572) -> accuracy=0.4800
  Config(lr=1.00e-03, bs=16, id=288389) -> accuracy=0.5200
  Config(lr=1.00e-03, bs=32, id=256787) -> accuracy=0.5333
  Config(lr=1.00e-03, bs=64, id=234053) -> accuracy=0.5000
  Config(lr=1.00e-03, bs=128, id=146316) -> accuracy=0.4917
  Config(lr=1.00e-02, bs=16, id=772246) -> accuracy=0.5283
  Config(lr=1.00e-02, bs=32, id=107473) -> accuracy=0.5283
  Config(lr=1.00e-02, bs=64, id=709570) -> accuracy=0.5267
  Config(lr=1.00e-02, bs=128, id=776646) -> accuracy=0.5467
  Config(lr=1.00e-01, bs=16, id=935518) -> accuracy=0.5217
  Config(lr=1.00e-01, bs=32, id=571858) -> accuracy=0.5283
  Config(lr=1.00e-01, bs=64, id=91161) -> accuracy=0.5383
  Config(lr=1.00e-01, bs=128, id=619176) -> accuracy=0.5167
  Keeping top 8 config(s)

Stage 1: 8 configs, budget=2 epochs
  Config(lr=1.00e-02, bs=128, id=776646) -> accuracy=0.5317
  Config(lr=1.00e-04, bs=16, id=670487) -> accuracy=0.5050
  Config(lr=1.00e-01, bs=64, id=91161) -> accuracy=0.4617
  Config(lr=1.00e-03, bs=32, id=256787) -> accuracy=0.4917
  Config(lr=1.00e-02, bs=16, id=772246) -> accuracy=0.4750
  Config(lr=1.00e-02, bs=32, id=107473) -> accuracy=0.5217
  Config(lr=1.00e-01, bs=32, id=571858) -> accuracy=0.5267
  Config(lr=1.00e-02, bs=64, id=709570) -> accuracy=0.4867
  Keeping top 4 config(s)

Stage 2: 4 configs, budget=4 epochs
  Config(lr=1.00e-02, bs=128, id=776646) -> accuracy=0.5000
  Config(lr=1.00e-01, bs=32, id=571858) -> accuracy=0.5267
  Config(lr=1.00e-02, bs=32, id=107473) -> accuracy=0.5167
  Config(lr=1.00e-04, bs=16, id=670487) -> accuracy=0.5083
  Keeping top 2 config(s)

Stage 3: 2 configs, budget=8 epochs
  Config(lr=1.00e-01, bs=32, id=571858) -> accuracy=0.4733
  Config(lr=1.00e-02, bs=32, id=107473) -> accuracy=0.5050
  Keeping top 1 config(s)

Best config: Config(lr=1.00e-02, bs=32, id=107473)
Final accuracy at r_max=16: 0.5367

======================================================================
HYPERBAND
======================================================================

======================================================================
SUMMARY
======================================================================
Successive Halving best: Config(lr=1.00e-02, bs=32, id=107473) with accuracy 0.5367
Hyperband best: Config(lr=1.00e-04, bs=32, id=116739) with accuracy 0.5417
```

## 2. 논의

이 구현은 각 단계마다 train_config_for_budget에서 새 모델을 생성한다. 앞 단계의 웨이트와 옵티마이저를 이어서 학습하는 방식이 아니라 예산만 늘려 처음부터 재학습한다. 최후 후보도 r_max 예산으로 새로 학습해 최종 점수를 얻는다.

후보 유지 수는 int(n/eta)로 내림하고 최소 하나를 남긴다. eta=2일 때 절반이고 일반 eta에서는 비율이 다르다. 문서의 ceil 설명과 코드의 내림은 같지 않다. eta가 1 이하이거나 초기 후보가 비어 있는 경우처럼 진행이 유효하지 않은 설정도 검사해야 한다.

초기 자원에서의 순위가 최종 성능을 충분히 반영하지 않으면 좋은 후보를 일찍 버릴 수 있다. 단계·브래킷별 재학습 비용과 난수 변동을 포함해 비교하며, 가장 좋은 검증 점수는 선택 결과이지 독립 최종 성능은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
후보 7개, eta=2이면 코드가 몇 개를 남기는가?

</div>

??? success "풀이"
    int(7/2)=3개이다. ceil이면 4개이므로 주석의 설명과 다르다. 이후 단계의 후보 수와 예산도 실제 코드 기준으로 계산한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 구현에서 예산 2와 4의 단계는 총 4에포크 학습과 같은가?

</div>

??? success "풀이"
    아니다. 새 모델로 각각 2와 4에포크를 학습하므로 후보가 살아남는 동안 총 6에포크 비용이 든다. 이어서 2에포크를 추가하는 체크포인트 재개와 다르다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
초기 성능이 나쁜 후보를 버리는 기준의 위험을 설명하라.

</div>

??? success "풀이"
    워밍업이나 작은 러닝레이트 때문에 늦게 좋아지는 후보는 초기 평가에서 탈락할 수 있다. 최소 예산과 단계 계수를 검증하고 동일한 충분한 예산의 기준 결과와 비교한다.

## 정리하며

자원 적응형 탐색은 초기 평가로 후보를 줄인다. 구현의 재학습·재개 방식과 내림 규칙, 초기 순위의 신뢰성을 확인하고 선택 후 최종 평가를 따로 유지한다.
