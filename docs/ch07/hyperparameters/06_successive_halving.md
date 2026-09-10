# 연속 반감법

연속 반감법과 하이퍼밴드는 자원을 아끼는 초매개변수 최적화 알고리즘이다. 적은 예산으로 많은 설정을 학습시킨 뒤 성적이 나쁜 것부터 차례로 떨어뜨리고, 가망 있는 설정에 자원을 더 준다. 모든 설정을 끝까지 학습시키는 것보다 훨씬 빠르다.

## 1. 코드

```python
"""
초매개변수 최적화를 위한 연속 반감법과 하이퍼밴드

이 모듈은 요즘의 초매개변수 최적화 알고리즘 두 가지를 구현한다:
1. 연속 반감법: 적은 예산으로 많은 설정을 학습시키고 나쁜 것부터 차례로
   떨어뜨리며 살아남은 설정에 예산을 늘려 준다.
2. 하이퍼밴드: 절충 매개변수를 달리하며 연속 반감법을 돌려 탐색과 활용의
   균형을 맞춘다.

이 알고리즘들은 가망 없는 설정을 일찍 버리므로 무작위 탐색이나 격자 탐색보다
효율적이다.

교육 목적: 5장 - 딥러닝 최적화와 초매개변수 조율
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
    """분류를 위한 간단한 2층 MLP."""
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
        input_dim: 특징의 차원
        num_classes: 출력 클래스의 수

    반환값:
        학습 집합과 검증 집합의 DataLoader
    """
    # 무작위 특징과 레이블 생성
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
        config: 학습률, 배치 크기 등을 담은 설정
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
    하이퍼밴드: 절충 매개변수를 달리하며 연속 반감법을 돌린다.

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

        # 연속 반감법의 브래킷을 훑기 (s 매개변수)
        for s in range(self.s_max, -1, -1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Hyperband Bracket s={s}")
                print(f"{'='*60}")

            # 브래킷별 매개변수 계산
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
    이 알고리즘들이 좋은 초매개변수를 효율적으로 찾는 모습을 보인다.
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
        SimpleMLPConfig(lr=lr, batch_size=bs)
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

## 2. 논의

연속 반감법은 많은 설정에 적은 예산(몇 에포크의 학습)을 나누어 주는 것으로 시작한다. 평가한 뒤 하위 절반을 버리고 살아남은 설정에 예산을 두 배로 준다. 설정이 하나만 남을 때까지 이를 되풀이하며, 그 설정은 전체 예산으로 학습된 셈이 된다. 핵심은 나쁜 설정을 대개 일찍 알아볼 수 있다는 점이다.

하이퍼밴드는 처음 설정과 예산을 달리하며 연속 반감법을 여러 번 돌려, 넓게 살펴보기(설정 많이, 예산 적게)와 깊게 평가하기(설정 적게, 예산 많이) 사이의 균형을 맞춘다. 각 "브래킷"은 이 절충 곡선 위의 서로 다른 지점에 해당한다.

두 알고리즘 모두 학습 비용이 크고 설정들 사이의 순위가 학습 초반에 자리 잡는 신경망의 초매개변수 최적화에 특히 효과적이다. 베이즈 최적화와 결합하면(BOHB처럼) 효율을 더 높일 수 있다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — 연속 반감법

연속 반감법은 많은 설정에 적은 예산(몇 에포크의 학습)을 나누어 주는 것으로 시작한다.

핵심 클래스는 `SimpleMLPConfig`, `SimpleMLP`, `SuccessiveHalvingScheduler`, `Hyperband`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
