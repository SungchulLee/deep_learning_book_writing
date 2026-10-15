# 학습률 워밍업

학습률 워밍업은 초기 갱신에 작은 학습률을 사용하고 목표값까지 점진적으로 올리는 방법이다. 선형·지수·코사인 보간과 워밍업 뒤의 코사인 감쇠를 직접 구현한다.

## 1. 코드

```python
"""
학습률 워밍업 구현

이 모듈은 심층 신경망 학습에 흔히 쓰이는 여러 학습률 워밍업 전략을
제공한다.
"""

import math
from typing import Optional

# ========================================================================
# 메인
# ========================================================================


class LinearWarmup:
    """
    선형 학습률 워밍업.
    warmup_steps 동안 학습률을 0에서 base_lr까지 선형으로 올린다.
    """
    
    def __init__(self, base_lr: float, warmup_steps: int):
        """
        인수:
            base_lr: 워밍업 뒤의 목표 학습률
            warmup_steps: 워밍업 구간의 단계 수
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        return self.base_lr


class ExponentialWarmup:
    """
    지수 학습률 워밍업.
    학습률을 start_lr에서 base_lr까지 지수적으로 올린다.
    """
    
    def __init__(self, base_lr: float, warmup_steps: int, start_lr: float = 1e-7):
        """
        인수:
            base_lr: 워밍업 뒤의 목표 학습률
            warmup_steps: 워밍업 구간의 단계 수
            start_lr: 처음 학습률 (아주 작아야 한다)
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        if step < self.warmup_steps:
            # 지수 보간
            factor = (step + 1) / self.warmup_steps
            return self.start_lr * (self.base_lr / self.start_lr) ** factor
        return self.base_lr


class CosineWarmup:
    """
    코사인 학습률 워밍업.
    코사인 곡선을 따라 학습률을 매끄럽게 올린다.
    """
    
    def __init__(self, base_lr: float, warmup_steps: int):
        """
        인수:
            base_lr: 워밍업 뒤의 목표 학습률
            warmup_steps: 워밍업 구간의 단계 수
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        if step < self.warmup_steps:
            # 0에서 base_lr까지 코사인 보간
            progress = (step + 1) / self.warmup_steps
            return self.base_lr * (1 - math.cos(progress * math.pi)) / 2
        return self.base_lr


class WarmupWithDecay:
    """
    워밍업과 학습률 감쇠를 결합한다.
    선형으로 워밍업한 뒤 코사인 감쇠를 적용한다.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        """
        인수:
            base_lr: 워밍업 뒤의 최고 학습률
            warmup_steps: 워밍업 구간의 단계 수
            total_steps: 전체 학습 단계 수
            min_lr: 학습이 끝날 때의 최소 학습률
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        if step < self.warmup_steps:
            # 선형 워밍업
            return self.base_lr * (step + 1) / self.warmup_steps
        
        # 워밍업 뒤의 코사인 감쇠
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


def plot_warmup_schedules(warmup_steps: int = 1000, total_steps: int = 10000):
    """
    여러 워밍업 일정을 그려 보인다.
    matplotlib이 필요하다.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    base_lr = 1e-3
    
    schedulers = {
        'Linear Warmup': LinearWarmup(base_lr, warmup_steps),
        'Exponential Warmup': ExponentialWarmup(base_lr, warmup_steps),
        'Cosine Warmup': CosineWarmup(base_lr, warmup_steps),
        'Warmup + Cosine Decay': WarmupWithDecay(base_lr, warmup_steps, total_steps)
    }
    
    steps = range(min(total_steps, 5000))
    
    plt.figure(figsize=(12, 6))
    for name, scheduler in schedulers.items():
        lrs = [scheduler.get_lr(step) for step in steps]
        plt.plot(steps, lrs, label=name, linewidth=2)
    
    plt.axvline(x=warmup_steps, color='red', linestyle='--', 
                label=f'Warmup End ({warmup_steps} steps)', alpha=0.5)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Warmup Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('warmup_schedules.png', dpi=150)
    print("Plot saved to warmup_schedules.png")


if __name__ == "__main__":
    # 사용 예
    print("Learning Rate Warmup Examples\n")
    
    base_lr = 1e-3
    warmup_steps = 1000
    
    # 여러 워밍업 전략 시험
    schedulers = {
        'Linear': LinearWarmup(base_lr, warmup_steps),
        'Exponential': ExponentialWarmup(base_lr, warmup_steps),
        'Cosine': CosineWarmup(base_lr, warmup_steps),
    }
    
    test_steps = [0, 250, 500, 750, 1000, 1500]
    
    for name, scheduler in schedulers.items():
        print(f"{name} Warmup:")
        for step in test_steps:
            lr = scheduler.get_lr(step)
            print(f"  Step {step:4d}: lr = {lr:.6f}")
        print()
    
    # 시각화 만들기
    plot_warmup_schedules()
```

**출력:**

```
Learning Rate Warmup Examples

Linear Warmup:
  Step    0: lr = 0.000001
  Step  250: lr = 0.000251
  Step  500: lr = 0.000501
  Step  750: lr = 0.000751
  Step 1000: lr = 0.001000
  Step 1500: lr = 0.001000

Exponential Warmup:
  Step    0: lr = 0.000000
  Step  250: lr = 0.000001
  Step  500: lr = 0.000010
  Step  750: lr = 0.000101
  Step 1000: lr = 0.001000
  Step 1500: lr = 0.001000

Cosine Warmup:
  Step    0: lr = 0.000000
  Step  250: lr = 0.000148
  Step  500: lr = 0.000502
  Step  750: lr = 0.000855
  Step 1000: lr = 0.001000
  Step 1500: lr = 0.001000

Plot saved to warmup_schedules.png
```

## 2. 논의

워밍업은 초반의 큰 갱신을 완화할 수 있지만 기울기 폭발을 반드시 막거나 모든 모델의 학습을 개선하는 방법은 아니다. 필요한 길이와 목표 학습률은 모델 초기화, 옵티마이저, 배치 크기와 전체 갱신 예산에 따라 달라진다.

이 구현은 0부터 세는 step에 대해 (step+1)/warmup_steps를 사용한다. 따라서 LinearWarmup의 첫 값은 0이 아니라 base_lr/warmup_steps이다. 지수 워밍업도 첫 값이 start_lr보다 크며, 양수인 시작값과 목표값 사이를 기하적으로 보간한다. 코사인 워밍업은 같은 진행률에 반코사인 함수를 적용한다.

WarmupWithDecay는 선형 워밍업 뒤 코사인 감쇠를 적용하고 total_steps 이후에는 min_lr를 유지한다. total_steps는 warmup_steps보다 커야 하며 단계는 음이 아닌 값으로 사용한다. get_lr는 숫자를 계산하는 함수이므로 실제 옵티마이저에 그 값을 전달하는 연결이 필요하다. 그림의 기본 범위는 전체 10000단계 중 앞 5000단계뿐이므로 마지막 감쇠값까지 보여 주는 그래프가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
base_lr=0.001, warmup_steps=1000인 LinearWarmup에서 step 0과 999의 값은 얼마인가?

</div>

??? success "연습문제 1 풀이"
    step 0에서는 0.000001이고 step 999에서는 0.001이다. 0부터 시작한다는 주석은 연속적인 보간의 개념 설명이며 실제 첫 호출의 반환값은 양수이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
지수 워밍업의 start_lr를 0으로 설정하면 되는가?

</div>

??? success "연습문제 2 풀이"
    안 된다. base_lr/start_lr를 계산하므로 0으로 나누게 된다. 기하 보간에는 양수인 시작값과 목표값이 필요하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
누적 배치 수를 바꾼 뒤 같은 워밍업 길이를 유지하려면 무엇을 기준으로 해야 하는가?

</div>

??? success "연습문제 3 풀이"
    의도한 길이가 옵티마이저 갱신 횟수인지 처리 표본 수인지 먼저 정한다. 갱신 기준이면 실제 optimizer.step()마다 단계를 증가시킨다. 같은 미니배치 수를 사용해도 누적 횟수가 바뀌면 갱신 수는 달라진다.

## 정리하며

워밍업 식의 step 인덱스와 목표값 도달 시점을 확인한다. 실제 학습과 연결할 때는 갱신 단위를 맞추고 워밍업과 감쇠 구간의 유효한 길이를 설정한다.
