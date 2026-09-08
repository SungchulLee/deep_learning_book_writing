# 학습률 워밍업

학습률 워밍업은 학습의 처음 몇 단계 동안 학습률을 아주 작은 값에서 목표 값까지 서서히 올린다. 이는 모델의 가중치가 아직 무작위인 학습 초반에 크고 불안정한 기울기 갱신이 일어나는 것을 막는다. 워밍업은 트랜스포머 구조와 큰 배치 학습에서 매우 중요하다.

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
    plt.savefig('/home/claude/warmup_schedules.png', dpi=150)
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
    plot_warmup_schedules()```

## 2. 논의

선형 워밍업은 정해진 단계 수 동안 학습률을 0에서 목표 값까지 올린다. 가장 간단하고 널리 쓰이는 워밍업 전략으로 BERT, GPT를 비롯한 대부분의 트랜스포머 학습 요령이 채택하고 있다. 서서히 올리면 가중치가 아직 무작위일 때 큰 기울기 갱신이 일어나지 않는다.

지수 워밍업은 아주 작은 시작 값과 목표 값 사이를 기하적으로 보간하여, 처음에는 천천히 오르다가 갈수록 빨라진다. 코사인 워밍업은 반코사인 곡선을 따라 매끄럽게 옮겨 가며, 워밍업 구간의 한가운데에서 가장 빠르게 오른다.

`WarmupWithDecay` 클래스는 선형 워밍업과 코사인 어닐링을 결합하여 요즘의 여러 구조가 쓰는 전체 일정을 구현한다. 워밍업이 끝나면 학습률은 최고점에서 최솟값까지 코사인 곡선을 따라 내려가며, 이러한 매끄러운 감쇠는 여러 상황에서 계단식 일정보다 나은 것으로 알려져 있다.

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

**다룬 것** — 학습률 워밍업

선형 워밍업은 정해진 단계 수 동안 학습률을 0에서 목표 값까지 올린다.

핵심 클래스는 `LinearWarmup`, `ExponentialWarmup`, `CosineWarmup`, `WarmupWithDecay`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
