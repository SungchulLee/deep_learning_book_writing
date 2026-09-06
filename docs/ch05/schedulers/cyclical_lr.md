# 순환 학습률

순환 학습률은 학습 중에 학습률을 최솟값과 최댓값 사이에서 주기적으로 오르내리게 한다. Leslie Smith의 원래 CLR 논문은 이 방식이 단조 감소 일정보다 더 빨리 수렴할 수 있음을 보였다. 변형으로는 삼각형, triangular2, exp_range 정책, 1cycle 정책, 웜 리스타트를 쓰는 코사인 어닐링이 있다.

## 코드

```python
"""
순환 학습률 구현

이 모듈은 신경망 학습을 위한 여러 순환 학습률 전략을 제공하며, 원래 CLR
논문의 구현과 1cycle 정책을 포함한다.

"""

import math
from typing import Optional, Literal

# ========================================================================
# 메인
# ========================================================================


class CyclicLR:
    """
    Leslie Smith가 제안한 순환 학습률(CLR).
    
    참고: "Cyclical Learning Rates for Training Neural Networks"
    https://arxiv.org/abs/1506.01186
    """
    
    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: Literal['triangular', 'triangular2', 'exp_range'] = 'triangular',
        gamma: float = 0.99994
    ):
        """
        인수:
            base_lr: 학습률의 아래 경계
            max_lr: 학습률의 위 경계
            step_size: 반주기의 학습 단계 수
            mode: {triangular, triangular2, exp_range} 가운데 하나
            gamma: exp_range 모드의 감쇠 상수
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** step
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        return lr


class OneCycleLR:
    """
    Leslie Smith의 1cycle 학습률 정책.
    
    다음으로 이루어진다:
    1. 워밍업: 학습률이 initial_lr에서 max_lr까지 오른다
    2. 어닐링: 학습률이 max_lr에서 final_lr까지 내려간다
    
    참고: "Super-Convergence: Very Fast Training of Neural Networks"
    https://arxiv.org/abs/1708.07120
    """
    
    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: Literal['cos', 'linear'] = 'cos'
    ):
        """
        인수:
            max_lr: 최고 학습률
            total_steps: 전체 학습 단계 수
            pct_start: 주기 가운데 학습률을 올리는 데 쓰는 비율
            div_factor: 처음 학습률 = max_lr / div_factor
            final_div_factor: 마지막 학습률 = max_lr / final_div_factor
            anneal_strategy: 'cos' 또는 'linear' 어닐링
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.anneal_strategy = anneal_strategy
        
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        if step < self.step_up:
            # 올라가는 국면
            progress = step / self.step_up
            return self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # 내려가는 국면
            progress = (step - self.step_up) / self.step_down
            
            if self.anneal_strategy == 'cos':
                # 코사인 어닐링
                cos_out = math.cos(math.pi * progress)
                return self.final_lr + (self.max_lr - self.final_lr) * (1 + cos_out) / 2
            else:
                # 선형 어닐링
                return self.max_lr - (self.max_lr - self.final_lr) * progress


class CosineAnnealingWarmRestarts:
    """
    웜 리스타트를 쓰는 코사인 어닐링 (SGDR).
    
    참고: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    https://arxiv.org/abs/1608.03983
    """
    
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        t_0: int,
        t_mult: int = 2
    ):
        """
        인수:
            max_lr: 최대 학습률
            min_lr: 최소 학습률
            t_0: 첫 재시작까지의 단계 수
            t_mult: 재시작마다 주기의 길이를 늘리는 인수
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.t_0 = t_0
        self.t_mult = t_mult
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        # 지금 몇 번째 주기인지 찾기
        t_cur = step
        t_i = self.t_0
        cycle = 0
        
        while t_cur >= t_i:
            t_cur -= t_i
            t_i *= self.t_mult
            cycle += 1
        
        # 현재 주기 안에서의 코사인 어닐링
        progress = t_cur / t_i
        lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        return lr


class ExponentialCyclicLR:
    """
    지수 순환 학습률.
    학습률이 지수 곡선을 그리며 base_lr과 max_lr 사이를 오간다.
    """
    
    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        cycle_length: int,
        decay_rate: float = 0.96
    ):
        """
        인수:
            base_lr: 최소 학습률
            max_lr: 최대 학습률
            cycle_length: 한 주기의 단계 수
            decay_rate: 주기마다 max_lr을 줄이는 인수
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.decay_rate = decay_rate
    
    def get_lr(self, step: int) -> float:
        """주어진 단계의 학습률을 얻는다."""
        cycle = step // self.cycle_length
        step_in_cycle = step % self.cycle_length
        
        # 시간이 갈수록 max_lr 줄이기
        current_max_lr = self.max_lr * (self.decay_rate ** cycle)
        
        # 주기 안에서의 지수 보간
        progress = step_in_cycle / self.cycle_length
        
        if progress < 0.5:
            # 올라가는 국면 (지수)
            phase_progress = progress * 2
            lr = self.base_lr + (current_max_lr - self.base_lr) * (math.exp(phase_progress) - 1) / (math.e - 1)
        else:
            # 내려가는 국면 (지수)
            phase_progress = (progress - 0.5) * 2
            lr = current_max_lr - (current_max_lr - self.base_lr) * (math.exp(phase_progress) - 1) / (math.e - 1)
        
        return lr


def plot_cyclical_schedules(total_steps: int = 10000):
    """
    여러 순환 학습률 일정을 그린다.
    matplotlib이 필요하다.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    schedulers = {
        'Triangular CLR': CyclicLR(1e-4, 1e-3, step_size=1000, mode='triangular'),
        'Triangular2 CLR': CyclicLR(1e-4, 1e-3, step_size=1000, mode='triangular2'),
        '1cycle': OneCycleLR(1e-3, total_steps, pct_start=0.3),
        'Cosine Warm Restarts': CosineAnnealingWarmRestarts(1e-3, 1e-5, t_0=2000),
    }
    
    steps = range(total_steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, scheduler) in enumerate(schedulers.items()):
        lrs = [scheduler.get_lr(step) for step in steps]
        axes[idx].plot(steps, lrs, linewidth=2, color=f'C{idx}')
        axes[idx].set_xlabel('Training Step')
        axes[idx].set_ylabel('Learning Rate')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/cyclical_schedules.png', dpi=150)
    print("Plot saved to cyclical_schedules.png")


if __name__ == "__main__":
    print("Cyclical Learning Rate Examples\n")
    
    # 여러 전략 시험
    total_steps = 10000
    test_steps = [0, 1000, 2000, 5000, 7500, 10000]
    
    schedulers = {
        'Triangular CLR': CyclicLR(1e-4, 1e-3, step_size=1000),
        '1cycle': OneCycleLR(1e-3, total_steps),
        'SGDR': CosineAnnealingWarmRestarts(1e-3, 1e-5, t_0=2000),
    }
    
    for name, scheduler in schedulers.items():
        print(f"{name}:")
        for step in test_steps:
            if step <= total_steps:
                lr = scheduler.get_lr(step)
                print(f"  Step {step:5d}: lr = {lr:.6e}")
        print()
    
    # 시각화 만들기
    plot_cyclical_schedules()```

## 논의

삼각형 정책은 학습 단계 `2 * step_size` 동안 학습률을 `base_lr`과 `max_lr` 사이에서 선형으로 순환시킨다. triangular2는 주기마다 진폭을 절반으로 줄이고, exp_range는 진폭에 $\gamma^{\text{step}}$을 곱해 지수적으로 줄인다. 세 정책 모두 최적화기가 안장점과 국소 극소점을 벗어나도록 돕는다.

Leslie Smith의 1cycle 정책은 두 국면으로 이루어진다. `initial_lr`에서 `max_lr`까지 올리는 워밍업(기본값으로 학습의 처음 30%)과, `max_lr`에서 아주 작은 `final_lr`까지 내리는 어닐링이다. 이 과감한 일정 덕분에 고정 학습률보다 10배 높은 학습률로 학습할 수 있어 더 빨리 수렴하는 일이 많다.

웜 리스타트를 쓰는 코사인 어닐링(SGDR)은 학습률을 주기적으로 최댓값으로 되돌리며, 뒤따르는 주기마다 길이가 `t_mult`배씩 길어진다. 이러한 재시작은 최적화기가 좁은 국소 극소점을 벗어나 손실 지형의 다른 영역을 탐색하도록 돕는다.

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

