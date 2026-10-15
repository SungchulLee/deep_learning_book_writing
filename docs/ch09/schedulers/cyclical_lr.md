# 순환 학습률

순환 학습률은 학습률을 주기적으로 높이고 낮추는 일정이다. 이 페이지는 삼각형 CLR, 1cycle과 코사인 재시작을 직접 계산하는 교육용 클래스를 비교한다.

## 1. 코드

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
    plt.savefig('cyclical_schedules.png', dpi=150)
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
    plot_cyclical_schedules()
```

**출력:**

```
Cyclical Learning Rate Examples

Triangular CLR:
  Step     0: lr = 1.000000e-04
  Step  1000: lr = 1.000000e-03
  Step  2000: lr = 1.000000e-04
  Step  5000: lr = 1.000000e-03
  Step  7500: lr = 5.500000e-04
  Step 10000: lr = 1.000000e-04

1cycle:
  Step     0: lr = 4.000000e-05
  Step  1000: lr = 3.600000e-04
  Step  2000: lr = 6.800000e-04
  Step  5000: lr = 8.117637e-04
  Step  7500: lr = 2.831298e-04
  Step 10000: lr = 1.000000e-07

SGDR:
  Step     0: lr = 1.000000e-03
  Step  1000: lr = 5.050000e-04
  Step  2000: lr = 1.000000e-03
  Step  5000: lr = 1.549821e-04
  Step  7500: lr = 9.165775e-04
  Step 10000: lr = 5.050000e-04

Plot saved to cyclical_schedules.png
```

## 2. 논의

삼각형 CLR에서 step_size는 반주기이며 한 주기는 2×step_size이다. triangular2는 주기마다 최대 진폭을 절반으로 줄이고, exp_range는 진폭에 $\gamma^{\mathrm{step}}$을 곱한다. 학습률을 높이는 구간은 탐색과 정규화 효과에 영향을 줄 수 있지만 안장점이나 국소 최솟값에서 벗어나는 것을 보장하지 않는다.

여기의 클래스들은 PyTorch 스케줄러와 이름이 같아도 별도의 구현이다. get_lr(step)은 숫자를 반환할 뿐 옵티마이저를 수정하지 않는다. OneCycleLR은 상승 구간에서 항상 선형 보간하며 하강 구간에서만 anneal_strategy를 적용한다. 마지막 학습률도 max_lr/final_div_factor로 정하므로 PyTorch OneCycleLR의 인수 의미와 동일하다고 가정하면 안 된다. 모멘텀의 역방향 스케줄링도 구현하지 않는다.

코사인 재시작은 주기가 끝나면 학습률을 다시 최댓값으로 올린다. 가중치나 옵티마이저 상태를 초기화한다는 뜻은 아니다. t_0는 양수, t_mult는 1 이상인 정수로 사용해야 반복문이 정상적으로 진행된다. 그 밖의 구현도 양수인 구간 길이와 유효한 단계 범위를 가정한다. ExponentialCyclicLR은 주기가 길어져 current_max_lr가 base_lr 아래로 내려가면 이름과 달리 상·하한이 뒤집힐 수 있다. 그래프는 일정의 모양만 보여 주며 실제 학습 속도의 비교가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
base_lr=0.001, max_lr=0.01, step_size=100인 triangular 정책에서 step 0, 100, 200의 값을 구하라.

</div>

??? success "연습문제 1 풀이"
    각각 0.001, 0.01, 0.001이다. step_size는 전체 주기가 아니라 최솟값에서 최댓값까지의 단계 수이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
직접 구현한 get_lr를 호출하기만 하면 옵티마이저 학습률이 바뀌는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 반환값을 실제 옵티마이저의 각 param_group 학습률에 연결해야 한다. 단계 단위가 에포크인지 갱신인지도 학습 루프에서 일관되게 정한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 페이지 OneCycleLR의 max_lr=0.01, div_factor=25, final_div_factor=10000이면 시작값과 마지막값은 얼마인가?

</div>

??? success "연습문제 3 풀이"
    initial_lr는 0.0004이고 final_lr는 0.000001이다. 이 코드의 final_lr는 max_lr를 기준으로 계산한다. 같은 이름의 라이브러리 인수도 이런 의미라고 추정하지 말고 해당 정의를 확인한다.

## 정리하며

일정의 구간과 끝값, 호출 단위를 명시해야 순환 학습률을 정확히 적용할 수 있다. 교육용 구현과 라이브러리 구현을 구분하고 성능 효과는 별도의 학습 실험으로 판단한다.
