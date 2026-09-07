# 랑주뱅 움직임 2

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 랑주뱅 움직임 2을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
FILE: 03_langevin_dynamics.py
어려움: 처음
걸리는 시간: 2시간
PREREQUISITES: 01_score_functions_basics.py, 02_score_matching_theory.py

학습 목표:
    1. 랑주뱅 마르코프 사슬 몬테카를로 뽑기를 이해한다
    2. 점수 바탕 뽑기 알고리즘을 짠다
    3. 뽑기 자취를 그려 본다
    4. 걸음 크기와 잡음의 몫을 이해한다

수학 바탕:
    랑주뱅 움직임은 기울기를 써서 분포에서 뽑는 마르코프 사슬 몬테카를로 알고리즘이다.
    
    Update rule: x_{t+1} = x_t + ε/2 * ∇log p(x_t) + √ε * z_t
    
    여기서 z_t ~ N(0, I)은 표준 정규 잡음이다.
    
    ε→0이고 t→∞이면 표본이 p(x)으로 모인다.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========================================================================
# 메인
# ========================================================================


def langevin_sampling(score_fn, x_init, n_steps=1000, step_size=0.01):
    """
    랑주뱅 움직임으로 분포에서 뽑는다.
    
    인수:
        score_fn: Function computing ∇log p(x)
        x_init: Initial position, shape (n_samples, dim)
        n_steps: 랑주뱅 걸음 수
        step_size: 걸음 크기 ε
    
    반환값:
        samples: Final samples, shape (n_samples, dim)
        trajectory: All intermediate positions, shape (n_steps, n_samples, dim)
    """
    x = x_init.clone()
    trajectory = [x.clone()]
    
    for step in range(n_steps):
        # 점수 셈하기
        score = score_fn(x)
        
        # 랑주뱅 고침
        noise = torch.randn_like(x)
        x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
        
        trajectory.append(x.clone())
    
    return x, torch.stack(trajectory)


def visualize_sampling_trajectories(trajectory, true_pdf=None, xlim=(-3,3), ylim=(-3,3)):
    """랑주뱅 뽑기 동안 표본이 어떻게 바뀌는지 그려 본다."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 참 밀도가 있으면 그린다
    if true_pdf is not None:
        x = np.linspace(xlim[0], xlim[1], 100)
        y = np.linspace(ylim[0], ylim[1], 100)
        X, Y = np.meshgrid(x, y)
        grid = np.stack([X.flatten(), Y.flatten()], axis=1)
        Z = true_pdf(grid).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
    
    # 자취를 그린다
    traj_np = trajectory.numpy()
    n_steps, n_samples, _ = traj_np.shape
    
    for i in range(min(n_samples, 10)):  # 앞 자취 10개를 그린다
        ax.plot(traj_np[:, i, 0], traj_np[:, i, 1], 
               alpha=0.3, linewidth=0.5)
        ax.scatter(traj_np[0, i, 0], traj_np[0, i, 1],
                  c='red', s=50, marker='o', label='Start' if i==0 else '')
        ax.scatter(traj_np[-1, i, 0], traj_np[-1, i, 1],
                  c='green', s=50, marker='*', label='End' if i==0 else '')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Langevin Sampling Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    print("Langevin Dynamics Demo")
    print("=" * 80)
    
    # 보여 주기: 2차원 정규 분포에서 뽑기
    def gaussian_score(x):
        return -x  # N(0, I)의 점수
    
    # 표본을 첫자리매김한다
    x_init = torch.randn(100, 2) * 3  # 넓은 분포에서 시작
    
    # 랑주뱅 뽑기를 돌린다
    samples, trajectory = langevin_sampling(
        lambda x: torch.tensor(gaussian_score(x.numpy()), dtype=torch.float32),
        x_init,
        n_steps=500,
        step_size=0.1
    )
    
    # 시각화한다
    visualize_sampling_trajectories(
        trajectory,
        true_pdf=lambda x: np.exp(-0.5 * np.sum(x**2, axis=1)) / (2*np.pi)
    )
    plt.savefig('/home/claude/demo_langevin.png', dpi=150, bbox_inches='tight')
    print("Saved demo_langevin.png")
    
    print(f"\nFinal sample statistics:")
    print(f"Mean: {samples.mean(dim=0).numpy()}")
    print(f"Std: {samples.std(dim=0).numpy()}")
    print("\nExpected: Mean=[0, 0], Std=[1, 1]")```

## 논의

랑주뱅 움직임 2의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.
