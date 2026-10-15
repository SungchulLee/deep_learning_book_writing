# 랑주뱅 샘플링의 최소 예제

이 페이지는 알려진 스코어를 받아 여러 상태를 동시에 갱신하고 궤적을 기록하는 간단한 랑주뱅 샘플러를 살펴본다. 예제의 목표는 2차원 표준 정규분포이며 스코어는 −x다. 신경망이나 노이즈 스케줄을 학습하지 않으므로 입력·출력의 모양과 이산화 편향에 집중한다.

## 1. 코드와 기존 출력

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
    
    고침 규칙: x_{t+1} = x_t + ε/2 * ∇log p(x_t) + √ε * z_t
    
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
        score_fn: ∇log p(x)를 셈하는 함수
        x_init: 첫 자리, 꼴 (n_samples, dim)
        n_steps: 랑주뱅 걸음 수
        step_size: 걸음 크기 ε
    
    반환값:
        samples: 마지막 표본, 꼴 (n_samples, dim)
        trajectory: 중간 자리 모두, 꼴 (n_steps, n_samples, dim)
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
    plt.savefig('demo_langevin.png', dpi=150, bbox_inches='tight')
    print("Saved demo_langevin.png")
    
    print(f"\nFinal sample statistics:")
    print(f"Mean: {samples.mean(dim=0).numpy()}")
    print(f"Std: {samples.std(dim=0).numpy()}")
    print("\nExpected: Mean=[0, 0], Std=[1, 1]")
```

```
Langevin Dynamics Demo
================================================================================
Saved demo_langevin.png

Final sample statistics:
Mean: [0.01577806 0.02004953]
Std: [1.1007179  0.93960726]

Expected: Mean=[0, 0], Std=[1, 1]
```

## 2. 갱신식과 텐서 모양

step_size를 h라 하면 다음과 같이 갱신한다.

$$x_{n+1}=x_n+\frac h2s(x_n)+\sqrt h\,\xi_n,\qquad \xi_n\sim N(0,I)$$

x_init가 N×d이면 스코어와 잡음도 같은 모양이어야 한다. 함수는 최종 상태 N×d와 초기 상태를 포함한 (K+1)×N×d 궤적을 반환한다. main에서는 N=100, d=2, K=500이므로 각각 100×2와 501×100×2다.

각 상태의 모든 시점을 저장해 메모리는 O(KNd)다. 샘플러는 no_grad나 detach를 사용하지 않으므로 스코어가 학습 가능한 네트워크이면 반복 전체의 미분 그래프가 유지될 수 있다. 단순 추론 목적에서는 이 구현의 메모리 계약을 그대로 일반화해서는 안 된다.

## 3. 표준 정규분포에서의 이산화 편향

s(x)=−x이면 AR 계수는 1−h/2다. 0<h<4에서 정상분산은 다음과 같다.

$$v=\frac h{1-(1-h/2)^2}=\frac1{1-h/4}$$

h=0.1에서는 v≈1.02564, 정상 표준편차는 약 1.01274다. 따라서 출력의 'Expected Std=[1,1]'은 연속 목표에 대한 값이며 고정 보폭 반복의 정확한 정상 표준편차는 아니다. 유한 100개 상태의 표본 평균·표준편차에는 표본 변동과 초기 상태의 영향도 남는다. main은 난수 시드를 고정하지 않는다.

같은 시점에 있는 100개 상태는 독립 초기화·잡음과 점별 스코어라는 이 예제 조건에서 서로 독립적으로 진화한다. 하지만 한 상태의 연속 시점을 독립 샘플로 볼 수는 없다. 보폭을 줄이면 편향을 줄일 수 있어도 같은 반복 수에서 탐색하는 물리적 시간도 줄어든다.

## 4. 궤적 그림의 해석과 구현 범위

배경은 선택한 격자의 참 밀도이고, 빨간 점은 시작 위치, 초록 별은 마지막 위치다. 최대 열 개 상태의 경로만 그리므로 모든 상태나 분포 일치를 검증하는 그림은 아니다. xlim·ylim 밖의 이동은 보이지 않으며 matplotlib의 축 종횡비도 별도로 고정하지 않는다.

시각화는 3차원 궤적 텐서에서 첫 두 좌표만 사용하고 `.numpy()`를 직접 호출한다. CPU에 있고 기울기 추적에서 분리된 텐서를 전제로 한다. main의 스코어 래퍼도 NumPy 변환 후 CPU float32 텐서를 다시 만들기 때문에 CUDA·다른 dtype으로 일반화된 구현이 아니다.

이 함수는 수락·거절이 없는 ULA다. 고정 보폭의 정확한 목표 보존이나 모든 초기화·분포에서의 안정성을 보장하지 않는다. [앞 페이지](03_langevin_dynamics.md)의 연속시간 과정과 Metropolis 보정의 차이를 함께 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
100×2 상태를 500번 갱신하면 두 반환값과 궤적 그림의 표본 수는 무엇인가?

</div>

??? success "연습문제 1 풀이"

    최종 상태는 100×2, 궤적은 501×100×2다. 초기 상태도 한 시점으로 저장한다. 시각화는 min(100,10)=10개 상태의 경로만 표시하므로 전체 100개 분포의 통계와 구분한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
표준 정규 스코어와 h=0.1에서 정상분산을 계산하라.

</div>

??? success "연습문제 2 풀이"

    반복은 x′=0.95x+√0.1 ξ다. v=0.95²v+0.1에서 v=0.1/(1−0.95²)≈1.02564다. 정확한 목표 분산 1과 다르므로 평균·표준편차가 가까워 보인다는 것만으로 정확한 샘플링을 입증할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
스코어를 신경망으로 바꿀 때 모양 외에 무엇을 확인해야 하는가?

</div>

??? success "연습문제 3 풀이"

    상태와 모델의 device·dtype, 점별 또는 배치 의존 동작, eval 모드와 기울기 기록을 확인한다. 현재 루프는 그래프를 보존할 수 있고 시각화는 CPU NumPy를 가정한다. 학습한 스코어의 오차와 원래 목표 분포의 스코어인지를 검토해야 하며 이 예제에는 노이즈 스케줄 학습이 없다.



## 정리하며

최소 랑주뱅 예제는 스코어와 잡음을 같은 상태 모양으로 결합하고 모든 시점을 저장한다. 고정 보폭의 정상분포는 정확한 목표와 다를 수 있으며, CPU 변환·기울기 기록·저장 비용은 일반화 전에 확인해야 한다.
