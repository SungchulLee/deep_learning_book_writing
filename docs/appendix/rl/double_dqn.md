# Double DQN

Double DQN은 2016년 논문 "Deep Reinforcement Learning with Double Q-learning"에서 나왔다. - 온라인 네트워크로 액션을 *고른다* - 타깃 네트워크로 그 액션을 *평가한다*.

여기 구현은 Double DQN을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
Double DQN - Q 러닝의 과대추정 편향 줄이기
논문: "이중 Q 러닝을 쓰는 심층 강화학습" (2016)
지은이: 하도 판 하셀트, 아르투어 게즈, 데이비드 실버
핵심 아이디어:
  - 온라인 네트워크로 액션을 *고른다*
  - 타깃 네트워크로 그 액션을 *평가한다*

타깃:
  a* = argmax_a Q_online(s', a)
  y  = r + gamma * (1-done) * Q_target(s', a*)

파일: appendix/rl/double_dqn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def double_dqn_td_loss(q_online: nn.Module, q_target: nn.Module, batch, gamma: float = 0.99):
    """
    Double DQN의 TD 로스를 계산한다.

    DQN과의 차이:
      - DQN은 max_a Q_target(s', a)을 쓴다
      - Double DQN은 argmax은 온라인 네트워크에서, 값은 타깃 네트워크에서 얻는다
    """
    s, a, r, s2, done = batch

    # 현재의 Q(s,a)
    q_values = q_online(s)  # (B, A)
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # 온라인 네트워크가 액션을 고른다
        a_star = q_online(s2).argmax(dim=1)  # (B,)

        # 타깃 네트워크가 액션을 평가한다
        q_next_target = q_target(s2)  # (B, A)
        q_s2_astar = q_next_target.gather(1, a_star.unsqueeze(1)).squeeze(1)

        target = r + gamma * (1.0 - done) * q_s2_astar

    loss = F.mse_loss(q_sa, target)
    return loss


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 강화학습의 핵심 아이디어를 보인다. 모듈식 구조 덕분에 구성 요소 하나하나를 살펴보고 다른 과제나 데이터셋에 맞추어 바꾸기 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 결정을 짚어라. 구조적 선택 셋을 들고, 각각이 왜 강화학습에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 결정은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 포화하지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동(internal covariate shift)을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 선택마다 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검증하려면 핵심 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 수 있는 실패 양상(failure mode) 둘을 들고, 각각을 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실·폭발** -- 그래디언트 노름을 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
Double DQN 구현이 옳은지 종합적으로 검증하는 시험 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_double_dqn():
        model = DoubleDQN(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 확인하려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — Double DQN

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 강화학습의 핵심 아이디어를 보인다.

앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
