# PPO

PPO는 2017년 논문 "Proximal Policy Optimization Algorithms"에서 나왔다. 폴리시가 한 번에 크게 바뀌는 것을 막는 *클리핑된 대리 목적함수(clipped surrogate objective)*를 쓰는 폴리시 그래디언트 방법이며, 흔히 GAE 어드밴티지와 미니배치 에포크와 함께 쓴다.

여기 구현은 PPO를 짧고 공부하기 좋게 보인 예제다. 코드는 PPO 로스 계산 하나만 담고 있어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다. 네트워크와 롤아웃 수집, 학습 루프는 들어 있지 않다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
PPO - 근접 폴리시 최적화
논문: "근접 폴리시 최적화 알고리즘" (2017)
지은이: 존 슐먼 외
핵심 아이디어:
  - 폴리시가 한 번에 크게 바뀌는 것을 막는 *클리핑된 대리 목적함수(clipped surrogate objective)*를 쓰는 폴리시 그래디언트
  - 흔히 GAE 어드밴티지와 미니배치 에포크를 쓴다

클리핑된 목적함수:
  r_t = pi(a|s) / pi_old(a|s)
  L_clip = E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]

파일: appendix/rl/ppo.py
참고: PPO 로스를 교육용으로 구현한 것이다(이산 액션).
"""

import torch
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def ppo_loss(
    logits_new,        # (B, A) current policy logits
    logits_old,        # (B, A) behavior/old policy logits (frozen)
    actions,           # (B,)
    advantages,        # (B,)
    returns,           # (B,)
    values,            # (B,) critic values from current network
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    PPO 로스를 계산한다:
      - 클리핑된 폴리시 로스
      - 밸류 로스
      - 엔트로피 보너스
    """
    # 새 폴리시와 옛 폴리시에서의 로그 확률을 계산한다
    logp_new = F.log_softmax(logits_new, dim=1)
    logp_old = F.log_softmax(logits_old, dim=1)

    logp_new_a = logp_new.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    logp_old_a = logp_old.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    # 중요도 비율(importance ratio) r_t = exp(log pi_new - log pi_old)
    ratio = torch.exp(logp_new_a - logp_old_a)

    # 어드밴티지는 흔히 정규화한다. 그래디언트가 어드밴티지 쪽으로 역전파되지 않도록 떼어 낸다(detach)
    A = advantages.detach()

    # 클리핑된 대리 목적함수
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    policy_loss = -torch.min(unclipped, clipped).mean()

    # 밸류 함수 로스
    value_loss = F.mse_loss(values, returns)

    # 엔트로피 보너스(탐색을 장려한다)
    p_new = torch.softmax(logits_new, dim=1)
    entropy = -(p_new * logp_new).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 PPO 로스의 세 항 — 클리핑된 폴리시 로스, 밸류 로스, 엔트로피 보너스 — 을 보인다. 함수 하나로 묶여 있어 항 하나하나를 살펴보고 계수를 바꾸어 보기 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 결정을 짚어라. 구조적 선택 셋을 들고, 각각이 왜 강화학습에 알맞은지 설명하라.

</div>

??? success "풀이"
    이 코드에서는 이런 셋을 들 수 있다. (1) 확률 비율을 로그 확률의 차에 지수를 취해 계산한다 -- 확률을 직접 나누면 확률이 아주 작을 때 언더플로나 0 나누기가 생기지만, 로그 공간에서 빼면 안전하다. (2) 클리핑하지 않은 항과 클리핑한 항 가운데 작은 쪽을 쓴다 -- 비율이 $[1-\epsilon, 1+\epsilon]$ 밖으로 나가 목적을 키우는 쪽의 이득을 잘라 내는 비관적 하한이라, 한 번의 갱신에서 폴리시가 옛 폴리시에서 멀리 벗어나지 않는다. 데이터를 옛 폴리시로 모은 강화학습에서는 이것이 학습을 안정시키는 핵심이다. (3) 폴리시 로스, 밸류 로스, 엔트로피 보너스를 하나의 로스로 합친다 -- 액터와 크리틱이 네트워크를 공유할 때 역전파 한 번으로 둘 다 갱신할 수 있고, 엔트로피를 빼 주어 폴리시가 너무 일찍 결정론적으로 굳는 것을 막는다(탐색 유지). 어드밴티지를 `detach`하는 것도 폴리시 로스의 그래디언트가 크리틱으로 새지 않게 하는 선택이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검증하려면 핵심 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 강건하게 만든다. 이 쪽의 `ppo_loss`라면 `logits_new`와 `logits_old`가 같은 `(B, A)` 형태인지, `actions`·`advantages`·`returns`·`values`가 모두 길이 `B`의 1차원 텐서인지, `actions`의 값이 `0` 이상 `A` 미만인지를 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 수 있는 실패 양상(failure mode) 둘을 들고, 각각을 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실·폭발** -- 그래디언트 노름을 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다. 다만 강화학습에는 고정된 검증 집합이 없으므로, PPO에서는 이보다 다음 둘이 더 흔하다. (3) **폴리시가 너무 크게 움직임** -- 옛 폴리시와 새 폴리시의 근사 KL이나, 비율이 클리핑 구간 밖에 있는 샘플의 비율(clip fraction)이 치솟는 것으로 탐지한다. 러닝레이트나 에포크 수를 줄이거나, KL이 기준을 넘으면 그 에포크를 일찍 멈춘다. (4) **엔트로피 붕괴** -- 엔트로피가 학습 초반에 0 가까이 떨어지고 리턴이 정체하면 탐지한다. `entropy_coef`를 키우거나 어드밴티지를 정규화해 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
PPO 구현이 옳은지 종합적으로 검증하는 시험 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_ppo():
        model = PPO(...)
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
    위 코드는 모델 클래스를 시험하는 틀이다. 이 쪽에는 `PPO` 클래스가 없고 `ppo_loss` 함수만 있으므로, `model(...)` 자리에서 `ppo_loss`를 부르고 출력 형태 대신 로스가 유한한 스칼라인지, `logits_new`에 그래디언트가 흐르는지를 확인한다. 특히 `logits_new`와 `logits_old`가 같으면 비율이 정확히 1이 되어 클리핑이 걸리지 않으므로, 이때 폴리시 로스가 `-advantages.mean()`과 같은지 보면 좋은 기준점이 된다.

## 정리하며

**다룬 것** — PPO

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 PPO 로스의 세 항을 보인다.

앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
