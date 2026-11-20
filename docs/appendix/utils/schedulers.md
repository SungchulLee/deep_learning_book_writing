# 러닝레이트 스케줄러

자주 쓰는 러닝레이트 스케줄을 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 러닝레이트 스케줄러의 핵심만 남긴 교육용 예시다. 직접 구현한 것은 워밍업 + 코사인 감쇠 스케줄 하나이고, 계단식 감쇠와 ReduceLROnPlateau는 PyTorch 내장 구현을 쓰라는 메모로 두었다. 스텝 번호를 받아 러닝레이트를 돌려주는 순수한 함수라, 값을 바꾸어 가며 스케줄의 모양을 살펴보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
러닝레이트 스케줄러 - 자주 쓰는 패턴
포함한 것:
  - 워밍업 + 코사인 감쇠
  - 계단식 감쇠(PyTorch 내장 구현 메모)
  - ReduceLROnPlateau(PyTorch 내장 구현 메모)

파일: appendix/utils/schedulers.py
참고: 스텝을 받아 러닝레이트 배율을 돌려주는 교육용 스케줄러다.
"""

import math

# ========================================================================
# 메인
# ========================================================================


def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """
    워밍업 + 코사인 감쇠 스케줄.

    - 워밍업 동안 러닝레이트를 0에서 base_lr까지 선형으로 올린다
    - 그다음 base_lr에서 0까지 코사인 곡선으로 감쇠한다

    반환:
      현재 스텝의 러닝레이트
    """
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    # 코사인 감쇠 단계
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# 참고:
# 실제 학습에서는 흔히 PyTorch 스케줄러를 쓴다:
#   torch.optim.lr_scheduler.StepLR
#   torch.optim.lr_scheduler.CosineAnnealingLR
#   torch.optim.lr_scheduler.ReduceLROnPlateau


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 `math` 모듈만 쓰는 짧은 파이썬 함수로 워밍업 + 코사인 감쇠 스케줄을 보인다. PyTorch에 기대지 않으므로 스케줄의 모양을 따로 그려 보거나 다른 학습 코드에 옮겨 쓰기 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 유틸리티 모듈에 알맞은지 설명하여라.

</div>

??? success "풀이"
    이 코드에서는 이런 셋을 들 수 있다. (1) 상태 없는 함수로 만든다 -- 스텝 번호만 있으면 어느 시점의 러닝레이트든 다시 계산할 수 있으므로, 체크포인트에서 학습을 재개해도 스케줄이 어긋나지 않는다. 배율을 돌려주도록 `base_lr`로 나누어 감싸면 `torch.optim.lr_scheduler.LambdaLR`에 그대로 넣을 수 있다. (2) 분모를 `max(1, ...)`로 감싼다 -- `warmup_steps=0`이거나 `total_steps == warmup_steps`여도 0으로 나누지 않는다. (3) 선형 워밍업 뒤에 코사인 감쇠를 둔다 -- 학습 초기에는 웨이트가 무작위이고 Adam의 2차 모멘트 추정도 불안정해 큰 러닝레이트가 발산을 부르기 쉬우므로 0에서 천천히 올리고, 끝에서는 러닝레이트를 0까지 부드럽게 내려 최솟값 근처에서 미세 조정이 되게 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 견고하게 만든다. 다만 `warmup_cosine_lr`는 텐서가 아니라 수를 받으므로, 여기서 확인할 것은 형태가 아니라 값의 범위다 — `step >= 0`, `0 <= warmup_steps <= total_steps`, `base_lr > 0`.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 만한 실패 유형 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    이 함수에는 파라메터도 그래디언트도 없으므로, 실패는 스케줄 값에서 나온다. (1) **`step`이 `total_steps`를 넘을 때** -- 진행률 `progress`가 1을 넘어 코사인이 다시 올라가므로, 러닝레이트가 0에서 도로 커진다. 학습 내내 러닝레이트를 기록해 그려 보면 끝에서 다시 솟는 것으로 탐지된다. `progress`를 `min(1.0, ...)`로 자르거나 `total_steps`를 실제 스텝 수와 맞추어 고친다. (2) **첫 스텝의 러닝레이트가 0** -- `step=0`이면 워밍업 식이 0을 돌려주어 첫 갱신이 아무것도 바꾸지 않는다. 첫 몇 스텝의 러닝레이트를 출력해 탐지하고, `step + 1`로 계산해 고친다. 이 함수를 `LambdaLR`에 넣을 때 배율이 아니라 러닝레이트 자체를 돌려주면 `base_lr`가 두 번 곱해지는 것도 흔한 실수다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
러닝레이트 스케줄러 구현이 올바른지 종합적으로 검증하는 시험 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력까지 경계 사례를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 시험 함수를 작성한다.
    ```python
    def test_learning rate schedulers():
        model = Learning Rate Schedulers(...)
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
    위 코드는 모델 클래스를 시험하는 틀이라 이 쪽에 그대로 맞지는 않는다. `warmup_cosine_lr`는 수를 받아 수를 돌려주는 함수이므로, 형태나 그래디언트 대신 알려진 값과 견준다. `step=0`이면 0, `step=warmup_steps`이면 `base_lr`, 코사인 구간의 한가운데이면 `base_lr / 2`, `step=total_steps`이면 0이어야 하고, 워밍업 구간에서는 증가, 코사인 구간에서는 감소해야 한다. 경계 사례로는 `warmup_steps=0`과 `total_steps == warmup_steps`를 넣어 0으로 나누지 않는지 확인한다.

## 정리하며

**다룬 것** — 러닝레이트 스케줄러

이 구현은 `math` 모듈만 쓰는 짧은 파이썬 함수로 워밍업 + 코사인 감쇠 스케줄을 보인다.

앞의 연습문제 4개로 스스로 확인할 수 있다.
