# 러닝레이트 스케줄러

자주 쓰는 러닝레이트 스케줄을 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 Learning Rate Schedulers의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

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

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 유틸리티 모듈의 핵심 아이디어를 보인다. 모듈로 나눈 구조 덕에 구성 요소 하나하나를 살펴보고 다른 과제나 데이터셋에 맞추어 바꾸기 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 유틸리티 모듈에 알맞은지 설명하여라.

</div>

??? success "풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 양수 구간에서 그래디언트가 포화하지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동(internal covariate shift, 학습 중 레이어 입력의 분포가 바뀌는 현상)을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 선택마다 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 견고하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 만한 실패 유형 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
Learning Rate Schedulers 구현이 올바른지 종합적으로 검증하는 시험 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력까지 경계 사례를 시험하여라.

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
    구조가 입력에서 출력까지 한 번에 이어지는 학습(엔드투엔드 학습)을 지원하는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 러닝레이트 스케줄러

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 유틸리티 모듈의 핵심 아이디어를 보인다.

앞의 연습문제 4개로 스스로 확인할 수 있다.
