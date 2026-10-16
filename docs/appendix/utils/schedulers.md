# 러닝레이트 스케줄러

담긴 것:

여기 구현은 Learning Rate Schedulers을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 걸음에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
러닝 빠르기 짜개 - 흔한 무늬
담긴 것:
  - 몸 풀기 + 코사인 잦아들기
  - 계단 잦아들기(PyTorch 붙박이 적바림)
  - ReduceLROnPlateau(PyTorch 붙박이 적바림)

두루마리: appendix/utils/schedulers.py
눈여겨볼 것: 걸음을 받아 러닝 빠르기 곱값을 돌려주는 배우기용 짜개다.
"""

import math

# ========================================================================
# 메인
# ========================================================================


def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """
    몸 풀기 + 코사인 잦아들기 짜임.

    - 몸 푸는 동안 러닝 빠르기를 0에서 base_lr까지 곧게 올린다
    - 그다음 base_lr에서 0까지 코사인으로 잦아든다

    돌려주는 것:
      이제 걸음의 러닝 빠르기
    """
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    # 코사인 잦아들기 판
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# 눈여겨볼 것:
# 참으로 굴릴 때는 흔히 PyTorch 짜개를 쓴다:
#   torch.optim.lr_scheduler.StepLR
#   torch.optim.lr_scheduler.CosineAnnealingLR
#   torch.optim.lr_scheduler.ReduceLROnPlateau


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 잔손질 묶음의 핵심 깨침을 보인다. 묶음으로 나눈 구조 덕에 몫 하나하나를 살펴보고 다른 일이나 데이터셋에 맞추어 바꾸기 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 잔손질 묶음에 알맞은지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 잣대 잡는 꾀 -- 배치 잣대 잡기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 드러내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 증강)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
Learning Rate Schedulers 구현이 옳은지 두루 따지는 시험 함수를 써라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 값이 끝으로 치우친 입력(0, 아주 큰 수)까지 금 언저리를 시험하여라.

</div>

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_learning rate schedulers():
        model = Learning Rate Schedulers(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 끝에서 끝까지 학습을 받치는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 러닝레이트 짜개

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 잔손질 묶음의 핵심 깨침을 보인다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
