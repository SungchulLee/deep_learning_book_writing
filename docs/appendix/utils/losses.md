# 로스 함수

자주 쓰는 딥러닝 로스 함수를 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 로스 함수의 핵심만 남긴 교육용 예시다. 코드는 계산식만 함수로 옮겨 두었으므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다. 독스트링은 다섯 가지를 들지만 직접 구현한 것은 초점 로스(focal loss)와 KL 발산 둘이다. 교차 엔트로피·MSE·BCEWithLogits는 PyTorch의 `F.cross_entropy`·`F.mse_loss`·`F.binary_cross_entropy_with_logits`를 그대로 쓰면 된다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
로스 함수 - 자주 쓰는 딥러닝 로스
포함한 것:
  - 교차 엔트로피(분류)
  - MSE(회귀)
  - BCEWithLogits(이진 분류)
  - 초점 로스(밀집 탐지)
  - KL 발산 도우미(VAE 형태)

파일: appendix/utils/losses.py
참고: 이해하기 쉽게 쓴 교육용 구현이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    이진 분류를 위한 초점 로스(흔히 다중 클래스로 확장한다).

    logits:  (B,) 또는 (B,1) 정규화 전 점수
    targets: (B,), {0,1} 안의 값

    FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
    여기서 p_t는 참 클래스에 모델이 준 확률이다.
    """
    targets = targets.float()

    # 시그모이드로 확률을 계산한다
    p = torch.sigmoid(logits)

    # y=1이면 p_t = p, 아니면 (1-p)
    p_t = p * targets + (1 - p) * (1 - targets)

    # y=1이면 alpha_t = alpha, 아니면 (1-alpha)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # 표준 BCE 로스 항: -log(p_t)
    ce = -torch.log(p_t.clamp(min=1e-8))

    loss = alpha_t * ((1 - p_t) ** gamma) * ce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def kl_normal(mu, logvar):
    """
    샘플마다 N(mu, sigma^2)과 N(0,1) 사이의 KL 발산.

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


if __name__ == "__main__":
    pass
```

## 2. 논의

로스 계산은 모델의 출력을 최적화 목표와 잇는다. 알맞은 로스 함수를 고르는 일은 매우 중요하다. 모델이 무엇을 최적화하도록 학습할지를 정하고, 학습된 표현과 결정 경계를 직접 결정하기 때문이다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 유틸리티 모듈에 알맞은지 설명하여라.

</div>

??? success "풀이"
    이 코드에서는 이런 셋을 들 수 있다. (1) `focal_loss`는 $p_t$와 $\alpha_t$를 `if` 분기 없이 `targets`를 곱해 만든다 -- 레이블이 0인 샘플과 1인 샘플을 한 번의 텐서 연산으로 처리하므로 배치 전체에 그대로 적용된다. (2) 로그를 취하기 전에 $p_t$를 `1e-8` 아래로 내려가지 않게 자른다 -- $p_t = 0$일 때 $\log 0 = -\infty$가 되는 것을 막는다. 다만 시그모이드를 먼저 계산한 뒤 로그를 취하므로, 로짓에서 바로 계산하는 `F.binary_cross_entropy_with_logits`보다는 수치적으로 덜 안정하다. (3) `kl_normal`은 분산 대신 로그 분산 `logvar`를 받는다 -- 네트워크가 내놓는 값은 음수일 수도 있는데 `exp`로 되돌리면 분산이 언제나 양수가 되고, 식도 닫힌 형태로 깔끔해진다. `reduction` 인자를 두어 PyTorch 로스 함수의 관례(`"mean"`, `"sum"`, 그대로)를 따른 것도 다른 코드에 끼워 쓰기 쉽게 하는 선택이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 견고하게 만든다. 이 쪽의 함수라면 `focal_loss`에서는 `logits`와 `targets`의 원소 수가 같은지와 `targets`의 값이 0 또는 1뿐인지를, `kl_normal`에서는 `mu`와 `logvar`가 같은 `(B, D)` 형태인지를 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 만한 실패 유형 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다. 이 쪽의 함수에 더 직접 걸리는 것은 수치 문제다. `kl_normal`에서 `logvar`가 크면 `exp`가 넘쳐 로스가 `inf`가 되고, `focal_loss`에서 로짓의 절댓값이 크면 시그모이드가 0이나 1에 붙어 그래디언트가 사라진다. 로스가 `nan`이나 `inf`가 되는지 매 스텝 `torch.isfinite`로 확인해 탐지하고, `logvar`를 적당한 범위로 자르거나 로짓에서 바로 계산하는 형태(`F.logsigmoid` 등)로 바꾸어 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
로스 함수 구현이 올바른지 종합적으로 검증하는 시험 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력까지 경계 사례를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 시험 함수를 작성한다.
    ```python
    def test_loss functions():
        model = Loss Functions(...)
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
    위 코드는 모델 클래스를 시험하는 틀이다. 이 쪽에는 클래스가 없고 함수 `focal_loss`와 `kl_normal`만 있으므로, `model(...)` 자리에서 두 함수를 부르고 로스가 유한한지, 입력 로짓에 그래디언트가 흐르는지를 확인한다. 알려진 값과 견주는 시험도 좋다. `gamma=0`이고 `alpha=0.5`이면 초점 로스는 BCE의 절반이 되어야 하고, `mu=0`·`logvar=0`이면 `kl_normal`은 0이 되어야 한다.

## 정리하며

**다룬 것** — 로스 함수

로스 계산은 모델의 출력을 최적화 목표와 잇는다.

앞의 연습문제 4개로 스스로 확인할 수 있다.
