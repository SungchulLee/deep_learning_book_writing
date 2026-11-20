# 데이터 어그멘테이션

자주 쓰는 비전 어그멘테이션을 텐서 연산으로 구현해 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 데이터 어그멘테이션의 핵심만 남긴 교육용 예시다. 변환마다 함수 하나로 짧게 적어, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다. 독스트링은 컬러 지터도 들지만 코드에 구현된 것은 좌우 뒤집기, 무작위 크롭, 믹스업 셋이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
데이터 어그멘테이션 - 자주 쓰는 비전 어그멘테이션(텐서 기반)
포함한 것:
  - 무작위 좌우 뒤집기
  - 무작위 크롭(단순 구현)
  - 컬러 지터(단순한 밝기와 대비)
  - 믹스업(분류)

파일: appendix/utils/augmentation.py
참고: 교육용 구현이다(torchvision.transforms만큼 갖추지는 않았다).
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def random_horizontal_flip(x, p=0.5):
    """
    이미지를 무작위로 좌우로 뒤집는다.

    x: (B, C, H, W)
    """
    if torch.rand(1).item() < p:
        return torch.flip(x, dims=[3])  # flip width dimension
    return x


def random_crop(x, crop_h, crop_w):
    """
    단순하게 구현한 무작위 크롭.

    x: (B, C, H, W)
    """
    B, C, H, W = x.shape
    if crop_h > H or crop_w > W:
        raise ValueError("Crop size must be <= image size")

    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()
    return x[:, :, top:top + crop_h, left:left + crop_w]


def mixup(x, y, alpha=0.2):
    """
    분류를 위한 믹스업 어그멘테이션.

    x: (B, C, H, W)
    y: (B,) 클래스 번호 또는 (B, num_classes) 원핫

    반환:
      x_mix, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B = x.size(0)
    perm = torch.randperm(B)

    x_mix = lam * x + (1 - lam) * x[perm]
    y_a = y
    y_b = y[perm]
    return x_mix, y_a, y_b, lam


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
    이 코드에서는 이런 셋을 들 수 있다. (1) 모든 변환을 `(B, C, H, W)` 배치 텐서에 torch 연산만으로 적용한다 -- 이미지를 하나씩 PIL로 다루지 않으므로 GPU에 올라간 배치에 바로 쓸 수 있다. (2) 난수를 배치 전체에 하나만 뽑는다 -- `torch.rand(1)`과 `torch.randint(..., (1,))`이 하나의 값이라, 배치 안의 이미지가 모두 함께 뒤집히고 같은 자리에서 잘린다. 코드는 짧아지지만 샘플마다 다른 변환을 받는 표준 방식보다 다양성이 떨어진다. (3) `mixup`은 섞은 레이블 대신 `y_a`, `y_b`, `lam`을 돌려준다 -- 로스를 $\lambda L(\hat y, y_a) + (1-\lambda) L(\hat y, y_b)$로 계산하면 되므로, `y`가 클래스 번호든 원핫이든 같은 코드로 다룰 수 있다.

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
    이 함수들에는 파라메터도 그래디언트도 없으므로, 실패는 변환 자체에서 나온다. (1) **배치 전체가 같은 변환을 받음** -- 난수를 배치에 하나만 뽑으므로 배치마다 다양성이 줄어든다. 변환한 배치를 격자로 그려 보면 모든 이미지가 함께 뒤집히거나 같은 자리에서 잘린 것으로 탐지된다. 샘플마다 난수를 뽑아 마스크로 고르거나(`torch.rand(B) < p`), 샘플마다 크롭 위치를 따로 정해 고친다. (2) **레이블과 어긋남** -- 뒤집기·크롭은 이미지만 바꾸므로, 바운딩 박스나 분할 마스크가 딸린 과제에 그대로 쓰면 레이블이 이미지와 어긋난다. 믹스업도 로스를 `lam`으로 섞지 않고 `y_a`만 쓰면 레이블이 틀린다. 변환 뒤 이미지에 레이블을 겹쳐 그려 탐지하고, 레이블에도 같은 변환을 적용하거나 로스를 `lam`으로 섞어 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
데이터 어그멘테이션 구현이 올바른지 종합적으로 검증하는 시험 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력까지 경계 사례를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 시험 함수를 작성한다.
    ```python
    def test_data augmentation():
        model = Data Augmentation(...)
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
    위 코드는 모델 클래스를 시험하는 틀이라 이 쪽에 그대로 맞지는 않는다. 어그멘테이션 함수에는 파라메터가 없으므로 그래디언트 흐름 대신 성질을 시험한다. `p=1`로 두 번 뒤집으면 원래 텐서로 돌아와야 하고, `random_crop`의 출력은 언제나 `(B, C, crop_h, crop_w)`여야 하며 크롭 크기가 이미지보다 크면 `ValueError`가 나야 한다. `mixup`은 `alpha=0`이면 입력을 그대로 돌려주고 `lam`이 1이어야 하며, `lam`은 언제나 0과 1 사이여야 한다. 원소 하나짜리 배치에서는 `torch.randperm(1)`이 자기 자신을 고르므로 믹스업이 아무것도 섞지 않는다는 점도 확인해 둘 만하다.

## 정리하며

**다룬 것** — 데이터 어그멘테이션

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 유틸리티 모듈의 핵심 아이디어를 보인다.

앞의 연습문제 4개로 스스로 확인할 수 있다.
