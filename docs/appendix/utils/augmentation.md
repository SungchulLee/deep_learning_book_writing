# 자료 불리기

자료 불리기 - 흔한 보기 불리기(텐서 바탕). 담긴 것:

여기 짜보기는 Data Augmentation을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
자료 불리기 - 흔한 보기 불리기(텐서 바탕)
담긴 것:
  - 아무렇게나 좌우 뒤집기
  - 아무렇게나 오려 내기(손쉽게 짠 것)
  - 빛깔 흔들기(단순한 밝기와 결 차이)
  - 섞어 올리기(가름)

두루마리: appendix/utils/augmentation.py
눈여겨볼 것: 배우기 위한 짜보기다(torchvision.transforms만큼 갖추지는 않았다).
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def random_horizontal_flip(x, p=0.5):
    """
    그림을 아무렇게나 좌우로 뒤집는다.

    x: (B, C, H, W)
    """
    if torch.rand(1).item() < p:
        return torch.flip(x, dims=[3])  # flip width dimension
    return x


def random_crop(x, crop_h, crop_w):
    """
    손쉽게 짠 아무렇게나 오려 내기.

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
    가름을 위한 섞어 올리기 불리기.

    x: (B, C, H, W)
    y: (B,) 갈래 번호 또는 (B, num_classes) 원핫

    돌려주는 것:
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
    pass```

## 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 잔손질 묶음의 고갱이 깨침을 보인다. 묶음으로 나눈 얼개 덕에 몫 하나하나를 살펴보고 다른 일이나 자료 묶음에 맞추어 바꾸기 쉽다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 잔손질 묶음에 알맞은지 밝혀라.

??? success "익힘 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 잣대 잡는 꾀 -- 묶음 잣대 잡기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 드러내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 보여 준다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
Data Augmentation 짜보기가 옳은지 두루 따지는 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 값이 끝으로 치우친 들임(0, 아주 큰 수)까지 금 언저리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_data augmentation():
        model = Data Augmentation(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 종요롭다.
