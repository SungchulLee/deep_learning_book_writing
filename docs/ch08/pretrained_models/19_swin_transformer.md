# 스윈 트랜스포머

2021년 논문 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"에서 나온 스윈 트랜스포머는 비전 과제에서 전역 자기 주의가 비효율적인 문제를 푼다. 국소 창 안에서 주의를 셈하고 층마다 그 창을 어긋나게 하여, 합성곱 신경망처럼 위계를 이루는 특징 지도를 쌓으면서 그림 크기에 대해 일차 계산 복잡도를 이룬다.

## 1. 코드

```python
import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.head = nn.Linear(embed_dim * 8, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = nn.functional.adaptive_avg_pool1d(
            x.transpose(1, 2), 1
        ).squeeze(-1)
        return self.head(x)


if __name__ == "__main__":
    model = SwinTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 2. 논의

`WindowAttention` 모듈은 자기 주의를 크기가 고정된 (대개 $7 \times 7$ 조각의) 국소 창으로 제한한다. 그러면 전역 주의의 이차 복잡도가 $O(N^2)$에서 $O(N \cdot W^2)$으로 줄며 여기서 $W$은 창 크기이다. QKV 사영과 다중 머리 나누기는 표준 트랜스포머 주의와 같은 방식이지만 창마다 그 안에서만 이루어진다.

(간추린 이 구현에 온전히 담기지는 않은) 어긋난 창 얼개는 잇따른 층에서 보통 창 나누기와 어긋난 창 나누기를 번갈아 한다. 홀수 층에서는 창을 창 크기의 절반만큼 어긋나게 하여 창끼리의 연결을 만들고 이웃한 창 사이로 정보가 흐르게 한다. 그러면 전역 주의의 비용 없이 여러 층에 걸쳐 전역 수용 영역을 이룬다.

스윈 트랜스포머는 단계마다 조각을 합쳐 위계를 이루는 특징 지도를 쌓으며, 합성곱 신경망의 풀링처럼 통로 차원을 두 배로 하고 공간 해상도를 반으로 줄인다. 그래서 스윈 트랜스포머는 분류뿐 아니라 여러 크기의 특징이 꼭 필요한 물체 탐지나 의미 분할 같은 빽빽한 예측 과제에도 알맞은 두루 쓰이는 등뼈가 된다.

## 연습문제

**연습문제 1.**
조각 크기가 4이고 창 크기가 7인 $224 \times 224$ 그림에서 창 주의와 전역 주의의 계산 복잡도를 셈하라. 답을 주의 점수를 셈하는 횟수로 나타내어라.

??? success "연습문제 1 풀이"
    조각 크기가 4이면 조각의 수는 $(224/4)^2 = 56^2 = 3136$이다. 전역 주의는 머리마다 주의 점수를 $3136^2 \approx 9.8 \times 10^6$번 셈한다. 창 크기가 7이면 창이 $(56/7)^2 = 64$개이고 창마다 토큰이 $7^2 = 49$개이다. 창마다 주의 점수를 $49^2 = 2401$번 셈하므로 머리마다 모두 $64 \times 2401 = 153{,}664$번으로, 전역 주의보다 대략 $64$배 적다.

---

**연습문제 2.**
어긋난 창 얼개의 목적을 설명하라. 스윈 트랜스포머가 전역 수용 영역을 이루는 데 그것이 왜 필요하며, 같은 목적으로 합성곱 신경망이 쓰는 팽창 합성곱과는 어떻게 다른가?

??? success "연습문제 2 풀이"
    창을 어긋나게 하지 않으면 토큰마다 제 국소 창 안에서만 주의할 수 있고 창끼리 주고받는 것이 없다. 어긋난 창 얼개는 층을 번갈아 가며 창 나누기를 창 크기의 절반만큼 옮겨, 창 가장자리의 토큰이 이웃 창의 토큰에 주의하게 한다. 여러 층에 걸쳐 이것이 그림 전체에 걸친 간접 연결을 만든다. 정해진 간격으로 뽑아 수용 영역을 넓히는 팽창 합성곱과 달리, 어긋난 창은 창마다 그 안에서 빽빽한 주의를 하게 하고 어느 토큰이 한 창을 함께 쓰는지만 바꾸어 국소 주의의 계산상 이점을 지킨다.

---

**연습문제 3.**
단계 사이에 조각 합치기를 둔 간단한 두 단계 구조를 구현하도록 `SwinTransformer` 클래스를 고쳐라. 첫 단계는 해상도 $56 \times 56$에서, 둘째 단계는 $28 \times 28$에서 처리해야 한다.

??? success "연습문제 3 풀이"
    ```python
    class PatchMerging(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(4 * dim)

        def forward(self, x, H, W):
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1)
            x = x.view(B, -1, 4 * C)
            x = self.norm(x)
            return self.reduction(x)
    ```
    조각 합치기는 이웃한 $2 \times 2$ 조각 묶음을 통로 차원을 따라 이어 붙여 ($4C$개의 통로를 만든 뒤) 선형으로 $2C$까지 낮추어 사영한다. 그러면 합성곱 신경망의 걸음 있는 합성곱이나 풀링처럼 공간 해상도가 반으로 줄고 특징 차원이 두 배가 된다.

## 정리하며

**다룬 것** — 스윈 트랜스포머

`WindowAttention` 모듈은 자기 주의를 크기가 고정된 (대개 $7 \times 7$ 조각의) 국소 창으로 제한한다.

핵심 클래스는 `WindowAttention`, `SwinTransformer`, `PatchMerging`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
