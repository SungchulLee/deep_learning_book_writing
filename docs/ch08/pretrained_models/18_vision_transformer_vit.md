# 비전 트랜스포머 ViT

2020년 논문 "An Image is Worth 16x16 Words"에서 나온 비전 트랜스포머(ViT)는 합성곱 없이 트랜스포머 구조를 이미지 분류에 곧바로 쓴다. 그림을 크기가 고정된 조각으로 나누고 조각마다 토큰으로 다루어, 큰 데이터셋으로 사전 학습하면 순수 트랜스포머가 합성곱 신경망의 성능과 맞먹거나 앞설 수 있음을 보인다.

## 1. 코드

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                embed_dim, num_heads,
                dim_feedforward=embed_dim * 4, batch_first=True
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


if __name__ == "__main__":
    model = VisionTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 2. 논의

조각 임베딩 층은 그림 데이터와 트랜스포머를 잇는 매우 중요한 다리이다. 핵 크기와 걸음이 조각 크기와 같은 합성곱이 겹치지 않는 조각을 뽑아 저마다 $d$차원 임베딩으로 사영한다. 조각이 $16 \times 16$인 $224 \times 224$ 그림에서는 토큰 $14 \times 14 = 196$개가 나와, 공간 격자가 트랜스포머가 다룰 수 있는 수열로 바뀐다.

학습되는 `[CLS]` 토큰을 수열 앞에 붙이고 토큰마다 학습되는 위치 임베딩을 더한다. `[CLS]` 토큰은 모든 층에 걸친 자기 주의로 전역 정보를 모으며 끝내 분류에 쓰인다. 위치 임베딩은 조각을 수열로 펼 때 사라졌을 공간 짜임을 담는다.

비전 트랜스포머에는 (합성곱 신경망에 붙박인 국소성이나 옮김 동변성 같은) 귀납 편향이 없어서 좋은 성능을 내려면 큰 규모의 사전 학습이 필요하다. 그러나 ImageNet-21k나 JFT-300M 같은 데이터셋으로 한번 사전 학습하고 나면 아래쪽 과제로 잘 옮겨 가고 합성곱 신경망을 앞서는 일이 많은데, 트랜스포머의 자유로움이 붙박인 가정이 적은 것을 메울 수 있음을 보여 준다.

## 연습문제

**연습문제 1.**
조각 크기가 16일 때 $384 \times 384$ 그림에서 나오는 조각의 수를 셈하라. 그다음 이 입력 크기를 받도록 `VisionTransformer`를 고치고 출력의 꼴을 확인하라.

??? success "연습문제 1 풀이"
    조각의 수는 $(384 / 16)^2 = 24^2 = 576$이다.
    ```python
    model = VisionTransformer(img_size=384, patch_size=16, num_classes=10)
    x = torch.randn(1, 3, 384, 384)
    out = model(x)
    print(out.shape)  # (1, 10)
    ```
    트랜스포머 안의 수열 길이는 (`[CLS]` 토큰을 넣어) $576 + 1 = 577$이 되지만 마지막 출력은 여전히 `(batch_size, num_classes)`이다.

---

**연습문제 2.**
비전 트랜스포머가 모든 조각 토큰의 전역 평균 풀링 대신 `[CLS]` 토큰으로 분류하는 까닭을 설명하라. 두 방식의 맞바꿈은 무엇인가?

??? success "연습문제 2 풀이"
    `[CLS]` 토큰은 자기 주의로 모든 조각의 정보를 모으는, 학습되는 요약 노릇을 한다. 전역 평균 풀링은 대신 모든 조각 표현을 똑같이 평균 내는데 중요한 신호가 묽어질 수 있다. `[CLS]` 방식은 모형이 어느 조각이 분류에 가장 중요한지를 배우게 한다. 다만 전역 평균 풀링이 더 간단하고, 특히 모든 공간 영역이 똑같이 뜻있는 과제에서는 비슷한 성능을 낼 때도 있다. 어떤 비전 트랜스포머 변형(이를테면 DeiT)은 잘 맞추면 두 방식의 정확도가 비슷함을 보였다.

---

**연습문제 3.**
걸음을 조각 크기보다 작게 두어 겹치는 조각을 받치도록 `PatchEmbedding` 클래스를 고쳐라. 그러면 토큰의 수와 자기 주의의 계산 비용이 어떻게 달라지는지 따져 보아라.

??? success "연습문제 3 풀이"
    ```python
    class OverlappingPatchEmbedding(nn.Module):
        def __init__(self, img_size=224, patch_size=16, stride=12,
                     in_channels=3, embed_dim=768):
            super().__init__()
            self.num_patches = ((img_size - patch_size) // stride + 1) ** 2
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=stride
            )

        def forward(self, x):
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return x
    ```
    `stride=12`, `patch_size=16`이면 조각의 수가 $((224 - 16) / 12 + 1)^2 \approx 18^2 = 324$이 되어 겹치지 않을 때의 196보다 많다. 자기 주의의 비용은 토큰 수 $N$에 대해 $O(N^2)$이므로 겹치는 조각은 주의 비용을 대략 $(324/196)^2 \approx 2.7$배 늘린다.

## 정리하며

**다룬 것** — 비전 트랜스포머 ViT

조각 임베딩 층은 그림 데이터와 트랜스포머를 잇는 매우 중요한 다리이다.

핵심 클래스는 `PatchEmbedding`, `VisionTransformer`, `OverlappingPatchEmbedding`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
