# 비전 트랜스포머 ViT

ViT는 이미지를 패치 토큰으로 바꾸어 트랜스포머로 분류한다. 패치 투영은 컨볼루션으로도 구현할 수 있으며, “컨볼루션이 전혀 없는 연산”이라는 뜻보다 공간 피처 추출의 중심을 어텐션에 둔 구조로 이해한다. 아래 모델은 무작위 초기화 예제다.

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

**출력:**

```
Parameters: 86,567,656
```

## 2. 논의

PatchEmbedding은 커널과 스트라이드를 patch_size로 둔 Conv2d로 겹치지 않는 패치를 투영한다. $(B,3,224,224)$에서 패치 196개가 생기고 [CLS]를 붙이면 토큰 수는 197개다. 반환 분류 로짓은 $(B,C)$다.

위치 표의 길이는 생성자의 img_size로 고정된다. 실제 입력이 다른 패치 수를 만들면 더하기에서 모양 오류가 나며, 패치 크기로 나누어떨어지지 않는 가장자리 픽셀은 기본 컨볼루션에서 버려질 수 있다. 새 해상도에 맞는 모델을 만드는 것과 기존 사전 학습 위치 표를 보간해 옮기는 것은 다르다.

이 코드는 PyTorch TransformerEncoderLayer의 Post-LN·ReLU 기본값을 사용한다. 원래 ViT의 모든 세부 설정과 사전 학습을 재현하지 않는다. ViT에도 패치 분할·공유 투영·위치 표 같은 귀납 바이어스는 있다. CNN과 비교해 국소적 연결 가정이 약하다는 뜻이지 귀납 바이어스가 전혀 없다는 뜻은 아니다. 학습 데이터 요구량은 사전 학습·증강·증류 등의 조건에 따라 달라진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
384×384 입력의 패치 수와 아래 새 모델 생성이 기존 웨이트 이전과 다른 점을 설명하라.

</div>

??? success "연습문제 1 풀이"
    패치 수는 24×24=576개, [CLS] 포함 길이는 577개다. 아래 코드는 새 모델을 생성하므로 기존 학습 웨이트를 옮기지 않는다.
    ```python
    model = VisionTransformer(img_size=384, patch_size=16, num_classes=10)
    x = torch.randn(1, 3, 384, 384)
    out = model(x)
    print(out.shape)  # (1, 10)
    ```
    출력은 (B,10)이다. 기존 224 해상도 모델을 옮기려면 패치 위치 표의 2차원 격자를 보간하는 등 별도 처리가 필요하며 단순 reshape만으로 같은 표현을 얻지는 못한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
[CLS]와 패치 평균 풀링을 비교할 때 주의할 점은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    [CLS]는 로스를 통해 정보를 모으는 역할을 학습하고 평균 풀링은 마지막 패치 표현들을 평균한다. 후자에서도 각 패치가 이미 어텐션으로 다른 패치 정보를 읽을 수 있으므로 중요도를 학습할 수 없다고 단정해서는 안 된다. 같은 학습 조건에서 선택하며 [CLS]를 평균에 포함할지도 명시한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 겹치는 패치 투영의 토큰 수와 전체 모델에 연결할 때 필요한 변경을 설명하라.

</div>

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
    패딩 0일 때 한 축의 수는 $\lfloor(224-16)/12\rfloor+1=18$로 총 324개다. [CLS]까지 포함한 점수 행렬 비율은 $(325/197)^2\approx2.72$다. 피처 투영·FFN까지 모두 같은 비율로 늘지는 않는다. 이 조각을 연결하려면 patch_embed뿐 아니라 위치 표 길이와 토큰 수 계산도 함께 바꿔야 한다.

## 정리하며

이미지 패치 수는 위치 표와 어텐션 비용을 결정한다. 이 예제는 사전 학습 ViT 전체를 재현하지 않으며, 해상도·겹치는 패치 변경 시 토큰 길이와 웨이트 이전을 함께 설계해야 한다.
