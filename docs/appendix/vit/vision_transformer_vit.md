# Vision Transformer VIT

비전 트랜스포머(ViT) - 이미지 한 장은 16x16 단어의 가치가 있다. 링크: https://arxiv.org/abs/2010.11929

이 쪽의 구현은 ViT의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
================================================================================
비전 트랜스포머(ViT) - 이미지 한 장은 16x16 단어의 가치가 있다
================================================================================

논문: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
저자: 알렉세이 도소비츠키 외(구글 리서치)
링크: https://arxiv.org/abs/2010.11929

================================================================================
역사적 의의
================================================================================
ViT는 데이터가 충분할 때 사전 학습하면 순수한 트랜스포머 구조만으로도
이미지 분류에서 뛰어난 성능을 낼 수 있음을 보였다.

- 데이터가 충분하면 ViT는 계산량을 4배 적게 쓰고도 ResNet을 앞선다
- 어텐션 패턴에서 물체의 위치가 저절로 드러난다

================================================================================
핵심 아이디어: 이미지를 패치의 시퀀스로 보기
================================================================================

224×224 이미지에 16×16 패치이면: (224/16)² = 196 패치
패치마다 → 선형 임베딩 → "시각 단어"의 시퀀스

================================================================================
학습 순서에서의 위치
================================================================================

이어지는 것: swin_transformer.py, convnext.py
================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """Conv2d로 이미지를 패치 임베딩의 시퀀스로 바꾼다."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    이미지 분류를 위한 비전 트랜스포머(ViT)
    
    인자:
        img_size: 입력 이미지 크기. 기본값: 224
        patch_size: 패치 하나의 크기. 기본값: 16
        num_classes: 출력 클래스의 수. 기본값: 1000
        embed_dim: 임베딩 차원. 기본값: 768
        depth: 트랜스포머 레이어의 수. 기본값: 12
        num_heads: 어텐션 헤드의 수. 기본값: 12
    """
    
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, batch_first=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 70)
    print("Vision Transformer (ViT-B/16)")
    print("=" * 70)
    
    model = VisionTransformer()
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print("=" * 70)
```

**출력:**

```
======================================================================
Vision Transformer (ViT-B/16)
======================================================================
Parameters: 86,567,656
Input: torch.Size([2, 3, 224, 224]), Output: torch.Size([2, 1000])
======================================================================
```

## 2. 논의

이 구현은 클래스 2개(`PatchEmbedding`, `VisionTransformer`)를 정의하고, 이들이 함께 완전한 비전 트랜스포머 구조를 이룬다. 인코더 레이어는 직접 짜지 않고 PyTorch의 `nn.TransformerEncoderLayer`를 12개 쌓았다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`PatchEmbedding`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 입력 샘플 4개를 배치로 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 시작해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라. `PatchEmbedding`에는 풀링도 선형 레이어도 없고, 커널과 스트라이드가 모두 16인 컨볼루션 하나뿐이다. $(4, 3, 224, 224)$가 컨볼루션을 지나면 $(4, 768, 14, 14)$, `flatten(2)` 뒤에 $(4, 768, 196)$, `transpose(1, 2)` 뒤에 $(4, 196, 768)$이 된다. 패치 196개가 저마다 768차원 토큰이 된 것이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    패치 임베딩 컨볼루션은 이미 입력 채널이 3이다. 바꿀 것은 이미지 크기다. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 계산하면 $64 \times 64$ 입력은 $4 \times 4$ 패치 격자, 곧 패치 16개가 된다. 위치 임베딩 `pos_embed`는 `1 + num_patches`개를 담아야 하므로 `VisionTransformer(img_size=64)`로 만들어 패치 수를 16으로 다시 계산하게 한다. 분류 헤드는 `[CLS]` 토큰의 768차원만 받으므로 바꿀 필요가 없다. `model = PatchEmbedding(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 나타내어라. 이것이 긴 시퀀스를 위해 Longformer나 Linformer 같은 구조가 나온 동기가 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    표준 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 슬라이딩 윈도 국소 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택한 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 투영하여 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`PatchEmbedding`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = PatchEmbedding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Vision Transformer VIT

이 구현은 클래스 2개(`PatchEmbedding`, `VisionTransformer`)를 정의하고, 이들이 함께 완전한 비전 트랜스포머 구조를 이룬다.

핵심 클래스는 `PatchEmbedding`, `VisionTransformer`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
