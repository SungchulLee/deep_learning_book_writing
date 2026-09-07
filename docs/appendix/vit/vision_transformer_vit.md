# Vision Transformer VIT

보기 변환기(ViT) - 그림 하나는 16x16 낱말만 하다. 이음: https://arxiv.org/abs/2010.11929

여기 짜보기는 Vision Transformer VIT을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
================================================================================
보기 변환기(ViT) - 그림 하나는 16x16 낱말만 하다
================================================================================

글: "그림 하나는 16x16 낱말만 하다: 크게 키운 그림 알아보기를 위한 변환기"
지은이: 알렉세이 도소비츠키 외(구글 리서치)
이음: https://arxiv.org/abs/2010.11929

================================================================================
발자취에서의 뜻
================================================================================
ViT은 자료가 넉넉할 때 미리 익히면 순수한 변환기 얼개만으로도
그림 가름에서 빼어난 됨됨이를 이룰 수 있음을 보였다.

- 자료가 넉넉하면 ViT은 셈을 4배 적게 쓰고도 ResNet을 앞선다
- 눈길 무늬에서 물체의 자리가 절로 드러난다

================================================================================
고갱이 깨침: 그림을 조각의 이음으로 보기
================================================================================

224×224 그림에 16×16 조각이면: (224/16)² = 196 조각
조각마다 → 선형 담기 → "보기 낱말"의 이음

================================================================================
배움 차례에서의 자리
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
    """Conv2d으로 그림을 조각 담음의 이음으로 바꾼다."""
    
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
    그림 가름을 위한 보기 변환기(ViT)
    
    매개변수:
        img_size: 들임 그림 크기. 기본값: 224
        patch_size: 조각 하나의 크기. 기본값: 16
        num_classes: 날임 갈래의 수. 기본값: 1000
        embed_dim: 담음 차수. 기본값: 768
        depth: 변환기 켜의 수. 기본값: 12
        num_heads: 눈길 머리의 수. 기본값: 12
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
    print("=" * 70)```

## 논의

이 짜보기는 갈래 2개(`PatchEmbedding`, `VisionTransformer`)를 매기고, 이들이 어울려 온전한 보기 변환기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
`PatchEmbedding`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "익힘 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**익힘 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "익힘 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = PatchEmbedding(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**익힘 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "익힘 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer은 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**익힘 4.**
`PatchEmbedding`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = PatchEmbedding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
