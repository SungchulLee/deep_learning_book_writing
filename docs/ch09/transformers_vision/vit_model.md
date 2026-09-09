# 비전 트랜스포머 모형

비전 트랜스포머(ViT) 구현. 이미지 분류에서 합성곱 신경망과 트랜스포머를 잇는다

트랜스포머에 바탕을 둔 구조는 자연어 처리를 뒤바꾸어 놓았다. 이 구현은 트랜스포머의 개념을 살피며, 갖가지 과제에서 최고 수준의 성능을 내게 하는 주의 얼개와 구조의 본을 보여 준다.

## 1. 코드

```python
"""
비전 트랜스포머(ViT) 구현
이미지 분류에서 합성곱 신경망과 트랜스포머를 잇는다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """
    그림을 조각으로 바꾸고 임베딩으로 사영한다.
    합성곱 신경망 방식의 입력과 트랜스포머 처리를 잇는 다리이다.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 편 조각의 선형 사영 (합성곱 신경망의 합성곱과 비슷하다)
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, channels, height, width)
        반환값:
            (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    트랜스포머에서 온 다중 머리 자기 주의 얼개.
    모형이 그림의 여러 부분에 한꺼번에 주의하게 해 준다.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, seq_len, embed_dim)
        반환값:
            (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Q, K, V를 만든다
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 주의 점수
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 값에 어텐션 적용
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    트랜스포머 블록에서 쓰는 순전파 신경망.
    비선형성과 특징 변환을 준다.
    """
    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    자기 주의와 다층 퍼셉트론을 갖춘 표준 트랜스포머 인코더 블록.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, 
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 잔차 연결을 곁들인 주의
        x = x + self.attn(self.norm1(x))
        # 잔차 연결을 곁들인 다층 퍼셉트론
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    비전 트랜스포머(ViT) 모형.
    
    핵심 혁신:
    1. 그림을 조각의 수열로 다룬다
    2. (본디 자연어 처리에서 온) 트랜스포머 인코더를 이미지 분류에 쓴다
    3. 합성곱 신경망 방식의 입력 처리와 트랜스포머 구조를 잇는다
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # 조각 임베딩 층 (그림에서 토큰으로 잇는 다리)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # 분류 토큰 (학습되며 수열 앞에 붙인다)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 위치 임베딩 (학습되는 것)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 트랜스포머 인코더 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 분류 머리
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # 가중치 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, channels, height, width)
        반환값:
            (batch_size, n_classes)
        """
        B = x.shape[0]
        
        # 그림을 조각 임베딩으로 바꾼다
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # 분류 토큰을 앞에 붙인다
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # 위치 임베딩을 더한다
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 트랜스포머 블록을 적용한다
        for block in self.blocks:
            x = block(x)
            
        # 분류 토큰으로 분류한다
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 분류 토큰만 쓴다
        x = self.head(cls_token_final)
        
        return x


def create_vit_tiny(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Tiny: 매개변수 500만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, 
        depth=12, n_heads=3, n_classes=n_classes
    )


def create_vit_small(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Small: 매개변수 2200만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384,
        depth=12, n_heads=6, n_classes=n_classes
    )


def create_vit_base(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Base: 매개변수 8600만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768,
        depth=12, n_heads=12, n_classes=n_classes
    )


def create_vit_large(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Large: 매개변수 3억 700만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024,
        depth=24, n_heads=16, n_classes=n_classes
    )


if __name__ == "__main__":
    # 사용 예
    model = create_vit_base(n_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Input shape: torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 10])
Number of parameters: 85,806,346
```

## 2. 논의

이 구현은 함께 어울려 온전한 트랜스포머 구조를 이루는 클래스 5개(`PatchEmbedding`, `MultiHeadAttention`, `MLP`, `TransformerBlock`, 그리고 하나 더)를 정한다. 클래스마다 서로 다른 부품을 감싸 코드를 모듈 방식으로 만들고 넓히기 쉽게 한다. `forward` 메서드가 파이토치가 자동 미분에 쓰는 계산 그래프를 정한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`PatchEmbedding`의 앞먹임을 따라가며 텐서의 꼴을 좇아라. 기본 매개변수로 표본 4개짜리 배치를 넣었을 때 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`를 지금 값에서 3으로 바꾸어라. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$ 공식으로 합성곱과 풀링 층마다 그 뒤의 공간 차원을 다시 셈하라. 마지막 합성곱·풀링 층의 편 출력에 맞도록 첫 선형 층의 `in_features`를 고쳐라. `model = PatchEmbedding(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인하라.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
`PatchEmbedding`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = PatchEmbedding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.

## 정리하며

**다룬 것** — 비전 트랜스포머 모형

이 구현은 함께 어울려 온전한 트랜스포머 구조를 이루는 클래스 5개(`PatchEmbedding`, `MultiHeadAttention`, `MLP`, `TransformerBlock`, 그리고 하나 더)를 정한다.

핵심 클래스는 `PatchEmbedding`, `MultiHeadAttention`, `MLP`, `TransformerBlock`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
