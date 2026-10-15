# 비전 트랜스포머 모델

패치 투영, 멀티헤드 어텐션, GELU MLP와 Pre-LN 블록으로 ViT 분류 모델을 구성한다. Tiny·Small·Base·Large 함수는 크기를 정하는 팩터리이며 사전 학습 체크포인트를 불러오지는 않는다.

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
    트랜스포머에서 온 다중 머리 자기 주의 구조.
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

기본 패치 16과 224 해상도에서 196개 패치에 CLS를 더해 197개 토큰을 사용한다. MultiHeadAttention은 $(B,N,E)$를 $(B,H,N,E/H)$로 나누고 점수·가중합 뒤 원래 특징 차원으로 합친다. E는 H로 나누어떨어져야 한다.

TransformerBlock은 입력을 정규화한 뒤 어텐션·MLP를 실행하고 잔차를 더하는 Pre-LN이다. 스택 끝에도 LayerNorm을 적용한다. 어텐션 확률·출력과 MLP 등에 드롭아웃이 있어 학습·평가 모드를 구분한다. 클래스 다섯 개의 역할은 PatchEmbedding, MultiHeadAttention, MLP, TransformerBlock과 VisionTransformer다.

정사각 img_size로 위치 표를 고정하며 입력 해상도 자동 보간과 패딩 마스크는 없다. 마지막 head는 로짓을 반환한다. 초기화는 Linear·LayerNorm과 CLS·위치 표를 다루지만 Conv2d는 기본 초기화를 유지한다. main은 Base의 출력 클래스를 10으로 바꾼 무작위 순전파 예제다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
main의 패치·CLS·최종 출력 모양과 매개변수 수를 설명하라.

</div>

??? success "연습문제 1 풀이"
    입력 (2,3,224,224)에서 패치 (2,196,768), CLS 포함 (2,197,768), 출력 (2,10)이다. 계수는 85,806,346개이며 학습 결과가 아니라 구조의 수다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
RGB 64×64, 패치 16으로 구성하려면 무엇을 맞춰야 하는가?

</div>

??? success "연습문제 2 풀이"
    in_channels는 기본값이 이미 3이다. VisionTransformer(img_size=64,patch_size=16)로 구성하면 16패치와 CLS 하나가 된다. 기존 224 모델의 위치 표는 맞지 않으므로 재초기화·보간과 미세 조정 조건을 정해야 한다. 풀링·평탄 분류층의 크기를 바꾸는 CNN 문제가 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
패치 크기를 줄일 때 어텐션과 다른 층의 비용은 어떻게 달라지는가?

</div>

??? success "연습문제 3 풀이"
    양 축의 패치 수가 두 배씩이면 토큰은 대략 네 배, 전체 점수·가중합은 대략 열여섯 배다. 투영·MLP는 토큰 수에 선형이므로 약 네 배다. CLS와 경계·배치·특징 차원을 반영해 실제 비용을 비교한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

직접 구현한 Pre-LN ViT의 각 모듈을 텐서 축과 연결해 이해한다. 크기 팩터리는 사전 학습을 뜻하지 않으며 해상도·패치 변경은 위치 표와 연산량도 함께 바꾼다.
