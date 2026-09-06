# 비전 트랜스포머

비전 트랜스포머 (ViT)

트랜스포머에 바탕을 둔 구조는 자연어 처리를 뒤바꾸어 놓았다. 이 구현은 트랜스포머의 개념을 살피며, 갖가지 과제에서 최고 수준의 성능을 내게 하는 주의 얼개와 구조의 본을 보여 준다.

## 코드

```python
"""
비전 트랜스포머 (ViT)
"""
import torch
import torch.nn as nn
from patch_embedding import PatchEmbedding

# ========================================================================
# 메인
# ========================================================================

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, d_model=768, 
                 num_heads=12, num_layers=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=d_model)
        
        # 분류 토큰과 자리 임베딩
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, d_model))
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 분류 머리
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # 조각 임베딩
        x = self.patch_embed(x)  # [B, n_patches, d_model]
        
        # 분류 토큰을 더한다
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 자리 임베딩을 더한다
        x = x + self.pos_embed
        
        # 트랜스포머
        x = x.transpose(0, 1)  # [seq, batch, dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch, seq, dim]
        
        # 분류
        cls_output = x[:, 0]
        return self.classifier(cls_output)


if __name__ == "__main__":
    pass```

## 논의

`VisionTransformer` 클래스는 파이토치의 `nn.Module` 인터페이스로 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정하여, 학습 중에 파이토치의 자동 미분 체계가 기울기 계산을 알아서 하게 한다. 이 모듈 방식의 설계 덕분에 낱낱의 부품을 고치거나 모델을 더 큰 파이프라인에 끼워 넣기가 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `VisionTransformer`에 든 학습 가능한 매개변수의 총 개수를 셈하라. 가중치와 편향을 모두 넣어 층별로 나누어 보여라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
`VisionTransformer`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = VisionTransformer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.
