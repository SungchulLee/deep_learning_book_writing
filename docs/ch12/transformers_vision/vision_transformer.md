# 비전 트랜스포머

외부 PatchEmbedding과 PyTorch 인코더를 연결해 이미지 분류 로짓을 만드는 간단한 ViT 예제다. 사전 학습 웨이트는 읽지 않으며, 패치 투영·CLS·위치 표와 분류 출력을 연결하는 데 초점을 둔다.

## 1. 코드

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
    pass
```

## 2. 논의

기본 입력은 $(B,3,224,224)$다. 패치 투영 뒤 $(B,196,768)$에 CLS 하나를 붙여 $(B,197,768)$로 만들고 위치 표를 더한다. $(197,B,768)$로 전치해 인코더에 넣은 뒤 첫 위치를 분류해 $(B,1000)$ 로짓을 반환한다.

num_layers 인수가 이미 깊이를 정한다. 기본 TransformerEncoderLayer는 Post-LN·ReLU를 사용하며 원래 ViT의 Pre-LN 구성과 같지 않다. 바깥의 추가 최종 LayerNorm도 없다. 무작위 CLS·위치 표와 계수 수를 사전 학습 성능으로 해석하지 않는다.

patch_embed.n_patches로 위치 길이를 생성하므로 실제 입력이 다른 패치 수를 만들면 더하기에서 모양이 맞지 않는다. 정사각 img_size 메타데이터만 저장하며 자동 위치 보간·패딩 마스크는 없다. 외부 모듈의 검색 경로와 패치 클래스 인터페이스도 맞춰야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
기본 모델의 파라메터를 구성별로 구하라.

</div>

??? success "풀이"
    패치 투영 590,592개, CLS 768개, 위치 표 151,296개, 인코더 12레이어 85,054,464개, 분류기 769,000개로 총 86,566,120개다. 별도의 최종 정규화가 있는 다른 ViT 예제와 수가 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력 검증에서 무엇을 확인해야 하는가?

</div>

??? success "풀이"
    4차원 실수 텐서의 RGB 채널, 장치·자료형, 실제 패치 격자와 위치 표 길이를 확인한다. 입력 피처를 무조건 float32로 고정하는 것은 혼합 정밀도 등에서 불필요하다. 임의 직사각형이 같은 패치 수를 만들더라도 공간 좌표 의미가 같다고 가정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
해상도를 바꾸거나 깊이를 조절할 때 각각 무엇을 바꾸는가?

</div>

??? success "풀이"
    해상도는 패치 격자·위치 표와 웨이트 보간·미세 조정 문제다. 깊이는 이미 num_layers로 정하며 계수 수와 계산량을 바꾼다. 한 변경이 다른 조건까지 자동으로 해결하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

이 모델은 패치 토큰과 CLS 분류의 흐름을 보여 준다. PyTorch 기본 블록의 세부 설정, 위치 표의 고정 길이와 사전 학습 여부를 구분해 다른 ViT 구현과 비교한다.
