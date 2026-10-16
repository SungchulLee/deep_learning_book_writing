# 패치 임베딩

비전 트랜스포머의 입력 이미지를 겹치지 않는 패치 토큰으로 투영한다. 하나의 Conv2d 뒤 공간 축을 펼치며, 분류·어텐션·위치 인코딩은 이 모듈에 포함되지 않는다.

## 1. 코드

```python
"""
비전 트랜스포머를 위한 그림 조각 임베딩
"""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


if __name__ == "__main__":
    pass
```

## 2. 논의

입력은 $(B,C,H,W)$다. 패치 크기 P, 패딩 0, stride=P이므로 컨볼루션 출력은 $(B,E,\lfloor H/P\rfloor,\lfloor W/P\rfloor)$다. flatten(2)와 transpose(1,2) 뒤 $(B,N,E)$ 토큰이 된다. 단순 1픽셀 스트라이드의 컨볼루션 공식을 적용해서는 안 된다.

기본 RGB 224×224, P=16, E=768이면 14×14=196개 토큰이다. 각 패치에 동일한 선형 투영을 적용하는 것과 같고 바이어스 포함 계수는 E(CP²+1)개다. 기본값은 590,592개다.

img_size와 n_patches는 생성 시 메타데이터일 뿐 forward에서 입력 크기를 검증하지 않는다. 다른 크기를 넣으면 실제 토큰 수는 컨볼루션으로 정해지며 바깥 ViT의 위치 표와 맞춰야 한다. 나누어떨어지지 않는 가장자리와 패치보다 작은 입력도 별도로 처리한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 설정에 이미지 4개를 넣을 때 컨볼루션부터 최종 토큰까지 모양을 적어라.

</div>

??? success "연습문제 1 풀이"
    (4,3,224,224)→(4,768,14,14)→(4,768,196)→(4,196,768)이다. 풀링이나 별도 선형 레이어는 없다. CLS 토큰도 이 모듈에서 붙이지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
RGB 64×64에 P=16을 사용할 때 바꿔야 할 메타데이터와 실제 출력은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    기본 in_channels가 이미 3이다. img_size=64로 구성하면 n_patches=16이고 실제 출력은 (B,16,768)이다. 바깥 모델의 위치 표·CLS 처리를 이 길이에 맞춰야 하며 존재하지 않는 평탄 분류기의 in_features를 수정하는 문제는 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 P×P 영역을 depthwise·pointwise로 투영하면 원래 패치 투영과 표현력이 같은가?

</div>

??? success "연습문제 3 풀이"
    바이어스를 빼면 원래 CP²E개, 분리형은 CP²+CE개다. 계수를 줄일 수 있지만 채널마다 공간 필터 하나를 적용한 뒤 채널을 섞는 제약이 생겨 임의의 원래 투영을 그대로 표현하지는 못한다. 연산·메모리·품질은 별도로 비교한다.

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
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

패치 크기·스트라이드와 입력 해상도가 토큰 길이를 결정한다. 패치 투영의 실제 출력과 생성자의 메타데이터를 구분하고 바깥 위치 표·분류 모델과 연결한다.
