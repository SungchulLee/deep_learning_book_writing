# Swin Transformer

Swin Transformer는 2021년 논문 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"에서 나왔다. 효율을 위해 이동 윈도(shifted window) 어텐션과 계층적 피처 맵을 쓴다. 어텐션을 이미지 전체가 아니라 작은 윈도 안에서만 계산하고, 레이어마다 윈도를 어긋나게 옮겨 이웃 윈도끼리도 정보가 섞이게 하는 방식이다.

이 쪽의 구현은 Swin Transformer의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Swin Transformer - 이동 윈도를 쓰는 계층적 비전 트랜스포머
논문: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
핵심: 효율을 위한 이동 윈도 어텐션과 계층적 피처 맵
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.head = nn.Linear(embed_dim * 8, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(-1)
        return self.head(x)

if __name__ == "__main__":
    model = SwinTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 773,704
```

## 2. 논의

이 구현은 클래스 2개(`WindowAttention`, `SwinTransformer`)를 정의한다. 다만 완전한 Swin Transformer는 아니다. `SwinTransformer`는 $4 \times 4$ 패치 임베딩 뒤에 곧바로 평균 풀링을 하고 분류 헤드로 넘길 뿐, `WindowAttention`을 쓰지 않으며 `depths`와 `num_heads` 인자도 받기만 하고 쓰지 않는다. 윈도 분할, 윈도 이동, 패치 병합(patch merging)으로 해상도를 줄이는 계층 구조는 들어 있지 않다. 또 헤드는 `embed_dim * 8`차원을 받도록 되어 있는데 풀링한 출력은 `embed_dim`차원이라, 지금 코드로는 순전파가 오류를 낸다(출력은 파라메터 수만 센 것이다). `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`WindowAttention`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 입력 샘플 4개를 배치로 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 시작해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라. `WindowAttention`은 이미지가 아니라 토큰 시퀀스 $(B, N, C)$를 받고 컨볼루션도 풀링도 없다. 첫 스테이지처럼 `dim=96`, `num_heads=3`이고 윈도 하나($7 \times 7 = 49$ 토큰)를 넣으면 입력 $(4, 49, 96)$이 `qkv`를 지나 $(4, 49, 288)$, 재배열 뒤 $(3, 4, 3, 49, 32)$가 되어 $q, k, v$가 저마다 $(4, 3, 49, 32)$다. 어텐션 행렬은 $(4, 3, 49, 49)$, 값과 곱해 헤드를 합치면 $(4, 49, 96)$이고 `proj`가 형태를 유지한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    `SwinTransformer`의 패치 임베딩 컨볼루션은 이미 입력 채널이 3이다. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 계산하면 $64 \times 64$ 입력은 $16 \times 16$ 패치 격자, 곧 토큰 256개가 되고, 평균 풀링이 이를 96차원 벡터 하나로 모으므로 입력 크기 때문에 고칠 것은 없다. 오류를 내는 것은 헤드다. 선형 레이어 `head`의 `in_features`가 `embed_dim * 8`인데 들어오는 것은 `embed_dim`차원이므로, 이를 `embed_dim`으로 맞추거나 패치 병합 스테이지를 넣어 채널을 8배로 늘려야 한다. 마지막 검사는 `WindowAttention`이 아니라 `SwinTransformer`로 한다. `model = WindowAttention(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 나타내어라. 이것이 긴 시퀀스를 위해 Longformer나 Linformer 같은 구조가 나온 동기가 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    표준 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 슬라이딩 윈도 국소 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택한 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 밸류를 더 낮은 차원 $k \ll n$으로 투영하여 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`WindowAttention`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = WindowAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Swin Transformer

이 구현은 클래스 2개(`WindowAttention`, `SwinTransformer`)를 정의하지만, 윈도 이동과 패치 병합은 담지 않은 뼈대다.

핵심 클래스는 `WindowAttention`, `SwinTransformer`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
