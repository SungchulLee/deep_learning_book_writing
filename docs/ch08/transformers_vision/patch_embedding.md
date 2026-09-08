# 조각 임베딩

비전 트랜스포머를 위한 그림 조각 임베딩

트랜스포머에 바탕을 둔 구조는 자연어 처리를 뒤바꾸어 놓았다. 이 구현은 트랜스포머의 개념을 살피며, 갖가지 과제에서 최고 수준의 성능을 내게 하는 주의 얼개와 구조의 본을 보여 준다.

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
    pass```

## 2. 논의

`PatchEmbedding` 클래스는 파이토치의 `nn.Module` 인터페이스로 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정하여, 학습 중에 파이토치의 자동 미분 체계가 기울기 계산을 알아서 하게 한다. 이 모듈 방식의 설계 덕분에 낱낱의 부품을 고치거나 모델을 더 큰 파이프라인에 끼워 넣기가 쉽다.

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
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

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

**다룬 것** — 조각 임베딩

`PatchEmbedding` 클래스는 파이토치의 `nn.Module` 인터페이스로 모델 구조를 감싼다.

핵심 클래스는 `PatchEmbedding`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
