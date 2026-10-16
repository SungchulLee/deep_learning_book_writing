# 위치 인코딩

위치 인코딩 - 이음 모델을 위한 흔한 클래스. 담긴 것:

여기 구현은 Positional Encodings을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
자리 담기 - 이음 모형을 위한 흔한 클래스
담긴 것:
  - 사인 자리 담기(변환기)
  - 배울 수 있는 자리 담기
  - 도는 자리 담기(RoPE)(깨침을 잡기 위한 도우미)

두루마리: appendix/utils/positional.py
눈여겨볼 것: 배우기 위한 본이다. RoPE은 깨침을 잡을 만큼만 넣었다.
"""

import math
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """
    "어텐션만 있으면 된다"에 나온 옛 사인 자리 담기.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # 버퍼로 올린다. 모형과 함께 갈무리되지만 익히지는 않는다
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        돌려주는 것:
          x + PE[:T]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LearnablePositionalEmbedding(nn.Module):
    """BERT이나 ViT 꼴 모형에서 쓰는, 배울 수 있는 자리 담기."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        return x + self.pos(positions)[None, :, :]


def rope_rotate_half(x):
    """
    RoPE 도우미: 마지막 차수의 짝을 돌린다.
    x = [..., 2i, 2i+1]이면 [-x_{2i+1}, x_{2i}]으로 돌린다
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """
    q과 k에 도는 자리 담기를 건다.
    LLaMA 꼴 모형에서 쓰는, 깨침을 잡기 위한 도우미다.

    q, k: (..., D), D은 짝수다
    cos, sin: (..., D) 또는 q/k으로 펴 맞출 수 있는 꼴
    """
    q_rot = (q * cos) + (rope_rotate_half(q) * sin)
    k_rot = (k * cos) + (rope_rotate_half(k) * sin)
    return q_rot, k_rot


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`)를 매기고, 이들이 어울려 온전한 잔손질 묶음 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 잔손질 묶음에 알맞은지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 잣대 잡는 꾀 -- 배치 잣대 잡기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 드러내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 다독임에 왜 도움이 되는지 밝혀라.

</div>

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 익히는 동안 어텐션 짐 얼마를 아무렇게나 0으로 만들어, 모델이 단어끼리의 어떤 사이에만 기대는 것을 막는다. 그래서 모델이 어텐션을 더 고루 나누고 더 든든한 리프리젠테이션을 배우게 되는데, 이는 여느 드롭아웃이 뉴럴 유닛끼리 함께 길드는 것을 막는 것과 같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 이음 길이 $n$과 모델 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 구조를 이끄는 까닭은 무엇인가?

</div>

??? success "연습문제 3 풀이"
    여느 셀프 어텐션은 $n \times n$ 어텐션 행렬을 셈하므로 때는 $O(n^2 d)$, 어텐션 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 그 자리 미닫이 창 어텐션($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 단어에 대한 성긴 두루 어텐션을 아울러 쓴다. Linformer는 키와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 입력에서 잘 들게 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`SinusoidalPositionalEncoding`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = SinusoidalPositionalEncoding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 위치 인코딩

이 구현은 클래스 2개(`SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`)를 매기고, 이들이 어울려 온전한 잔손질 묶음 구조를 이룬다.

핵심 클래스는 `SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
