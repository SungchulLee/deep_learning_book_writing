# 위치 인코딩

시퀀스 모델에서 자주 쓰는 위치 인코딩을 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 위치 인코딩의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
위치 인코딩 - 시퀀스 모델에서 자주 쓰는 종류
포함한 것:
  - 사인파 위치 인코딩(트랜스포머)
  - 학습 가능한 위치 임베딩
  - 회전 위치 임베딩(RoPE)(아이디어를 보이기 위한 도우미)

파일: appendix/utils/positional.py
참고: 교육용 예시다. RoPE는 아이디어를 보일 만큼만 넣었다.
"""

import math
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """
    "Attention Is All You Need"에 나온 고전적인 사인파 위치 인코딩.

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

        # 버퍼로 등록한다. 모델과 함께 저장되지만 학습되지는 않는다
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        반환:
          x + PE[:T]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LearnablePositionalEmbedding(nn.Module):
    """BERT나 ViT 형태 모델에서 쓰는, 학습 가능한 위치 임베딩."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        return x + self.pos(positions)[None, :, :]


def rope_rotate_half(x):
    """
    RoPE 도우미: 마지막 차원의 쌍을 회전시킨다.
    x = [..., 2i, 2i+1]이면 [-x_{2i+1}, x_{2i}]로 회전시킨다
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """
    q와 k에 회전 위치 임베딩을 적용한다.
    LLaMA 형태 모델에서 쓰는, 아이디어를 보이기 위한 도우미다.

    q, k: (..., D), D는 짝수다
    cos, sin: (..., D) 또는 q/k로 브로드캐스트할 수 있는 형태
    """
    q_rot = (q * cos) + (rope_rotate_half(q) * sin)
    k_rot = (k * cos) + (rope_rotate_half(k) * sin)
    return q_rot, k_rot


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`)를 정의하고, 이들이 함께 하나의 유틸리티 모듈을 이룬다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 유틸리티 모듈에 알맞은지 설명하여라.

</div>

??? success "풀이"
    이 코드에서는 이런 셋을 들 수 있다. (1) 사인파 표를 생성자에서 한 번 계산해 `register_buffer`로 등록한다 -- 학습되지 않는 값이지만 `state_dict`에 저장되고 `.to(device)`를 따라 옮겨지므로, 순전파마다 다시 계산하지 않아도 된다. (2) 주파수 $1/10000^{2i/d}$를 거듭제곱 대신 `exp`와 `log`로 계산한다 -- 지수가 큰 거듭제곱을 직접 구하지 않아 수치적으로 안정하고, 모든 차원을 한 번에 벡터 연산으로 만든다. (3) 위치를 입력에 더하는 방식(사인파·학습형)과 쿼리·키를 회전시키는 방식(RoPE)을 함께 둔다 -- 앞의 둘은 임베딩에 절대 위치를 더하는 반면, RoPE는 쿼리와 키의 내적이 두 위치의 차이에만 의존하게 만들어 상대 위치를 담는다. 학습형 임베딩은 `max_len`을 넘는 위치를 다룰 수 없지만 사인파와 RoPE는 식으로 어떤 위치든 계산할 수 있다는 차이도 있다. 단, 이 코드의 사인파 표는 `max_len`까지만 미리 만들어 둔다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 웨이트 뒤(값과 곱하기 앞)에 드롭아웃 레이어를 추가하여라. 학습하는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하여라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 학습하는 동안 어텐션 웨이트 일부를 무작위로 0으로 만들어, 모델이 토큰 사이의 특정 관계에만 의존하는 것을 막는다. 그래서 모델이 어텐션을 더 고르게 분산하고 더 강건한 표현을 학습하게 되는데, 이는 일반 드롭아웃이 뉴런끼리의 공적응(co-adaptation, 서로에게 지나치게 의존하게 되는 것)을 막는 것과 같다.

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
`SinusoidalPositionalEncoding`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    `SinusoidalPositionalEncoding`은 학습하는 파라메터가 하나도 없고(표는 버퍼다), 위치 인코딩은 보통 입력에 한 번만 더한다. 그래서 깊이를 늘리는 대상은 위치 인코딩 자체가 아니라 그 뒤에 쌓는 블록이다. 위치 인코딩을 맨 앞에 한 번 두고 그 뒤의 블록을 아래처럼 `num_layers`개 쌓는다. 하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = SinusoidalPositionalEncoding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다. 위치 인코딩만 담은 모듈이라면 이 수는 늘 0이 나온다.

## 정리하며

**다룬 것** — 위치 인코딩

이 구현은 클래스 2개(`SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`)를 정의하고, 이들이 함께 하나의 유틸리티 모듈을 이룬다.

핵심 클래스는 `SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
