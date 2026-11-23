# 어텐션 구조

자주 쓰는 어텐션 구성 요소를 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 어텐션 메커니즘의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
어텐션 구조 - 자주 쓰는 구성 요소
포함한 것:
  - 스케일 내적 어텐션
  - 멀티헤드 어텐션(가장 단순한 형태)
  - 가산(바다나우) 어텐션(seq2seq용)

파일: appendix/utils/attention.py
참고: 주석을 넉넉히 단 교육용 예시 구현이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    스케일 내적 어텐션(트랜스포머 어텐션의 핵심).

    입력:
      Q: 쿼리   (B, H, Tq, Dh)
      K: 키     (B, H, Tk, Dh)
      V: 값    (B, H, Tk, Dh)
      mask: (B, H, Tq, Tk)로 브로드캐스트할 수 있는, 선택적 어텐션 마스크
            - True/1이면 "남긴다", False/0이면 "마스킹한다"

    출력:
      out: (B, H, Tq, Dh)
      attn: (B, H, Tq, Tk) 어텐션 웨이트
    """
    Dh = Q.size(-1)

    # 정규화 전 어텐션 점수: (B, H, Tq, Tk)
    scores = (Q @ K.transpose(-2, -1)) / (Dh ** 0.5)

    # 마스크가 주어지면 소프트맥스 앞에서 마스킹된 위치를 -inf로 둔다
    if mask is not None:
        # 마스크를 불리언으로 바꾼다. False이면 마스킹된 것이다
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    # 키 차원을 따라 소프트맥스
    attn = F.softmax(scores, dim=-1)

    # 밸류의 가중합
    out = attn @ V
    return out, attn


class MultiHeadAttention(nn.Module):
    """
    가장 단순한 멀티헤드 어텐션 레이어.

    스텝:
      1) 입력을 Q,K,V로 투영한다
      2) 헤드로 나눈다
      3) 스케일 내적 어텐션을 적용한다
      4) 헤드를 합치고 출력으로 투영한다
    """
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, context=None, mask=None):
        """
        x: (B, Tq, D) 쿼리는 x에서 온다
        context: (B, Tk, D) 키와 밸류는 context에서 온다(None이면 셀프 어텐션)
        mask: (B, H, Tq, Tk)로 브로드캐스트할 수 있는, 선택적 마스크
        """
        if context is None:
            context = x

        B, Tq, D = x.shape
        Tk = context.size(1)

        # 모델 차원에서 Q,K,V로 투영한다
        Q = self.q_proj(x)        # (B, Tq, D)
        K = self.k_proj(context)  # (B, Tk, D)
        V = self.v_proj(context)  # (B, Tk, D)

        # 헤드 형태로 바꾼다: (B, H, T, Dh)
        Q = Q.view(B, Tq, self.nhead, self.dh).transpose(1, 2)
        K = K.view(B, Tk, self.nhead, self.dh).transpose(1, 2)
        V = V.view(B, Tk, self.nhead, self.dh).transpose(1, 2)

        # 어텐션을 계산한다
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 헤드를 다시 합친다: (B, Tq, D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.out_proj(out)
        return out, attn


class AdditiveAttention(nn.Module):
    """
    가산(바다나우) 어텐션. seq2seq RNN 모델에서 흔하다.

    점수:
      e_{t,s} = v^T tanh(W_h h_s + W_q q_t)

    여기서:
      - h_s = 소스 위치 s에서의 인코더 은닉 상태
      - q_t = 타깃 위치 t에서의 디코더 은닉 상태
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, query):
        """
        encoder_outputs: (B, S, H)
        query: (B, H)  (시각 t에서의 디코더 은닉 상태 따위)

        반환:
          context: (B, H)
          alpha:   (B, S) 소스 위치에 대한 어텐션 웨이트
        """
        # 인코더 출력과 쿼리를 같은 공간으로 투영한다
        h_proj = self.W_h(encoder_outputs)              # (B, S, H)
        q_proj = self.W_q(query).unsqueeze(1)           # (B, 1, H)

        # 점수: (B, S, 1) -> (B, S)
        scores = self.v(torch.tanh(h_proj + q_proj)).squeeze(-1)

        # 소스 토큰에 대한 어텐션 웨이트
        alpha = F.softmax(scores, dim=1)                # (B, S)

        # 인코더 출력의 가중합
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return context, alpha


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 함수 `scaled_dot_product_attention`과 클래스 2개(`MultiHeadAttention`, `AdditiveAttention`)를 정의하고, 이들이 함께 하나의 유틸리티 모듈을 이룬다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `MultiHeadAttention`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. `MultiHeadAttention`은 `d_model`과 `nhead`에 기본값이 없으므로 `d_model`을 $D$라 하면, `q_proj`·`k_proj`·`v_proj`·`out_proj` 넷이 모두 `bias=False`인 $D \times D$ 웨이트라 저마다 $D^2$개, 모두 $4D^2$개다. 헤드 수 `nhead`는 이 수에 영향을 주지 않는다 — 같은 $D \times D$ 투영을 헤드로 나누어 쓸 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 웨이트 뒤(밸류와 곱하기 앞)에 드롭아웃 레이어를 추가하여라. 트레인하는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하여라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 트레인하는 동안 어텐션 웨이트 일부를 무작위로 0으로 만들어, 모델이 토큰 사이의 특정 관계에만 의존하는 것을 막는다. 그래서 모델이 어텐션을 더 고르게 분산하고 더 강건한 표현을 학습하게 되는데, 이는 일반 드롭아웃이 뉴런끼리의 공적응(co-adaptation, 서로에게 지나치게 의존하게 되는 것)을 막는 것과 같다.

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
`MultiHeadAttention`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = MultiHeadAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 어텐션 구조

이 구현은 클래스 2개(`MultiHeadAttention`, `AdditiveAttention`)를 정의하고, 이들이 함께 하나의 유틸리티 모듈을 이룬다.

핵심 클래스는 `MultiHeadAttention`, `AdditiveAttention`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
