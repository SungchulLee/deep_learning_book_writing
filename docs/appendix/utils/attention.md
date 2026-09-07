# 눈길 얼개

눈길 얼개 - 흔한 벽돌. 담긴 것:

여기 짜보기는 Attention Mechanisms을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
눈길 얼개 - 흔한 벽돌
담긴 것:
  - 잣대 맞춘 점곱 눈길
  - 여러 머리 눈길(가장 단출한 꼴)
  - 더하기(바다나우) 눈길(seq2seq을 위함)

두루마리: appendix/utils/attention.py
눈여겨볼 것: 주석을 넉넉히 단, 배우기 위한 본 짜보기다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    잣대 맞춘 점곱 눈길(변환기 눈길의 고갱이).

    들임:
      Q: 물음   (B, H, Tq, Dh)
      K: 열쇠      (B, H, Tk, Dh)
      V: 값    (B, H, Tk, Dh)
      mask: (B, H, Tq, Tk)으로 펴 맞출 수 있는, 골라 쓰는 눈길 가림
            - True/1이면 "남긴다", False/0이면 "가린다"

    날임:
      out: (B, H, Tq, Dh)
      attn: (B, H, Tq, Tk) 눈길 짐
    """
    Dh = Q.size(-1)

    # 날것 눈길 점수: (B, H, Tq, Tk)
    scores = (Q @ K.transpose(-2, -1)) / (Dh ** 0.5)

    # 가림이 주어지면 소프트맥스 앞에서 가린 자리를 -inf로 둔다
    if mask is not None:
        # 가림을 참거짓으로 바꾼다. False이면 가린 것이다
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    # 열쇠 차수를 따라 소프트맥스
    attn = F.softmax(scores, dim=-1)

    # 값의 짐 실은 합
    out = attn @ V
    return out, attn


class MultiHeadAttention(nn.Module):
    """
    가장 단출한 여러 머리 눈길 켜.

    걸음:
      1) 들임을 Q,K,V으로 되비춘다
      2) 머리로 가른다
      3) 잣대 맞춘 점곱 눈길을 건다
      4) 머리를 합치고 날임으로 되비춘다
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
        x: (B, Tq, D) 물음은 x에서 온다
        context: (B, Tk, D) 열쇠와 값은 context에서 온다(None이면 스스로 눈길)
        mask: (B, H, Tq, Tk)으로 펴 맞출 수 있는, 골라 쓰는 가림
        """
        if context is None:
            context = x

        B, Tq, D = x.shape
        Tk = context.size(1)

        # 모형 차수에서 Q,K,V으로 되비춘다
        Q = self.q_proj(x)        # (B, Tq, D)
        K = self.k_proj(context)  # (B, Tk, D)
        V = self.v_proj(context)  # (B, Tk, D)

        # 머리 꼴로 바꾼다: (B, H, T, Dh)
        Q = Q.view(B, Tq, self.nhead, self.dh).transpose(1, 2)
        K = K.view(B, Tk, self.nhead, self.dh).transpose(1, 2)
        V = V.view(B, Tk, self.nhead, self.dh).transpose(1, 2)

        # 눈길을 셈한다
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 머리를 다시 합친다: (B, Tq, D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.out_proj(out)
        return out, attn


class AdditiveAttention(nn.Module):
    """
    더하기(바다나우) 눈길. seq2seq RNN 모형에서 흔하다.

    점수:
      e_{t,s} = v^T tanh(W_h h_s + W_q q_t)

    여기서:
      - h_s = 밑 자리 s에서의 부호기 숨은 상태
      - q_t = 과녁 자리 t에서의 풀개 숨은 상태
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, query):
        """
        encoder_outputs: (B, S, H)
        query: (B, H)  (때 t에서의 풀개 숨은 상태 따위)

        돌려주는 것:
          context: (B, H)
          alpha:   (B, S) 밑 자리에 대한 눈길 짐
        """
        # 부호기와 물음을 같은 밭으로 되비춘다
        h_proj = self.W_h(encoder_outputs)              # (B, S, H)
        q_proj = self.W_q(query).unsqueeze(1)           # (B, 1, H)

        # 점수: (B, S, 1) -> (B, S)
        scores = self.v(torch.tanh(h_proj + q_proj)).squeeze(-1)

        # 밑 낱말에 대한 눈길 짐
        alpha = F.softmax(scores, dim=1)                # (B, S)

        # 부호기 날임의 짐 실은 합
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return context, alpha


if __name__ == "__main__":
    pass```

## 논의

이 짜보기는 갈래 2개(`MultiHeadAttention`, `AdditiveAttention`)를 매기고, 이들이 어울려 온전한 잔손질 묶음 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `MultiHeadAttention`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율 0.1을 써라. 눈길 드롭아웃이 다독임에 왜 도움이 되는지 밝혀라.

??? success "익힘 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 얼마를 아무렇게나 0으로 만들어, 모형이 낱말끼리의 어떤 사이에만 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 드러냄을 배우게 되는데, 이는 여느 드롭아웃이 신경 낱자리끼리 함께 길드는 것을 막는 것과 같다.

---

**익힘 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "익힘 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**익힘 4.**
`MultiHeadAttention`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MultiHeadAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
