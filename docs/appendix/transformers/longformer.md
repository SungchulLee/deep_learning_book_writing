# Longformer

Longformer은 2020년 글 "Longformer: The Long-Document Transformer"에서 나왔다. - 온전한 O(S^2) 눈길을 다음으로 갈음한다: (가) 미닫이 창 그 자리 눈길(O(S * 창)) (나) 골라 쓰는 두루 눈길 낱말(예: [CLS], 물음 낱말).

여기 짜보기는 Longformer을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
Longformer - 긴 글월 변환기
글: "Longformer: 긴 글월 변환기" (2020)
지은이: 이즈 벨타기, 매슈 피터스, 아르만 코한
고갱이 깨침:
  - 온전한 O(S^2) 눈길을 다음으로 갈음한다:
      (가) 미닫이 창 그 자리 눈길 (O(S * window))
      (나) 골라 쓰는 두루 눈길 낱말(예: [CLS], 물음 낱말)

두루마리: appendix/transformers/longformer.py
눈여겨볼 것: *그 자리 눈길*(창을 쓰는 스스로 눈길)을 배우기 위해 짜 본 것이다.
      잘 다듬은 알갱이가 아니라 또렷하게 보이려는 본이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class WindowSelfAttention(nn.Module):
    """
    손쉽게 짠 창 스스로 눈길(알아보기 쉽도록 머리 하나).

    자리 i마다 [i-w, i+w] 안의 낱말에만 눈길을 준다.
    번거로움이 O(S^2)이 아니라 ~O(S * window)이 된다.
    """
    def __init__(self, d_model=256, window=4):
        super().__init__()
        self.window = window
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        Q = self.q(x)  # (B, S, D)
        K = self.k(x)
        V = self.v(x)

        outputs = []
        for i in range(S):
            # 그 자리 창의 번호를 정한다
            left = max(0, i - self.window)
            right = min(S, i + self.window + 1)

            # 낱말 i과 그 자리 창의 낱말 사이 눈길을 셈한다
            q_i = Q[:, i : i + 1, :]            # (B, 1, D)
            k_w = K[:, left:right, :]           # (B, W, D)
            v_w = V[:, left:right, :]           # (B, W, D)

            # 잣대 맞춘 점곱 눈길
            scores = (q_i @ k_w.transpose(1, 2)) / (D ** 0.5)  # (B, 1, W)
            attn = F.softmax(scores, dim=-1)                   # (B, 1, W)
            out_i = attn @ v_w                                 # (B, 1, D)
            outputs.append(out_i)

        y = torch.cat(outputs, dim=1)  # (B, S, D)
        return self.out(y)


class LongformerBlock(nn.Module):
    """창 눈길 + 앞먹임 그물을 쓰는 변환기 꼴 덩이 하나."""
    def __init__(self, d_model=256, window=4, ff_dim=1024):
        super().__init__()
        self.attn = WindowSelfAttention(d_model, window)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 눈길 + 나머지 + 잣대 잡기
        x = self.norm1(x + self.attn(x))

        # 앞먹임 + 나머지 + 잣대 잡기
        x = self.norm2(x + self.ff(x))
        return x


class Longformer(nn.Module):
    """
    창 눈길 덩이를 쓰는 Longformer 꼴 부호기.
    """
    def __init__(self, vocab_size=30522, d_model=256, window=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([LongformerBlock(d_model, window) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(x)   # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = Longformer(vocab_size=1000, d_model=128, window=2, num_layers=2)
    ids = torch.randint(0, 1000, (2, 20))
    logits = model(ids)
    print("logits:", logits.shape)  # (2, 20, 1000)```

## 논의

이 짜보기는 갈래 3개(`WindowSelfAttention`, `LongformerBlock`, `Longformer`)를 매기고, 이들이 어울려 온전한 변환기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `WindowSelfAttention`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

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
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer은 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**익힘 4.**
`WindowSelfAttention`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = WindowSelfAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
