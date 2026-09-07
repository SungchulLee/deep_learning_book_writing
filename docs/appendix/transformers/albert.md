# ALBERT

ALBERT은 2019년 글 "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"에서 나왔다. 1) 나눈 담음 매기기: 낱말 담음 차수(E)를 숨은 차수(H)보다 작게 두고 낱말 -> E으로 담은 뒤 E -> H으로 되비춘다 2) 켜를 넘나드는 매개변수 나눠 쓰기: 같은 변환기 켜의 짐을 모든 켜에서 되쓴다 3) 월 차례 미루어 보기(SOP)(흔히 NSP과 견주어 이야기된다).

여기 짜보기는 ALBERT을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
ALBERT - 가벼운 BERT
글: "ALBERT: 말 드러냄을 스스로 이끌며 배우기 위한 가벼운 BERT" (2019)
지은이: 전중 란 외
고갱이 깨침:
  1) 나눈 담음 매기기:
       낱말 담음 차수(E)를 숨은 차수(H)보다 작게 둔다
       담기: 낱말 -> E, 그다음 E -> H으로 되비춘다
  2) 켜를 넘나드는 매개변수 나눠 쓰기:
       같은 변환기 켜의 짐을 모든 켜에서 되쓴다
  3) 월 차례 미루어 보기(SOP)(흔히 NSP과 견주어 이야기된다)

두루마리: appendix/transformers/albert.py
눈여겨볼 것: 나눈 담음과 나누어 쓰는 켜를 보이는, 배우기 위한 짜보기다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class SharedTransformerBlock(nn.Module):
    """여러 번 되쓰려고 만든 변환기 부호기 켜 하나."""
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        return self.layer(x, src_key_padding_mask=src_key_padding_mask)


class ALBERT(nn.Module):
    """
    다음을 갖춘 부호기만 있는 ALBERT 모형:
      - 나눈 담음
      - 나누어 쓰는 변환기 덩이를 num_layers 번 되풀이
    """
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=768, nhead=12, num_layers=12):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)    # vocab -> E
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)        # E -> H

        self.shared_block = SharedTransformerBlock(d_model=hidden_dim, nhead=nhead)
        self.num_layers = num_layers

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # 낮은 차수 E에서의 낱말 담기
        x = self.token_embed(input_ids)     # (B, S, E)

        # 숨은 차수 H으로 되비춘다
        x = self.embed_proj(x)              # (B, S, H)

        # 덧대기 가림(True이면 셈에서 뺀다)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        # 같은 덩이를 여러 번 되쓴다(매개변수 나눠 쓰기)
        for _ in range(self.num_layers):
            x = self.shared_block(x, src_key_padding_mask=src_key_padding_mask)

        # MLM 로짓
        logits = self.lm_head(x)            # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = ALBERT(vocab_size=1000, embed_dim=64, hidden_dim=256, nhead=8, num_layers=4)
    ids = torch.randint(0, 1000, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    logits = model(ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 9, 1000)```

## 논의

이 짜보기는 갈래 2개(`SharedTransformerBlock`, `ALBERT`)를 매기고, 이들이 어울려 온전한 변환기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `SharedTransformerBlock`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

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
`SharedTransformerBlock`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = SharedTransformerBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
