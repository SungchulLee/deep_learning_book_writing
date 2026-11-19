# ALBERT

ALBERT은 2019년 논문 "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"에서 나왔다. 1) 분해된 임베딩 파라메터화(factorized embedding parameterization): 단어 임베딩 차원(E)을 은닉 차원(H)보다 작게 두고 단어 -> E으로 임베딩한 뒤 E -> H으로 사영한다 2) 레이어 간 파라메터 공유: 같은 트랜스포머 레이어의 웨이트를 모든 레이어에서 재사용한다 3) 문장 순서 예측(SOP)(흔히 NSP과 비교하여 이야기된다).

여기 구현은 ALBERT을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
ALBERT - 가벼운 BERT
논문: "ALBERT: 언어 표현의 자기 지도 학습을 위한 가벼운 BERT" (2019)
지은이: 전중 란 외
핵심 아이디어:
  1) 분해된 임베딩 파라메터화:
       단어 임베딩 차원(E)을 은닉 차원(H)보다 작게 둔다
       임베딩: 단어 -> E, 그다음 E -> H으로 사영한다
  2) 레이어 간 파라메터 공유:
       같은 트랜스포머 레이어의 웨이트를 모든 레이어에서 재사용한다
  3) 문장 순서 예측(SOP)(흔히 NSP과 견주어 이야기된다)

파일: appendix/transformers/albert.py
참고: 분해된 임베딩과 공유 레이어를 보이는, 교육용 구현이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class SharedTransformerBlock(nn.Module):
    """여러 번 재사용하려고 만든 트랜스포머 인코더 레이어 하나."""
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        return self.layer(x, src_key_padding_mask=src_key_padding_mask)


class ALBERT(nn.Module):
    """
    다음을 갖춘 인코더 전용 ALBERT 모델:
      - 분해된 임베딩
      - 공유 트랜스포머 블록을 num_layers 번 반복
    """
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=768, nhead=12, num_layers=12):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)    # vocab -> E
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)        # E -> H

        self.shared_block = SharedTransformerBlock(d_model=hidden_dim, nhead=nhead)
        self.num_layers = num_layers

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # 낮은 차원 E에서의 단어 임베딩
        x = self.token_embed(input_ids)     # (B, S, E)

        # 은닉 차원 H으로 사영한다
        x = self.embed_proj(x)              # (B, S, H)

        # 패딩 마스크(True이면 계산에서 제외한다)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        # 같은 블록을 여러 번 재사용한다(파라메터 공유)
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
    print("logits:", logits.shape)  # (2, 9, 1000)
```

**출력:**

```
logits: torch.Size([2, 9, 1000])
```

## 2. 논의

이 구현은 클래스 2개(`SharedTransformerBlock`, `ALBERT`)를 정의하고, 이들이 함께 완전한 트랜스포머 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `SharedTransformerBlock`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 웨이트 뒤(값과 곱하기 앞)에 드롭아웃 레이어를 더하여라. 학습하는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 학습하는 동안 어텐션 웨이트의 일부를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에만 의존하는 것을 막는다. 그래서 모델이 어텐션을 더 고르게 분산하고 더 강건한 표현을 학습하게 되는데, 이는 표준 드롭아웃이 뉴런들의 공적응(co-adaptation)을 막는 것과 같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 구하라. 이것이 긴 시퀀스에서 Longformer이나 Linformer 같은 구조가 나온 동기가 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    표준 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$은 창 크기)과 선택한 토큰에 대한 희소 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`SharedTransformerBlock`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = SharedTransformerBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — ALBERT

이 구현은 클래스 2개(`SharedTransformerBlock`, `ALBERT`)를 정의하고, 이들이 함께 완전한 트랜스포머 구조를 이룬다.

핵심 클래스는 `SharedTransformerBlock`, `ALBERT`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
