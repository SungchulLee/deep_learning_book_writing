# DistilBERT

DistilBERT는 2019년 논문 "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"에서 나왔다. 이 모델은 **지식 증류**(knowledge distillation)로 만든다 — 더 큰 교사 모델의 출력을 따라 하도록 더 작은 학생 트랜스포머를 트레인하는 것이다. 대개 레이어 수는 적고 은닉 크기는 비슷하다.

이 쪽의 구현은 DistilBERT의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DistilBERT - 지식 증류로 줄인 BERT
논문: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (2019)
저자: 빅토르 상 외
핵심 아이디어:
  - 지식 증류: 더 큰 교사 모델을 따라 하도록 더 작은 학생 트랜스포머를 트레인한다
  - 대개 레이어 수는 적고 은닉 크기는 비슷하다

파일: appendix/transformers/distilbert.py
참고: 인코더만 있는 더 작은 트랜스포머를 보이는 교육용 구현이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class DistilBERT(nn.Module):
    """
    인코더만 있는 DistilBERT 형태 트랜스포머.

    BERT나 RoBERTa와 비슷하되 인코더 레이어가 더 적다.
    지식 증류는 트레인 과정에서 일어난다(여기서는 구현하지 않는다).
    """
    def __init__(self, vocab_size=30522, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(h)
        return logits


if __name__ == "__main__":
    model = DistilBERT(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
    ids = torch.randint(0, 1000, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    logits = model(ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 10, 1000)
```

**출력:**

```
logits: torch.Size([2, 10, 1000])
```

## 2. 논의

`DistilBERT` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다. `forward` 메서드가 계산 그래프를 정의하므로 트레인하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이렇게 모듈로 나눈 설계 덕에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `DistilBERT`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
`DistilBERT`를 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = DistilBERT(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — DistilBERT

`DistilBERT` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다.

핵심 클래스는 `DistilBERT`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
