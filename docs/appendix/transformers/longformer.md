# Longformer

Longformer는 2020년 논문 "Longformer: The Long-Document Transformer"에서 나왔다. 모든 토큰 쌍을 보는 O(S^2) 어텐션을 두 가지로 대체한다: (a) 슬라이딩 윈도 국소 어텐션(O(S * 윈도)) — 각 토큰이 가까운 이웃만 본다 — 과 (b) 선택한 토큰에만 주는 전역 어텐션(예: [CLS], 질문 토큰).

이 쪽의 구현은 Longformer의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
Longformer - 긴 문서를 위한 트랜스포머
논문: "Longformer: The Long-Document Transformer" (2020)
저자: 이즈 벨타기, 매슈 피터스, 아르만 코한
핵심 아이디어:
  - 완전한 O(S^2) 어텐션을 다음으로 대체한다:
      (a) 슬라이딩 윈도 국소 어텐션 (O(S * window))
      (b) 선택적 전역 어텐션 토큰(예: [CLS], 질문 토큰)

파일: appendix/transformers/longformer.py
참고: *국소 어텐션*(윈도를 쓰는 셀프 어텐션)을 보이는 교육용 구현이다.
      최적화된 커널이 아니라 명료하게 보이려는 예시다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class WindowSelfAttention(nn.Module):
    """
    단순하게 구현한 윈도 셀프 어텐션(이해하기 쉽도록 헤드 하나).

    위치 i마다 [i-w, i+w] 안의 토큰에만 어텐션을 준다.
    복잡도가 O(S^2)이 아니라 ~O(S * window)이 된다.
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
            # 국소 윈도의 인덱스 범위를 정한다
            left = max(0, i - self.window)
            right = min(S, i + self.window + 1)

            # 토큰 i와 국소 윈도의 토큰 사이 어텐션을 계산한다
            q_i = Q[:, i : i + 1, :]            # (B, 1, D)
            k_w = K[:, left:right, :]           # (B, W, D)
            v_w = V[:, left:right, :]           # (B, W, D)

            # 스케일 내적 어텐션
            scores = (q_i @ k_w.transpose(1, 2)) / (D ** 0.5)  # (B, 1, W)
            attn = F.softmax(scores, dim=-1)                   # (B, 1, W)
            out_i = attn @ v_w                                 # (B, 1, D)
            outputs.append(out_i)

        y = torch.cat(outputs, dim=1)  # (B, S, D)
        return self.out(y)


class LongformerBlock(nn.Module):
    """윈도 어텐션 + 피드포워드 네트워크를 쓰는 트랜스포머 형태 블록 하나."""
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
        # 어텐션 + 잔차 연결 + 정규화
        x = self.norm1(x + self.attn(x))

        # 피드포워드 + 잔차 연결 + 정규화
        x = self.norm2(x + self.ff(x))
        return x


class Longformer(nn.Module):
    """
    윈도 어텐션 블록을 쓰는 Longformer 형태 인코더.
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
    print("logits:", logits.shape)  # (2, 20, 1000)
```

**출력:**

```
logits: torch.Size([2, 20, 1000])
```

## 2. 논의

이 구현은 클래스 3개(`WindowSelfAttention`, `LongformerBlock`, `Longformer`)를 정의하고, 이들이 함께 완전한 트랜스포머 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `WindowSelfAttention`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
`WindowSelfAttention`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = WindowSelfAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Longformer

이 구현은 클래스 3개(`WindowSelfAttention`, `LongformerBlock`, `Longformer`)를 정의하고, 이들이 함께 완전한 트랜스포머 구조를 이룬다.

핵심 클래스는 `WindowSelfAttention`, `LongformerBlock`, `Longformer`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
