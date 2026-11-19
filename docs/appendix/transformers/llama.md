# LLaMA

LLaMA는 2023년 논문 "LLaMA: Open and Efficient Foundation Language Models"에서 나왔다.

이 쪽의 구현은 LLaMA의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
LLaMA - 메타 AI의 대형 언어 모델
논문: "LLaMA: Open and Efficient Foundation Language Models" (2023)
저자: 메타 AI
핵심 아이디어(개략):
  - 디코더만 있는 트랜스포머(GPT 계열)
  - 레이어 정규화 대신 RMSNorm
  - SwiGLU 피드포워드
  - 학습되는 절대 위치 임베딩 대신 회전 위치 임베딩(RoPE)

파일: appendix/transformers/llama.py
참고: RMSNorm + SwiGLU + 인과 어텐션에 집중한, 주석을 단 교육용 구현이다.
      최적화된 LLaMA가 아니라 읽기 쉬운 예시다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class RMSNorm(nn.Module):
    """
    RMSNorm: 제곱 평균의 제곱근으로 정규화한다(평균을 빼지 않는다).

    x_norm = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU 피드포워드(최근 대형 언어 모델에서 널리 쓴다):

      FF(x) = (SiLU(xW1) * (xW3)) W2

    표준 GELU 피드포워드와 견주면 이 게이트 구조가 성능과 효율을 흔히 높인다.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """
    인과(마스크드) 셀프 어텐션(멀티헤드).
    단순하게 하려고 RoPE 계산은 빼고 표준 인과 마스크를 쓴다.
    """
    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        qkv = self.qkv(x)                 # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)    # each: (B, S, D)

        # 헤드 형태로 바꾼다: (B, nhead, S, head_dim)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # 어텐션 점수: (B, nhead, S, S)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 인과 마스크: 뒤에 올 토큰에 어텐션을 주지 못하게 한다
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, nhead, S, head_dim)

        # 헤드를 다시 합친다: (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class LLaMABlock(nn.Module):
    """
    LLaMA 방식의 트랜스포머 블록 하나(단순화함):
      - RMSNorm
      - 인과 셀프 어텐션
      - RMSNorm
      - SwiGLU 피드포워드
      - 잔차 연결
    """
    def __init__(self, dim: int, nhead: int, ff_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, nhead)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LLaMA(nn.Module):
    """
    디코더만 있는 언어 모델(GPT 계열).

    입력:
      input_ids: (B, S)
    출력:
      logits: (B, S, vocab_size)
    """
    def __init__(self, vocab_size=32000, dim=512, nhead=8, num_layers=8, ff_hidden=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([LLaMABlock(dim, nhead, ff_hidden) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)   # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = LLaMA(vocab_size=1000, dim=256, nhead=8, num_layers=2, ff_hidden=1024)
    ids = torch.randint(0, 1000, (2, 12))
    logits = model(ids)
    print("logits:", logits.shape)  # (2, 12, 1000)
```

**출력:**

```
logits: torch.Size([2, 12, 1000])
```

## 2. 논의

이 구현은 클래스 5개(`RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock` 외 1개)를 정의하고, 이들이 함께 완전한 트랜스포머 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `RMSNorm`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

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
`RMSNorm`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — LLaMA

이 구현은 클래스 5개(`RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock` 외 1개)를 정의하고, 이들이 함께 완전한 트랜스포머 구조를 이룬다.

핵심 클래스는 `RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
