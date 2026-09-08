# LLaMA

LLaMA은 2023년 글 "LLaMA: Open and Efficient Foundation Language Models"에서 나왔다.

여기 짜보기는 LLaMA을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
LLaMA - 메타 AI의 큰 말 모형
글: "LLaMA: 열려 있고 잘 드는 밑바탕 말 모형" (2023)
지은이: 메타 AI
고갱이 깨침(크게 보아):
  - 풀개만 있는 변환기(GPT 결)
  - 켜 잣대 잡기 대신 RMSNorm
  - SwiGLU 앞먹임
  - 배운 붙박이 자리 대신 도는 자리 담기(RoPE)

두루마리: appendix/transformers/llama.py
눈여겨볼 것: RMSNorm + SwiGLU + 앞만 보는 눈길에 마음을 둔, 주석을 단 배우기용 짜보기다.
      잘 다듬은 LLaMA이 아니라 읽기 쉬운 본이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class RMSNorm(nn.Module):
    """
    RMSNorm: 제곱 평균의 제곱근으로 잣대를 맞춘다(평균을 빼지 않는다).

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
    SwiGLU 앞먹임(요즘 큰 말 모형에서 널리 쓴다):

      FF(x) = (SiLU(xW1) * (xW3)) W2

    여느 GELU 앞먹임과 견주면 이 문 얼개가 됨됨이와 잘 듦을 흔히 올린다.
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
    앞만 보는(가린) 스스로 눈길(여러 머리).
    쉽게 하려고 RoPE 셈은 빼고 여느 앞만 보는 가림을 쓴다.
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

        # 머리 꼴로 바꾼다: (B, nhead, S, head_dim)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # 눈길 점수: (B, nhead, S, S)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 앞만 보는 가림: 뒤에 올 낱말에 눈길을 주지 못하게 한다
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, nhead, S, head_dim)

        # 머리를 다시 합친다: (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class LLaMABlock(nn.Module):
    """
    LLaMA 결의 변환기 덩이 하나(단순하게 만듦):
      - RMSNorm
      - 앞만 보는 스스로 눈길
      - RMSNorm
      - SwiGLU 앞먹임
      - 나머지 이음
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
    풀개만 있는 말 모형(GPT 결).

    들임:
      input_ids: (B, S)
    날임:
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

이 짜보기는 갈래 5개(`RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock`, and 1 more)를 매기고, 이들이 어울려 온전한 변환기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
기본 첫자리로 잡은 `RMSNorm`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**연습문제 2.**
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율 0.1을 써라. 눈길 드롭아웃이 다독임에 왜 도움이 되는지 밝혀라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 얼마를 아무렇게나 0으로 만들어, 모형이 낱말끼리의 어떤 사이에만 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 드러냄을 배우게 되는데, 이는 여느 드롭아웃이 신경 낱자리끼리 함께 길드는 것을 막는 것과 같다.

---

**연습문제 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**연습문제 4.**
`RMSNorm`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — LLaMA

이 짜보기는 갈래 5개(`RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock`, and 1 more)를 매기고, 이들이 어울려 온전한 변환기 얼개를 이룬다.

고갱이 갈래는 `RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
