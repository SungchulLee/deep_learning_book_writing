# GPT

GPT(생성 사전 학습 트랜스포머)는 2018년 논문 "Improving Language Understanding by Generative Pre-Training"에서 나왔다. BERT의 양방향 방식과 달리 GPT는 자기 회귀 언어 모형화를 위해 한 방향(왼쪽에서 오른쪽)의 트랜스포머 디코더를 쓴다. 이 설계 덕분에 GPT는 한 번에 토큰 하나를 맞혀 앞뒤가 맞는 글을 지을 수 있고, 그것이 GPT 계열 모형의 바탕이 되었다.

## 1. 코드

```python
import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_mask = torch.triu(
            torch.ones(x.size(1), x.size(1)), diagonal=1
        ).bool().to(x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12,
                 n_heads=12, max_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.Sequential(
            *[GPTBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)

        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


if __name__ == "__main__":
    model = GPT()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 163,037,184
```

## 2. 논의

GPT 블록마다 위 삼각 가림으로 인과 주의를 지켜, 토큰이 앞으로의 자리에 주의하지 못하게 한다. 이것이 BERT와의 핵심 차이이다. BERT는 입력 전체를 한꺼번에 보지만 GPT는 이미 만들어진 토큰만 볼 수 있다. 이 블록은 본디 GPT가 쓴 뒤 정규화 관례에 따라 아래 층(주의와 다층 퍼셉트론)마다 그 뒤에 층 정규화를 한다.

블록 안의 다층 퍼셉트론은 숨은 차원을 네 배로 넓히고 ReLU보다 매끄러운 비선형을 주는 GELU 활성 함수를 적용한다. 이 넓혔다 줄이는 방식 덕분에 신경망이 모형 차원으로 되사영하기 전에 더 넉넉한 중간 표현을 배울 수 있다.

마지막 선형 층("언어 모형 머리")은 모형 차원에서 어휘 크기로 되사영하여 다음 토큰 예측을 위한 로짓을 낸다. 눈여겨볼 점은 이 머리에 편향 항이 없고, 많은 구현에서 토큰 임베딩 행렬과 가중치를 함께 쓴다는 것이다. 매개변수 수를 줄이고 일반화를 낫게 할 수 있는 가중치 묶기라는 기법이다.

## 연습문제

**연습문제 1.**
기본 매개변수로 GPT 모형을 만들고 무작위 토큰 번호의 배치를 넣어 보아라. 출력의 꼴이 `(batch_size, sequence_length, vocab_size)`인지, 그리고 인과 가림이 앞으로의 토큰에서 정보가 새는 것을 막는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    model = GPT()
    x = torch.randint(0, 50257, (2, 64))  # batch=2, seq_len=64
    logits = model(x)
    print(f"Output shape: {logits.shape}")  # (2, 64, 50257)
    ```
    출력의 꼴은 `(2, 64, 50257)`으로, 자리마다 어휘 전체에 대한 분포를 냄을 확인해 준다. 인과 가림은 자리 $i$이 자리 $0, 1, \ldots, i$에만 주의하게 하여 앞으로의 정보가 새지 않게 한다.

---

**연습문제 2.**
`d_model=768`, `n_layers=12`로 같게 두고 GPT와 BERT의 매개변수 수를 견주어라. 차이가 어디서 오는지 짚고 어떤 구조 선택 때문인지 설명하라.

??? success "연습문제 2 풀이"
    BERT는 어휘가 토큰 3만 개인데 GPT는 5만 257개이므로 GPT에서 토큰 임베딩의 매개변수가 훨씬 많다. BERT에는 구간 임베딩(3종류)과 다음 문장 맞히기 머리도 있는데 GPT에는 없다. GPT의 언어 모형 머리는 `(768, 50257)`짜리 큰 선형 층이다. 이런 차이에도 트랜스포머 블록 자체는 둘 다 $4 \times d_{\text{model}}$으로 넓히는 같은 주의·순전파 짜임을 쓰므로 매개변수 수가 비슷하다.

---

**연습문제 3.**
프롬프트 텐서를 받아 탐욕 디코딩(단계마다 argmax)으로 토큰 `max_new_tokens`개를 자기 회귀로 더 만들어 내는 간단한 `generate` 메서드를 `GPT` 클래스에 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def generate(self, prompt, max_new_tokens=50):
        self.eval()
        tokens = prompt.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(tokens[:, -1024:])  # max_len으로 자른다
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
    ```
    단계마다 마지막 자리의 로짓만 쓴다. 위치 임베딩의 한계를 지키려고 수열을 `max_len=1024`으로 자른다. 탐욕 디코딩은 언제나 확률이 가장 높은 토큰을 고르는데, 결정론적이지만 되풀이되는 글이 나올 수 있다.

## 정리하며

**다룬 것** — GPT

GPT 블록마다 위 삼각 가림으로 인과 주의를 지켜, 토큰이 앞으로의 자리에 주의하지 못하게 한다.

핵심 클래스는 `GPTBlock`, `GPT`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
