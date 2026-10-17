# GPT

GPT 계열은 인과적으로 앞선 입력을 읽어 다음 토큰을 예측한다. 아래는 학습 가능한 위치 임베딩과 인과 셀프 어텐션을 연결한 간단한 모델이며 특정 사전 학습 GPT 체크포인트를 재현하지는 않는다.

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

GPTBlock은 batch_first=True와 상삼각 True=차단 마스크로 각 표본의 미래 키 접근을 막는다. 블록은 Post-LN이며 GELU 피드포워드의 중간 폭은 4d다. GPT 계열의 모든 변형이 이 정규화 위치와 세부 설정을 쓰는 것은 아니다.

토큰·위치 임베딩을 더하고 블록·마지막 LayerNorm·출력 선형 레이어를 적용한다. 위치 테이블은 최대 1024개이고 별도의 패딩 마스크는 없다. 반환값은 $(B,L,V)$의 로짓이며 확률로 읽으려면 소프트맥스가 필요하다. 다음 토큰 학습에는 입력·목표 시프트와 PAD 로스 제외를 별도로 구성한다.

head는 편향이 없지만 token_emb와 웨이트를 공유하지 않는다. 따라서 기본 어휘 50,257개에서 입력 임베딩과 출력 레이어가 각각 38,597,376개를 사용한다. 코드에 없는 웨이트 공유의 계수 절감 효과를 현재 출력 수에 적용하면 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 모양 검사 외에 전체 모델의 인과성을 어떻게 검토할 수 있는가?

</div>

??? success "풀이"
    ```python
    model = GPT()
    x = torch.randint(0, 50257, (2, 64))  # batch=2, seq_len=64
    logits = model(x)
    print(f"Output shape: {logits.shape}")  # (2, 64, 50257)
    ```
    출력은 (2,64,50257)의 로짓이다. eval 모드에서 접두는 같게 두고 뒤쪽 토큰만 바꾼 두 입력의 접두 위치 로짓을 비교하는 검사가 필요하다. 아래 코드는 모양만 출력하므로 인과성 검증을 실제 수행한 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 페이지 GPT와 08_bert.md의 모델을 같은 d=768, 12레이어에서 비교하라.

</div>

??? success "풀이"
    GPT는 163,037,184개, 단순 BERT는 131,563,058개다. 둘 다 입력·출력 어휘 웨이트가 별도이므로 어휘 수 차이가 두 행렬에 영향을 준다. 위치 테이블 길이, 구간 임베딩, NSP 머리, 최종 정규화도 다르다. 단순한 구조 이름만으로 계수 차이를 설명하거나 같은 예산의 성능 비교로 해석해서는 안 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 탐욕 생성 함수의 컨텍스트 길이 처리와 생성 종료의 한계를 설명하라.

</div>

??? success "풀이"
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
    max_new_tokens만큼 새 토큰을 추가하며 매번 최근 1024개만 입력한다. 이는 기본 위치 표에는 맞지만 생성자 max_len을 다르게 주면 하드코딩된 길이와 맞지 않을 수 있다. 오래된 문맥은 버리고 위치를 다시 0부터 배정한다. EOS 종료나 KV 캐시는 없으며, 함수 조각을 실제 클래스에 연결하는 작업도 필요하다.

## 정리하며

인과 마스크·토큰 축·목표 시프트를 연결해 자기 회귀 모델을 이해한다. 이 예제는 Post-LN과 비공유 어휘 웨이트를 사용하며, 생성 시 고정 길이 절단은 긴 문맥을 그대로 계산하는 것과 다르다.
