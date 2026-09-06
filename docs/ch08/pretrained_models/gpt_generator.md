# GPT 생성기

GPT 글 생성기는 트랜스포머 디코더 구조로 하는 자기 회귀 언어 생성을 보여 준다. 인과 가림으로 앞으로의 토큰에 주의하지 못하게 하여, 모형은 한 번에 토큰 하나씩 글을 지으며 새 토큰마다 앞서 만든 토큰 전체를 조건으로 삼는다. 이 방식이 요즘 대형 언어 모형의 바탕에 있다.

## 코드

```python
import torch
import torch.nn as nn


class GPTGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        # 인과 가림을 만든다
        mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool().to(x.device)

        x = x.transpose(0, 1)
        x = self.transformer(x, x, tgt_mask=mask)
        x = x.transpose(0, 1)

        return self.fc_out(x)


if __name__ == "__main__":
    pass
```

## 논의

`GPTGenerator`는 가린 자기 주의와 교차 주의 아래 층을 모두 담은 파이토치의 `nn.TransformerDecoder`를 쓴다. 이 구현에서는 같은 수열을 목표와 기억 입력으로 함께 넣어 사실상 교차 주의 층이 둘째 자기 주의처럼 움직이게 한다. 인코더-디코더 API를 디코더 전용 모형에 맞출 때 흔한 방식이다.

인과 가림은 위 삼각 참거짓 행렬이며 `True` 값이 앞으로의 자리에 대한 주의를 막는다. 자리 $i$의 토큰이 주의 점수를 셈할 때 $j > i$인 자리는 모두 가려져, 모형이 지난 토큰과 지금 토큰만 조건으로 삼게 한다. 앞으로의 맥락을 보지 않고 다음 토큰을 맞혀야 하는 자기 회귀 생성에 꼭 필요하다.

학습되는 위치 인코딩은 무작위로 초기화되고 최대 수열 길이 1024의 매개변수로 담긴다. 사인파 인코딩과 달리 이 학습되는 자리는 모형이 학습 중에 어떤 자리 무늬든 찾아내게 해 준다. 마지막 선형 층은 디코더의 출력을 어휘 크기로 되사영하여 소프트맥스로 토큰 확률로 바꿀 수 있는 로짓을 낸다.

## 연습문제

**연습문제 1.**
`vocab_size=5000`으로 `GPTGenerator`를 만들고 길이 64인 수열을 넣어 보아라. 출력의 꼴이 `(batch_size, 64, 5000)`인지, 인과 가림이 올바른 삼각 짜임을 갖추었는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    model = GPTGenerator(vocab_size=5000)
    x = torch.randint(0, 5000, (2, 64))
    output = model(x)
    print(f"Output shape: {output.shape}")  # (2, 64, 5000)

    # 인과 가림을 확인한다
    mask = torch.triu(torch.ones(64, 64), diagonal=1).bool()
    print(f"Mask shape: {mask.shape}")       # (64, 64)
    print(f"Mask[0, 0] = {mask[0, 0]}")     # False (주의할 수 있다)
    print(f"Mask[0, 1] = {mask[0, 1]}")     # True (막혔다)
    ```

---

**연습문제 2.**
GPT 방식의 모형을 세울 때 (이 코드처럼) `nn.TransformerDecoder`를 쓰는 것과 인과 가림을 곁들인 `nn.TransformerEncoder`를 쓰는 것의 차이를 설명하라. 구조와 실제 면에서 어떤 뜻이 있는가?

??? success "연습문제 2 풀이"
    `nn.TransformerDecoder`는 자기 주의와 순전파 신경망 사이에 교차 주의 아래 층을 두는데, 수열 대 수열 과제에서 인코더 출력에 주의하도록 설계된 것이다. 디코더만 쓰는 모형에 (같은 입력을 목표와 기억으로 함께 넣어) 쓰면 교차 주의가 군더더기 자기 주의가 되어 쓸데없는 매개변수와 계산이 는다. 인과 가림을 곁들인 `nn.TransformerEncoder`는 자기 주의와 순전파 아래 층만 담으므로 GPT 방식 모형에 더 효율적이다. 인코더 방식이 더 간단하고 빨라서 실제 GPT 구현 대부분이 인과 가림을 쓰는 인코더 방식 블록을 쓴다.

---

**연습문제 3.**
온도로 다스리는 표집을 받치는 `generate` 메서드를 구현하라. 프롬프트 텐서가 주어지면 로짓에 온도 크기 조정을 적용한 뒤 표집하여 토큰을 자기 회귀로 만들어야 한다.

??? success "연습문제 3 풀이"
    ```python
    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        self.eval()
        tokens = prompt.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(tokens[:, -1024:])
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
    ```
    온도가 1.0보다 작으면 분포가 뾰족해지고(더 결정론적이고) 1.0보다 크면 평평해진다(더 무작위이다). 온도가 1.0이면 본디 확률 분포가 그대로이다.
