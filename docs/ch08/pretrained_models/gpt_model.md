# GPT 모형

이 모듈은 트랜스포머 디코더에 글 생성 메서드를 씌워 GPT 방식의 언어 모형을 구현한다. 이 모형은 새 토큰마다 앞선 토큰 전체를 조건으로 한 확률 분포에서 뽑는 자기 회귀 생성 방식을 보여 준다. 이 구조가 GPT 계열 모형의 바탕이다.

## 1. 코드

```python
import torch
import torch.nn as nn
from transformer_decoder import TransformerDecoder


class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072):
        super().__init__()
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, num_layers, d_ff
        )

    def forward(self, x, mask=None):
        return self.decoder(x, mask)

    def generate(self, start_tokens, max_len=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(start_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                start_tokens = torch.cat([start_tokens, next_token], dim=1)
        return start_tokens


if __name__ == "__main__":
    pass
```

## 2. 논의

`GPTModel`은 `TransformerDecoder`를 얇게 감싸고 자기 회귀 글 생성을 위한 `generate` 메서드를 더한 것이다. 앞먹임은 그냥 디코더에 맡기며, 디코더가 토큰 임베딩, 위치 인코딩, 가린 자기 주의, 출력 사영을 다룬다. 이렇게 할 일을 갈라 두면 생성 논리를 그대로 둔 채 다른 디코더 구현으로 바꾸기 쉽다.

`generate` 메서드는 온도로 다스리는 표집을 구현한다. 단계마다 모형이 다음 토큰의 로짓을 셈하고 온도 매개변수의 역수를 곱한 뒤 소프트맥스로 확률로 바꾸고 `torch.multinomial`으로 그 분포에서 뽑는다. 온도가 낮으면 확률이 가장 그럴듯한 토큰에 몰려 더 결정론적이고 되풀이되는 글이 나온다. 온도가 높으면 분포가 평평해져 앞뒤 맞음을 대가로 다양함과 창의를 북돋운다.

눈여겨볼 점은 이 메서드가 `self.eval()`을 부르고 계산을 `torch.no_grad()`로 감싼다는 것이다. 평가 모드는 드롭아웃을 꺼서 예측을 한결같게 한다. 기울기 계산을 끄면 계산 그래프를 담아 둘 필요가 없어 기억이 덜 들고 추론이 빨라진다.

## 연습문제

**연습문제 1.**
`vocab_size=100`에서 `start_tokens = torch.tensor([[1, 2, 3]])`으로 시작해 토큰 20개짜리 수열을 만들어라. `temperature=0.5`, `1.0`, `2.0`으로 세 번 돌려 출력이 어떻게 다른지 살펴라.

??? success "연습문제 1 풀이"
    ```python
    model = GPTModel(vocab_size=100, d_model=128, num_heads=4,
                     num_layers=2, d_ff=256)
    start = torch.tensor([[1, 2, 3]])

    for temp in [0.5, 1.0, 2.0]:
        output = model.generate(start.clone(), max_len=20, temperature=temp)
        print(f"temp={temp}: {output[0].tolist()}")
    ```
    `temperature=0.5`이면 출력이 더 되풀이되고 결정론적이다. `temperature=2.0`이면 더 다양하지만 앞뒤가 덜 맞을 수 있다. 모형이 학습되지 않았으므로 출력은 모두 무작위이겠지만 돌릴 때마다의 흩어짐은 온도가 높을수록 커진다.

---

**연습문제 2.**
지금의 `generate` 메서드는 위치 인코딩이 정한 최대 수열 길이 제약을 다루지 않는다. `start_tokens`과 만들어진 토큰의 합이 디코더의 `max_len`을 넘으면 어떻게 되는지 설명하고 고칠 방법을 내놓아라.

??? success "연습문제 2 풀이"
    수열 전체 길이가 (디코더에서 기본값 5000인) `max_len`을 넘으면 위치 인코딩의 색인 `self.pos_encoding[:, :seq_len, :]`이 잡아 둔 버퍼 바깥의 자리에 닿으려 하여 색인 오류가 난다. 고치는 방법은 미끄러지는 창을 쓰는 것이다. 생성 단계마다 마지막 `max_len`개의 토큰만 앞먹임 메서드에 넘긴다.
    ```python
    def generate(self, start_tokens, max_len=50, temperature=1.0,
                 context_window=5000):
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                input_tokens = start_tokens[:, -context_window:]
                logits = self.forward(input_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                start_tokens = torch.cat([start_tokens, next_token], dim=1)
        return start_tokens
    ```

---

**연습문제 3.**
`generate` 메서드에서 순수 다항 표집 대신 쓸 수 있는 상위 $k$ 표집을 구현하라. 확률이 가장 높은 $k$개의 토큰만 뽑기 후보가 되어야 한다.

??? success "연습문제 3 풀이"
    ```python
    def generate_top_k(self, start_tokens, max_len=50, k=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(start_tokens)
                next_logits = logits[:, -1, :] / temperature
                top_k_values, top_k_indices = torch.topk(next_logits, k)
                probs = torch.softmax(top_k_values, dim=-1)
                sampled = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, sampled)
                start_tokens = torch.cat([start_tokens, next_token], dim=1)
        return start_tokens
    ```
    상위 $k$ 표집은 후보를 확률이 가장 높은 $k$개의 토큰으로 제한하여, 모형이 생성을 어그러뜨릴 만큼 그럴듯하지 않은 토큰을 고르지 못하게 한다. 다양함과 질 사이의 균형을 준다.

## 정리하며

**다룬 것** — GPT 모형

`GPTModel`은 `TransformerDecoder`를 얇게 감싸고 자기 회귀 글 생성을 위한 `generate` 메서드를 더한 것이다.

핵심 클래스는 `GPTModel`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
