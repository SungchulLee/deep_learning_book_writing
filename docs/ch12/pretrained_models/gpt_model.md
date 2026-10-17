# GPT 모델

사용자 정의 TransformerDecoder에 토큰 샘플링 루프를 연결한 예제다. 사전 학습 GPT 웨이트를 불러오지 않으며, 기반 디코더의 축·마스크 문제도 이어진다. 생성 절차를 설명하는 코드와 올바르게 학습된 언어 모델을 구분한다.

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

generate는 전체 토큰을 매 단계 다시 모델에 전달하고 마지막 위치 로짓에 온도 T를 적용해 softmax(logits/T)에서 표본을 뽑는다. T는 양수여야 한다. 고정 로짓에서는 T가 낮을수록 최댓값에 집중하지만 생성 문장의 창의성·정확성이나 반복률을 단일 값으로 보장하지는 않는다.

기반 TransformerDecoder는 배치 우선 입력을 시퀀스 우선 어텐션에 그대로 넘기며 인과 마스크도 자동 생성하지 않는다. GPTModel은 mask=None으로 호출한다. 특히 배치 1에서는 내부가 길이 1의 여러 배치로 해석해 의도한 토큰 간 어텐션을 계산하지 못한다. 샘플링 루프를 추가하는 것만으로 이 문제가 해결되지는 않는다.

self.eval은 드롭아웃을 끄고 no_grad는 경사 기록을 끈다. 샘플링 자체의 무작위성은 남는다. 메서드는 호출 전 학습 모드를 복원하지 않으며 EOS 종료·패딩 처리·KV 캐시가 없다. max_len은 전체 길이가 아니라 추가 생성 횟수이므로 프롬프트 길이도 따로 계산해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 코드의 생성 길이와 온도 비교가 보여 주는 범위를 설명하라.

</div>

??? success "풀이"
    ```python
    model = GPTModel(vocab_size=100, d_model=128, num_heads=4,
                     num_layers=2, d_ff=256)
    start = torch.tensor([[1, 2, 3]])

    for temp in [0.5, 1.0, 2.0]:
        output = model.generate(start.clone(), max_len=20, temperature=temp)
        print(f"temp={temp}: {output[0].tolist()}")
    ```
    max_len=20은 새 토큰 20개를 뜻해 길이 3의 프롬프트와 합쳐 23개를 반환한다. 고정 로짓에서 온도는 분포 집중도를 바꾸지만, 여기서는 모델이 미학습 상태이고 기반 어텐션도 잘못 연결되어 의미 있는 문장 품질 비교가 아니다. 독립된 한 번씩의 표본만으로 출력 분산을 판단할 수도 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
위치 표 길이를 넘을 때의 오류와 아래 슬라이딩 컨텍스트 조각의 한계를 설명하라.

</div>

??? success "풀이"
    저장된 표보다 입력이 길면 슬라이스는 표의 끝에서 멈추고, 입력 임베딩과 더할 때 위치 축이 달라 모양 오류가 난다. 범위 밖 위치를 직접 인덱싱하는 오류와 다르다. 아래 방식은 최근 context_window개만 넘겨 길이를 제한하지만 오래된 정보를 버리고 위치를 매번 0부터 다시 배정한다. 원래 긴 문맥 계산과 동등하지 않으며 디코더의 축·마스크 문제도 별도로 해결해야 한다.
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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 top-k 샘플링의 후보 선택·재정규화와 입력 조건을 설명하라.

</div>

??? success "풀이"
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
    상위 k개 로짓만 골라 소프트맥스한 뒤 뽑은 후보 인덱스를 원래 어휘 ID로 되돌린다. 1≤k≤어휘 크기와 temperature>0을 확인해야 한다. 낮은 확률 토큰을 제외할 뿐, 잘못된 내용이나 비문이 생성되지 않는다는 보장은 없다. EOS 종료와 길이·마스크 처리는 별도로 필요하다.

## 정리하며

토큰 샘플링 루프는 사전 학습과 올바른 인과 디코더를 전제로 한다. 이 예제의 기반 축·마스크 한계, 추가 토큰 수의 의미와 컨텍스트 절단을 확인하고 온도·top-k의 효과를 품질 보장으로 해석하지 않는다.
