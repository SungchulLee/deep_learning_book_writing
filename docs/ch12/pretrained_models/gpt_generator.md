# GPT 생성기

nn.TransformerDecoder를 이용해 토큰 로짓을 만드는 예제다. 그러나 같은 입력을 마스킹하지 않은 교차 어텐션 메모리로도 전달하므로 미래 토큰이 유출된다. 표준 GPT처럼 인과 셀프 어텐션만 사용하는 디코더 전용 구현과 다르다.

## 1. 코드

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

## 2. 논의

입력 $(B,L)$에 토큰 임베딩과 학습 위치 표를 더하고 $(L,B,E)$로 전치한다. 이 축 변환은 PyTorch 기본 규약에 맞다. 반환 로짓은 $(B,L,V)$이며 위치 표 길이 1024 이내의 입력을 가정한다. 모델은 무작위로 초기화된다.

**미래 정보 유출:** self.transformer(x,x,tgt_mask=mask)는 목표 셀프 어텐션에만 인과 마스크를 적용한다. memory로 전달한 두 번째 x를 읽는 교차 어텐션에는 memory_mask가 없다. 따라서 앞 위치가 미래 입력 토큰을 읽을 수 있다. 상삼각 tgt_mask를 검사하는 것만으로 전체 모델의 인과성을 검증할 수 없다.

디코더 전용 모델에는 보통 셀프 어텐션과 피드포워드가 필요하며, 원문 메모리를 읽는 교차 어텐션은 별도의 선택이다. 같은 입력을 두 경로에 넣는 것을 일반적인 안전한 GPT 구현법으로 제시해서는 안 된다. 이 예제에 단순히 생성 메서드를 추가해도 학습 단계의 유출 문제는 해결되지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 출력·삼각 마스크 검사가 전체 모델의 인과성을 보장하지 않는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    출력 모양은 (2,64,5000)이지만 마스크 검사는 목표 셀프 어텐션만 다룬다. 미래 토큰을 바꿨을 때 과거 위치 로짓이 그대로인지도 확인해야 한다. 현재 구현은 교차 어텐션 메모리에 미래 입력이 남아 있어 그 조건을 만족하는 인과 모델이 아니다.

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
디코더 전용 블록과 nn.TransformerDecoder의 교차 어텐션을 구분하라.

</div>

??? success "연습문제 2 풀이"
    nn.TransformerDecoderLayer에는 목표 셀프 어텐션 외에 외부 메모리 교차 어텐션이 있다. 이를 인과 모델에 쓰려면 모든 정보 경로의 마스킹을 맞춰야 한다. 인과 마스크를 쓰는 셀프 어텐션·FFN 블록은 교차 어텐션 없이 디코더 전용 모델을 구성할 수 있다. 클래스 이름보다 실제 연결과 마스크를 기준으로 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 샘플링 메서드의 길이·모드·온도와 여전히 남는 모델 한계를 설명하라.

</div>

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
    temperature는 양수여야 하며 낮을수록 고정 로짓의 분포가 더 집중된다. 이 함수는 eval과 no_grad를 사용하지만 EOS에서 멈추지 않고 매번 전체 최근 1024개 토큰을 재계산한다. 이 절단은 오래된 정보를 버리고 위치를 다시 0부터 배정한다. 교차 어텐션의 미래 유출 문제는 학습 모델 쪽에서 별도로 해결해야 한다. 코드 조각은 클래스 메서드로 연결하는 작업도 필요하다.

## 정리하며

목표 셀프 어텐션 마스크만으로 전체 모델의 인과성이 보장되지는 않는다. 이 구현은 같은 입력을 무마스크 메모리로 읽어 미래 토큰을 볼 수 있다. 사전 학습 GPT·인과 디코더와 구분해 구조와 샘플링 절차를 검토한다.
