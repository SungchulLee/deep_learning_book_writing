# BERT

BERT는 양방향 인코더에서 마스크 언어 모델링(MLM)과 다음 문장 예측(NSP)을 학습한 모델이다. 아래 코드는 그 구성 요소를 단순화한 무작위 초기화 예제이며 사전 학습 웨이트나 학습 루프를 제공하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size=30000, d_model=768, max_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
        self.segment = nn.Embedding(3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, segment_label):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        embedding = self.token(x) + self.position(pos) + self.segment(segment_label)
        return self.norm(embedding)


class BERT(nn.Module):
    def __init__(self, vocab_size=30000, d_model=768, n_layers=12, heads=12):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.fc = nn.Linear(d_model, vocab_size)
        self.nsp = nn.Linear(d_model, 2)

    def forward(self, x, segment_label):
        x = self.embedding(x, segment_label)
        x = self.transformer(x)
        return self.fc(x), self.nsp(x[:, 0])


if __name__ == "__main__":
    model = BERT()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 131,563,058
```

## 2. 논의

BERTEmbedding은 토큰·절대 위치·구간 임베딩을 더하고 LayerNorm을 적용한다. 토큰 입력과 segment_label은 $(B,L)$의 정수 ID이고 길이는 512 이하여야 한다. 이 예제의 구간 테이블은 세 행이지만 원래 BERT의 기본 구간 종류는 두 개다.

nn.TransformerEncoder는 batch_first=True로 연결되어 각 표본의 토큰을 처리한다. 다만 패딩 마스크를 받지 않아 PAD 토큰도 참조할 수 있다. 첫 위치가 [CLS]가 되게 하는 데이터 구성과 MLM 로스에서 선택한 위치만 사용하는 마스크도 별도로 필요하다.

출력은 $(B,L,V)$의 MLM 로짓과 $(B,2)$의 NSP 로짓이다. fc는 인코더 출력에서 어휘로 직접 투영하며 원래 BERT의 MLM 변환·LayerNorm·입력 임베딩 공유를 모두 재현하지 않는다. NSP 역시 풀러 없이 첫 위치를 바로 분류한다. 출력 헤드를 정의하는 일과 두 목표로 실제 사전 학습하는 일은 다르다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
12레이어·6레이어 모델의 파라메터를 임베딩, 블록과 출력 헤드로 나누어 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    model_12 = BERT(n_layers=12)
    model_6 = BERT(n_layers=6)
    params_12 = sum(p.numel() for p in model_12.parameters())
    params_6 = sum(p.numel() for p in model_6.parameters())
    print(f"12-layer: {params_12:,}")
    print(f"6-layer:  {params_6:,}")
    print(f"Transformer layer params: {params_12 - params_6:,} (for 6 layers)")
    ```
    임베딩 합은 23,437,056개, 블록 하나는 7,087,872개, MLM·NSP 머리 합은 23,071,538개다. 총수는 12레이어 131,563,058개, 6레이어 89,035,826개이며 차이는 블록 6개의 42,527,232개다. 이 예제는 입력·출력 어휘 웨이트를 공유하지 않아 원래 BERT와 계수 구성이 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 예제의 구간 임베딩이 세 개인 사실을 표준 BERT의 요구 조건으로 해석해도 되는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 기본 BERT는 두 구간을 사용한다. 세 번째는 이 코드가 임의로 추가한 행으로, 특정 데이터에서 구간 ID 2를 사용해야 그 용도의 학습 신호가 생긴다. 질문·지문이 있다고 자동으로 세 구간 표현이 필요한 것도 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 변형이 추가하는 드롭아웃과 이미 존재하는 드롭아웃을 구분하고 비교 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    class BERTWithDropout(nn.Module):
        def __init__(self, vocab_size=30000, d_model=768, n_layers=12, heads=12, dropout=0.1):
            super().__init__()
            self.embedding = BERTEmbedding(vocab_size, d_model)
            self.dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads,
                dim_feedforward=d_model * 4, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
            self.fc = nn.Linear(d_model, vocab_size)
            self.nsp = nn.Linear(d_model, 2)

        def forward(self, x, segment_label):
            x = self.embedding(x, segment_label)
            x = self.dropout(x)
            x = self.transformer(x)
            return self.fc(x), self.nsp(x[:, 0])
    ```
    기본 nn.TransformerEncoderLayer에도 드롭아웃이 있다. 이 변형은 임베딩 합 뒤에 드롭아웃을 추가하며 encoder_layer의 확률은 기본값 0.1 그대로다. 생성자 dropout 하나가 모든 확률을 함께 바꾸는 것은 아니다. 같은 초기값·마스크·로스와 분할에서 비교해야 하며 수렴 지연이나 일반화 개선을 보장하지 않는다.

## 정리하며

이 BERT 형태의 예제는 세 임베딩과 두 출력 헤드를 연결하지만 원래 MLM·NSP 구조와 학습을 모두 재현하지 않는다. 패딩·[CLS]·MLM 대상 위치와 웨이트 공유 여부를 확인하며 무작위 초기화 모델을 사전 학습 모델과 구분한다.
