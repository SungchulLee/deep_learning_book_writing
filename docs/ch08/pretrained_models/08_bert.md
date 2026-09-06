# BERT

BERT(트랜스포머 기반 양방향 인코더 표현)는 2018년 논문 "BERT: Pre-training of Deep Bidirectional Transformers"에서 나왔다. 글을 왼쪽에서 오른쪽으로 읽던 앞선 언어 모형과 달리 BERT는 가린 언어 모형화와 다음 문장 맞히기로 글을 양방향으로 처리한다. 이 구조는 질의응답, 감성 분석, 개체명 인식을 비롯한 폭넓은 자연어 처리 과제의 바탕이 되었다.

## 코드

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

## 논의

BERT의 임베딩 층은 서로 다른 임베딩 셋을 아우른다. 낱말 조각마다를 나타내는 토큰 임베딩, 토큰이 수열의 어디에 있는지를 담는 위치 임베딩, 두 입력 문장을 갈라 주는 구간 임베딩이다. 셋을 더한 뒤 층 정규화를 하는 이 설계가 BERT에 한 문장 과제와 문장 쌍 과제 모두에 필요한 맥락을 준다.

BERT의 알맹이는 트랜스포머 인코더 층의 더미이다. 층마다 다중 머리 자기 주의를 하고 이어 자리별 순전파 신경망을 적용하며, 둘 다 잔차 연결과 층 정규화로 감싼다. 인코더가 모든 자리에 한꺼번에 주의하므로 BERT는 양방향 맥락을 잡아낸다. 왼쪽만 볼 수 있는 자기 회귀 모형에 견주어 큰 이점이다.

이 모형은 출력을 둘 낸다. 첫째는 가린 언어 모형화에 쓰는 토큰별 출력으로, 무작위로 가린 토큰을 그 둘레 맥락에서 맞힌다. 둘째는 다음 문장 맞히기를 위해 `[CLS]` 토큰에 적용하는 이진 분류기로, 모형에 문장 사이의 관계를 이해하도록 가르친다. 이 두 사전 학습 목표가 함께 BERT를 두루 쓰이는 바탕 모형으로 만든다.

## 연습문제

**연습문제 1.**
기본 매개변수로 `BERT` 모형을 만들고 학습 가능한 매개변수의 총 개수를 셈하라. 그다음 `n_layers`를 12에서 6으로 줄여 견주어라. 매개변수 가운데 트랜스포머 층에서 오는 몫과 임베딩 층에서 오는 몫은 각각 얼마인가?

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
    임베딩 층(토큰, 위치, 구간, 층 정규화)은 깊이와 무관하게 정해진 몫을 차지하고 트랜스포머 인코더 층마다 대략 같은 수의 매개변수를 더한다. 12층 모형에서 임베딩은 대체로 전체의 20~25%쯤이다.

---

**연습문제 2.**
BERT가 구간 임베딩에 값을 둘이 아니라 셋(0, 1, 2) 두는 까닭을 설명하라. 어떤 사전 학습이나 미세 조정 상황에서 셋째 구간이 쓸모 있겠는가?

??? success "연습문제 2 풀이"
    표준 BERT 사전 학습은 첫 문장에 구간 0, 둘째 문장에 구간 1을 쓴다. 셋째 값(구간 2)은 서로 다른 글 구간이 셋 필요할 수 있는 아래쪽 과제, 이를테면 입력이 질문과 지문과 추가 맥락으로 이루어지는 어떤 질의응답 설정을 위한 편의로 넣어 두었다. 실제로 본디 BERT 논문은 구간을 둘만 쓰지만, 셋째 임베딩이 있으면 구조를 바꾸지 않고도 여유가 생긴다.

---

**연습문제 3.**
임베딩 뒤, 트랜스포머 인코더 앞에 드롭아웃 층을 더하도록 `BERT` 클래스를 고쳐라. 인공 데이터로 작은 가린 언어 모형화 과제를 학습시키며 드롭아웃이 수렴 속도에 영향을 주는지 살펴라.

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
    임베딩 뒤에 드롭아웃을 더하면 모형이 규제되어 처음 수렴이 조금 느려질 수 있지만, 특히 학습 자료가 적을 때 검증 데이터에서의 일반화가 대체로 나아진다.
