# BERT 분류기

BERT 기반 글 분류는 사전 학습된 양방향 트랜스포머 인코더를 아래쪽 분류 과제에 맞추어 미세 조정한다. `[CLS]` 토큰 표현 위에 과제에 맞는 분류 머리를 얹어, BERT의 넉넉한 맥락 임베딩으로 감성 분석, 주제 분류, 자연어 추론 같은 과제에서 좋은 성능을 낸다.

## 1. 코드

```python
import torch
import torch.nn as nn
import sys

sys.path.append('..')
from utils.positional_encoding import PositionalEncoding


class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=768,
                 num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [seq, batch, dim]
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[0]  # [CLS] 토큰
        return self.classifier(x)


if __name__ == "__main__":
    pass
```

## 2. 논의

`BERTClassifier`는 트랜스포머 기반 모형의 표준 미세 조정 방식을 따른다. 입력 토큰을 먼저 임베딩으로 바꾸고 수열의 차례 정보를 넣으려고 위치 인코딩을 더한다. `src_key_padding_mask` 매개변수는 길이가 제각각인 배치에서 채움 토큰을 무시하게 하여 주의 가중치가 실제 내용에 대해서만 셈해지도록 한다.

분류 머리는 첫 토큰의 표현에 적용하는, ReLU 활성과 드롭아웃을 갖춘 두 층짜리 다층 퍼셉트론이다. BERT의 관례에서 첫 토큰은 특별한 `[CLS]` 토큰이고 그 마지막 숨은 상태가 입력 전체를 아우르는 표현 노릇을 한다. 두 선형 사영 사이의 드롭아웃 층이 규제를 주는데, 작은 데이터셋으로 미세 조정할 때 특히 중요하다.

트랜스포머 인코더는 파이토치의 `nn.TransformerEncoder`가 요구하는 `(seq_len, batch, dim)` 꼴로 입력을 처리하므로 인코딩 전에 입력을 옮겨 놓음에 유의하라. `[CLS]` 표현을 꺼낸 뒤 분류기가 목표 부류에 대한 로짓을 내고, 학습 중에는 대개 그것을 교차 엔트로피 손실에 넣는다.

## 연습문제

**연습문제 1.**
`vocab_size=10000`, `num_classes=5`로 `BERTClassifier`를 만들고 무작위 토큰 번호의 배치를 넣어 보아라. 출력의 꼴을 확인하고 전체 매개변수를 세어라.

??? success "연습문제 1 풀이"
    ```python
    model = BERTClassifier(vocab_size=10000, num_classes=5)
    x = torch.randint(0, 10000, (4, 32))  # batch=4, seq_len=32
    output = model(x)
    print(f"Output shape: {output.shape}")  # (4, 5)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    ```
    출력의 꼴은 `(batch_size, num_classes)` = `(4, 5)`이다.

---

**연습문제 2.**
이를테면 모든 토큰 표현의 평균이 아니라 `[CLS]` 토큰을 분류기의 입력으로 고르는 까닭을 설명하라. 어떤 상황에서 평균 풀링이 더 나을 수 있는가?

??? success "연습문제 2 풀이"
    `[CLS]` 토큰은 사전 학습 중에 (다음 문장 맞히기로) 수열 전체의 요약 표현을 담도록 특별히 학습된다. 자기 주의로 다른 모든 토큰에 주의하므로 전역 정보를 모은다. 평균 풀링은 모든 토큰 표현을 평균 내는데, 채움이나 덜 관련 있는 토큰의 잡음이 중요한 토큰의 신호를 묽게 할 수 있다. 다만 문장 비슷함 같은 과제나 `[CLS]` 토큰이 따로 사전 학습되지 않은 경우에는 모든 자리에 걸쳐 더 고른 표현을 주므로 평균 풀링이 나을 수 있다.

---

**연습문제 3.**
분류 머리를 더 표현력 있는 짜임으로 고쳐라. 셋째 선형 층을 더하고 ReLU 대신 GELU를 쓰며 마지막 사영 앞에 층 정규화를 넣어라. 본디 것과 매개변수 수를 견주어라.

??? success "연습문제 3 풀이"
    ```python
    self.classifier = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
        nn.Dropout(0.1),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(d_model // 2, num_classes)
    )
    ```
    세 층짜리 머리는 두 층짜리보다 대략 $d_{\text{model}} \times (d_{\text{model}} / 2) + d_{\text{model}}$개의 매개변수를 더 쓴다. GELU 활성은 더 매끄러운 기울기를 주고 층 정규화는 중간 표현을 안정되게 하여 미세 조정의 안정성을 높일 수 있다.

## 정리하며

**다룬 것** — BERT 분류기

`BERTClassifier`는 트랜스포머 기반 모형의 표준 미세 조정 방식을 따른다.

핵심 클래스는 `BERTClassifier`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
