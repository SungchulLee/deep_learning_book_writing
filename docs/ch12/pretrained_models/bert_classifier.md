# BERT 분류기

BERT의 첫 토큰 분류 방식에서 아이디어를 얻은 트랜스포머 분류기다. 사전 학습 가중치를 읽지 않고 무작위로 초기화하므로, 이 코드 자체를 사전 학습 BERT의 미세 조정이라고 부르지는 않는다.

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

정수 입력 $(B,L)$를 임베딩하고 사인·코사인 위치 표현을 더한 뒤 $(L,B,E)$로 전치한다. nn.TransformerEncoder의 기본 축 규약에 맞게 처리하며 첫 위치의 출력 x[0]을 분류한다. 같은 폴더의 사용자 정의 TransformerEncoder를 호출하는 것은 아니다.

첫 위치가 [CLS]가 되도록 토큰을 구성하는 일은 호출자의 책임이다. mask는 (B,L)의 key_padding_mask이며 불리언 True가 패딩을 뜻한다. Hugging Face에서 흔한 1=유효 attention_mask와 그대로 교환해서는 안 된다. 첫 위치가 패딩이거나 모든 키가 제외된 입력도 따로 다뤄야 한다.

분류기는 E→E→C의 두 선형층 사이에 ReLU와 드롭아웃을 사용하고 (B,C) 로짓을 반환한다. 교차 엔트로피에는 로짓을 직접 전달한다. 위치 표현은 utils.positional_encoding에서 가져오므로 모듈 검색 경로가 실행 위치에 의존한다는 점도 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
제시된 설정의 출력 모양과 매개변수 수를 계산하고 무작위 입력 검사의 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    model = BERTClassifier(vocab_size=10000, num_classes=5)
    x = torch.randint(0, 10000, (4, 32))  # batch=4, seq_len=32
    output = model(x)
    print(f"Output shape: {output.shape}")  # (4, 5)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    ```
    출력은 (4,5)이며 전체 매개변수는 50,801,669개다. 임베딩 7,680,000개, 인코더 6층 42,527,232개, 분류기 594,437개를 합한다. 고정 위치 버퍼는 제외한다. 무작위 토큰 검사는 실제 [CLS] 구성이나 분류 성능을 확인하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 코드에서 첫 토큰 풀링과 마스크 평균 풀링을 어떻게 비교해야 하는가?

</div>

??? success "연습문제 2 풀이"
    이 모델은 사전 학습되지 않았으므로 첫 토큰이 이미 문장 요약을 배웠다고 가정하면 안 된다. 분류 손실을 통해 그 역할을 학습하도록 구성한다. 평균 풀링은 PAD를 제외한 유효 토큰 수로 나누고 빈 입력 정책을 정해야 한다. 같은 데이터 분할과 학습 예산에서 검증 성능을 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 분류기 변형의 LayerNorm 위치와 매개변수 증가량을 실제 코드에 따라 설명하라.

</div>

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
    코드의 LayerNorm은 첫 E→E 층 뒤에 있으며 마지막 투영 바로 앞이 아니다. $k=\lfloor E/2\rfloor$, 클래스 C일 때 기존 머리 대비 증가량은 $Ek+k+2E+(k-E)C$개다. E=768, C=5이면 294,912개가 증가한다. GELU·정규화·폭 변경의 효과는 개별 검증이 필요하며 더 깊다고 반드시 성능이 좋아지지는 않는다.

## 정리하며

이 분류기는 무작위 초기화된 인코더의 첫 위치 출력을 사용한다. [CLS] 입력 구성과 패딩 마스크 규약을 맞추고, 사전 학습 모델 미세 조정과 새 모델 학습을 구분한다. 분류기 변형은 실제 계수 수와 검증 성능을 함께 비교한다.
