# 트랜스포머 모델

실수 특징 시퀀스를 분류하는 트랜스포머 기준 모델이다. 입력·출력 형식을 LSTM·CNN과 맞춰 비교할 수 있지만 층 수·계수·수용 영역이 달라 자동으로 공정한 비교가 되는 것은 아니다.

## 1. 코드

```python
import torch
import torch.nn as nn


class TransformerForComparison(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8,
                 num_layers=6, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 전역 평균 풀링
        return self.classifier(x)


if __name__ == "__main__":
    pass
```

## 2. 논의

입력 $(B,L,64)$를 선형층으로 256차원에 투영하고 학습 위치 표를 더한다. 이후 $(L,B,256)$로 전치해 PyTorch 기본 인코더 규약에 맞춘다. 위치 축 평균을 분류해 $(B,10)$ 로짓을 반환한다.

위치 표는 Parameter이며 길이는 최대 100이다. L>100이면 슬라이스는 100에서 멈추어 덧셈의 길이가 맞지 않는다. 인과·패딩 마스크는 없고 평균도 모든 위치를 포함하므로, 가변 길이 패딩 입력에는 마스크 전달과 유효 위치 평균이 필요하다.

분류기 입력을 동일 가중치로 평균해도 각 토큰은 이미 셀프 어텐션으로 다른 정보를 읽는다. 평균 풀링이 토큰의 학습된 중요도 표현을 원천적으로 막는 것은 아니다. [CLS] 풀링과의 품질·속도는 같은 분할·학습 예산에서 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 페이지 기본 모델과 두 기준 모델의 출력·계수 수를 비교하고 공정성의 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    model = TransformerForComparison(input_dim=64)
    x = torch.randn(32, 100, 64)
    output = model(x)
    print(f"Output shape: {output.shape}")  # (32, 10)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    ```
    출력은 (32,10), 트랜스포머는 4,783,370개다. 입력 64·출력 10으로 맞춘 LSTM은 858,634개, CNN은 138,186개다. 계수와 깊이가 크게 달라 한 번의 실행 시간을 구조 자체의 우열로 해석할 수 없다. 과제 성능·학습 예산과 지연을 함께 보고한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
길이 100을 넘길 때 발생하는 문제와 위치 표 확장·고정 수식 변경의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    위치 슬라이스가 범위 밖을 읽는 것이 아니라 길이 100에 멈춰 임베딩과 더할 때 모양이 맞지 않는다. 표를 늘리면 새 위치 계수는 별도 초기화·학습이 필요하다. 고정 사인·코사인 수식으로 바꾸어도 모델에 더하는 코드와 저장·장치를 연결해야 하며, 긴 입력 품질을 보장하지 않는다. 아래는 math·seq_len·d_model 등이 외부에 있는 위치 표 생성 조각이지 완성된 모델 수정이 아니다.

       ```python
       pe = torch.zeros(seq_len, d_model)
       pos = torch.arange(0, seq_len).unsqueeze(1).float()
       div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
       pe[:, 0::2] = torch.sin(pos * div)
       pe[:, 1::2] = torch.cos(pos * div)
       ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
[CLS] 변형의 토큰·위치 수와 패딩 처리를 검토하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    class TransformerWithCLS(nn.Module):
        def __init__(self, input_dim, d_model=256, num_heads=8,
                     num_layers=6, num_classes=10):
            super().__init__()
            self.embedding = nn.Linear(input_dim, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            self.pos_encoding = nn.Parameter(torch.randn(1, 101, d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward=d_model * 4
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.classifier = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.size(0)
            x = self.embedding(x)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = x.transpose(0, 1)
            x = self.transformer(x)
            return self.classifier(x[0])  # CLS 토큰의 출력
    ```
    [CLS]를 붙여 입력 길이 L에서 L+1개 토큰이 되고 학습 위치 표는 101개다. 기존 최대 원문 길이는 100개로 유지된다. 이 변형도 패딩 마스크가 없어 유효 입력 길이를 처리하려면 확장이 필요하다. [CLS]가 처음부터 좋은 요약인 것은 아니며 분류 손실로 학습한다.

## 정리하며

실수 특징 투영·위치 표·평균 풀링의 차원과 패딩 규약을 확인한다. 이 모델과 기준 모델은 입력·출력을 맞췄지만 계수·계산 예산이 다르므로 속도만으로 공정한 구조 비교를 결론내리지 않는다.
