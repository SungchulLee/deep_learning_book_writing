# 트랜스포머 모형

이 트랜스포머 모형은 수열 분류 과제에서 순환 신경망과 합성곱 신경망 기준선과 곧바로 견주도록 만들었다. (토큰 번호가 아니라) 이어진 값의 입력 수열을 받고 분류에 인코더 출력의 전역 평균 풀링을 써서, 귀납 편향이 다른 구조들과 공평하게 견줄 자리를 준다.

## 코드

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

## 논의

임베딩 찾기 표를 쓰는 글 기반 트랜스포머와 달리 이 모형은 선형 사영으로 이어진 값의 입력 특징을 모형 차원으로 잇댄다. 입력이 띄엄띄엄한 토큰이 아니라 (센서 측정값이나 소리 특징처럼) 실숫값 벡터의 수열일 때 이렇게 해야 한다. 학습되는 위치 인코딩이 최대 수열 길이 100까지 자리 정보를 더한다.

이 모형은 전역 평균 풀링(`x.mean(dim=0)`)으로 트랜스포머의 출력을 모든 자리에 걸쳐 모아 분류용 벡터 하나로 만든다. `[CLS]` 토큰을 쓰는 대신 택할 수 있는 길이며 모든 자리에 같은 무게를 준다는 이점이 있다. 특정 자리가 더 중요한 과제에서는 `[CLS]` 토큰이나 주의 기반 풀링이 더 알맞을 수 있다.

이 구조는 파이토치의 `nn.TransformerEncoder`를 쓰는데 입력 꼴이 `(seq_len, batch, d_model)`이어야 한다. 트랜스포머에 넣기 전에 입력을 옮겨 놓고 출력은 수열 차원(dim=0)을 따라 평균 낸다. 층 6개, 머리 8개, 모형 차원 256으로 크기가 알맞아 순환 신경망·합성곱 신경망 기준선과 속도를 공평하게 견줄 수 있다.

## 연습문제

**연습문제 1.**
꼴이 `(32, 100, 64)`인 배치를 `TransformerForComparison` 모형에 넣고 출력의 꼴이 `(32, 10)`인지 확인하라. 전체 매개변수를 세어 순환 신경망·합성곱 신경망과 견주어라.

??? success "연습문제 1 풀이"
    ```python
    model = TransformerForComparison(input_dim=64)
    x = torch.randn(32, 100, 64)
    output = model(x)
    print(f"Output shape: {output.shape}")  # (32, 10)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    ```
    출력의 꼴은 기대대로 `(32, 10)`이다. 트랜스포머는 주의와 순전파 아래 층을 갖춘 인코더 층이 여섯이라 대체로 순환 신경망이나 합성곱 신경망 기준선보다 매개변수가 훨씬 많다.

---

**연습문제 2.**
위치 인코딩의 최대 길이가 100으로 고정되어 있다. 100보다 긴 수열을 모형에 넣으면 어떻게 되는지 설명하고 서로 다른 해결책을 둘 내놓아라.

??? success "연습문제 2 풀이"
    `x.size(1) > 100`이면 `self.pos_encoding[:, :x.size(1), :]` 자르기가 버퍼 바깥의 색인에 닿으려 하여 색인 오류가 난다. 해결책은 둘이다.

    1. **`max_len`을 늘린다**: 더 긴 수열을 담도록 위치 인코딩의 크기를 (이를테면 5000으로) 바꾼다. 기억이 더 들지만 간단하다.
    2. **사인파 인코딩을 쓴다**: 학습되는 자리를 사인파 인코딩으로 바꾼다. 담아 두는 매개변수 없이 어떤 길이든 셈할 수 있다.
       ```python
       pe = torch.zeros(seq_len, d_model)
       pos = torch.arange(0, seq_len).unsqueeze(1).float()
       div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
       pe[:, 0::2] = torch.sin(pos * div)
       pe[:, 1::2] = torch.cos(pos * div)
       ```

---

**연습문제 3.**
전역 평균 풀링을 학습되는 `[CLS]` 토큰 방식으로 바꾸어라. 학습되는 토큰을 수열 앞에 붙여 트랜스포머에 넣고 첫 출력 자리를 분류에 써라. 평균 풀링과 성능을 견주어라.

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
    `[CLS]` 토큰은 자기 주의로 분류에 관련된 정보를 모으는 법을 배운다. 모든 자리가 똑같이 뜻있는 과제에서는 평균 풀링도 비슷한 성능을 낼 수 있다. 자리마다 중요도가 다른 과제에서는 `[CLS]` 토큰이 가장 관련 있는 부분에 집중하는 법을 배울 수 있다.
