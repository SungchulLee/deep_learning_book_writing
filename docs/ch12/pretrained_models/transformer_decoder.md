# 트랜스포머 디코더

교차 어텐션 없이 셀프 어텐션·피드포워드를 쌓는 디코더 전용 구조의 출발점이다. 다만 아래 구현은 인과 마스크를 자동 생성하지 않고 배치·시퀀스 축도 일치하지 않는다. 코드의 실제 동작과 자기 회귀 모델의 요구 조건을 구분한다.

## 1. 코드

```python
import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 가린 자기 주의
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 순전파
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x, mask)

        return self.fc_out(x)


if __name__ == "__main__":
    pass
```

## 2. 논의

**축과 마스크의 제한:** 토큰 입력 $(B,L)$를 임베딩하면 $(B,L,E)$가 되지만 nn.MultiheadAttention은 batch_first=False로 $(L,B,E)$를 기대한다. 전치 없이 전달하므로 서로 다른 표본을 시퀀스 위치로 해석해 정보가 섞인다. (L,L) 인과 마스크를 주면 내부가 기대하는 (B,B)와 달라 오류가 날 수 있다.

이 모듈은 mask를 받아 전달할 뿐 인과 마스크를 만들지 않는다. 기본 mask=None인 예제는 의도한 위치별 인과성을 검증하지 않는다. 먼저 batch_first=True 또는 일관된 축 전치로 규약을 맞춘 뒤 미래 키 접근을 제한해야 한다. 출력 모양이 (B,L,V)라는 사실만으로 올바른 디코더임을 확인할 수 없다.

블록은 잔차 합 뒤에 LayerNorm을 두는 Post-LN이며 E→d_ff→E 피드포워드를 사용한다. 위치 표는 학습 가능한 Parameter이고 max_len 이내의 길이를 가정한다. 패딩 마스크·KV 캐시·목표 시프트와 학습 로스는 이 클래스에서 처리하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 예제가 출력 모양만 확인하는 이유와 추가로 검토해야 할 축·인과성을 설명하라.

</div>

??? success "풀이"
    배치 우선 텐서를 기본 시퀀스 우선 어텐션에 전달하는 오류가 있어도 같은 모양의 출력이 나올 수 있다. 축을 맞춘 후 다른 표본 변화가 현재 표본의 예측에 영향을 주지 않는지, 미래 토큰 변경이 과거 위치 출력에 영향을 주지 않는지 평가 모드에서 확인해야 한다. 현재 예제는 마스크도 생략한다.

    ```python
    decoder = TransformerDecoder(
        vocab_size=1000, d_model=256, num_heads=8,
        num_layers=4, d_ff=1024
    )
    x = torch.randint(0, 1000, (4, 50))  # batch=4, seq_len=50
    output = decoder(x)
    print(f"Output shape: {output.shape}")  # (4, 50, 1000)
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
인코더 메모리를 읽는 교차 어텐션을 추가하려면 아래 선언 외에 무엇이 필요한가?

</div>

??? success "풀이"
    별도 원문 인코더의 출력을 읽는 구조라면 디코더 쿼리와 원문 키·값의 교차 어텐션을 연결할 수 있다. 아래는 모듈 선언 조각일 뿐이다. forward에서 원문 메모리와 패딩 마스크를 받아 교차 어텐션을 실행하고 잔차·정규화를 연결해야 한다. 디코더 전용 모델이 프롬프트를 읽는 데 교차 어텐션이 반드시 필요한 것은 아니다.
    ```python
    self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    self.norm_cross = nn.LayerNorm(d_model)
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 PreNormDecoderBlock이 바꾸는 잔차 경로와 해결하지 않는 축·마스크 문제를 설명하라.

</div>

??? success "풀이"
    ```python
    class PreNormDecoderBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            normed = self.norm1(x)
            attn_output, _ = self.self_attention(normed, normed, normed, attn_mask=mask)
            x = x + self.dropout(attn_output)
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
            return x
    ```
    Pre-LN은 하위 레이어 입력을 정규화하고 잔차 항등 경로는 바깥에 남긴다. 그러나 이 클래스도 MultiheadAttention 기본 축 규약을 쓰므로 상위 모델의 배치 축 문제를 해결하지 않는다. 인과 마스크도 자동 생성하지 않는다. 레이어 수 12를 경계로 학습 안정성이 보장되는 규칙은 없다.

## 정리하며

이 코드는 디코더 전용 블록의 구성 요소를 보여 주지만 배치 축과 인과 마스크 처리가 미완성이다. Post-LN·Pre-LN의 차이와 별개로, 표본 간 정보 분리·미래 토큰 차단·패딩과 위치 길이를 먼저 확인해야 한다.
