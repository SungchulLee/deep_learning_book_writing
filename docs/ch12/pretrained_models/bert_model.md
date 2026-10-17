# BERT 모델

BERT 방식의 토큰 표현과 첫 토큰 풀링을 설명하는 간단한 래퍼다. 체크포인트를 로드하거나 BERT 사전 학습을 수행하지 않는다. 사용하는 TransformerEncoder의 축 불일치도 그대로 이어지므로 실제 사전 학습 BERT와 구분해 읽는다.

## 1. 코드

```python
import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder


class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff
        )
        self.pooler = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        encoded = self.encoder(x, mask)
        pooled = torch.tanh(self.pooler(encoded[:, 0, :]))  # [CLS] 토큰
        return encoded, pooled


if __name__ == "__main__":
    pass
```

## 2. 논의

입력 토큰은 $(B,L)$이며 의도한 encoded는 $(B,L,E)$, pooled는 $(B,E)$다. 풀러는 첫 위치의 표현에 선형 레이어와 tanh를 적용한다. 첫 토큰을 [CLS]로 만드는 것은 토크나이저와 데이터 구성의 역할이며 이 모듈이 자동으로 삽입하지 않는다.

현재 외부 TransformerEncoder는 배치 우선 입력을 시퀀스 우선 어텐션에 그대로 넘긴다. 따라서 마스크가 없으면 다른 표본의 정보가 섞일 수 있고, (B,L) 마스크를 주면 모양 오류가 생길 수 있다. 반환 모양이 의도와 같아도 각 문장의 양방향 문맥을 올바르게 처리했다는 증거는 아니다.

tanh는 성분을 (-1,1) 범위로 제한하지만 평균·분산을 정규화하는 연산은 아니며 포화하면 그래디언트가 작아질 수 있다. 풀러가 학습되는지는 로스에 연결되는지에 달려 있다. 아래 MLM 예제는 pooled를 반환만 하므로 토큰 MLM 로스만 쓰면 풀러에 학습 신호가 가지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (8,128), d_model=768일 때 두 반환값의 의도한 모양과 모양 검증의 한계를 설명하라.

</div>

??? success "풀이"
    encoded는 (8,128,768), pooled는 (8,768)이다. 하지만 내부 어텐션의 축 오류가 이 모양을 그대로 유지할 수 있으므로 모양만으로 표본별 문맥 처리를 검증할 수 없다. 첫 위치가 실제로 [CLS]인지도 입력 구성에서 확인한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
풀러의 tanh를 정규화라고 부를 수 있는가?

</div>

??? success "풀이"
    출력 범위를 제한하는 엑티베이션 함수이며 LayerNorm처럼 평균·분산을 맞추지는 않는다. 큰 절댓값에서 포화할 수 있다. 선형 풀링·마스크 평균 풀링 등과의 품질 차이는 학습 목표와 과제에 따라 평가한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 MLM 헤드의 로스 대상, 바이어스와 웨이트 공유를 검토하라.

</div>

??? success "풀이"
    ```python
    class BERTForMLM(nn.Module):
        def __init__(self, vocab_size, d_model=768, num_heads=12,
                     num_layers=12, d_ff=3072):
            super().__init__()
            self.bert = BERTModel(vocab_size, d_model, num_heads,
                                  num_layers, d_ff)
            self.mlm_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.decoder = nn.Linear(d_model, vocab_size)
            self.bias = nn.Parameter(torch.zeros(vocab_size))

        def forward(self, x, mask=None):
            encoded, pooled = self.bert(x, mask)
            mlm_output = self.mlm_head(encoded)
            predictions = self.decoder(mlm_output) + self.bias
            return predictions, pooled
    ```
    출력은 (B,L,V)의 로짓이다. 마스크 대상으로 선정한 위치의 원래 토큰만 로스에 포함하고 나머지 정답은 ignore_index로 제외해야 한다. 이 코드의 decoder는 기본 바이어스를 가지는데 self.bias도 추가하므로 어휘 바이어스 두 개가 중복된다. 입력 임베딩과 출력 웨이트를 공유하지 않으며, pooled를 반환하는 것만으로 풀러가 학습되지는 않는다.

## 정리하며

토큰 표현과 첫 위치 풀링은 서로 다른 출력을 제공한다. 사전 학습·특수 토큰 삽입을 이 래퍼가 자동 수행하지 않으며, 외부 인코더의 축 문제와 MLM의 로스 대상·중복 바이어스·비공유 웨이트를 확인해야 한다.
