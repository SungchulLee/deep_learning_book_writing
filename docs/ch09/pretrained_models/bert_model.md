# BERT 모형

이 모듈은 트랜스포머 인코더에 풀링 층을 얹어 BERT 방식의 모형을 구현한다. BERT의 구조는 인코딩 등뼈와 과제에 맞는 머리를 갈라 두어, 출력 머리만 바꾸면 같은 사전 학습 인코더를 가린 언어 모형화, 분류, 질의응답 등 여러 아래쪽 과제에 다시 쓸 수 있다.

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

`BERTModel` 클래스는 `TransformerEncoder`를 감싸고 풀링 연산을 더한다. 인코더는 입력의 토큰마다 맥락이 담긴 표현을 낸다. 풀러는 첫 자리(`[CLS]` 토큰)의 표현을 꺼내 선형 변환을 적용하고 `tanh` 활성을 거치게 한다. 이 풀링된 출력이 수열 전체에 대한 크기가 고정된 표현을 준다.

이 모형은 출력을 둘 돌려준다. 인코딩된 수열 전체와 풀링된 `[CLS]` 표현이다. 수열 전체 출력은 토큰 하나하나에 대해 예측해야 하는 개체명 인식이나 가린 언어 모형화 같은 토큰 수준 과제에 쓴다. 풀링된 출력은 분류나 문장 비슷함 같은 수열 수준 과제에 쓴다.

풀러의 `tanh` 활성은 정규화 단계 노릇을 하여 풀링된 표현을 $[-1, 1]$ 범위로 묶는다. 본디 BERT 구현에서 온 이 설계 선택은 표현이 아래쪽 과제 머리로 넘어가기 전에 안정되게 해 준다. 풀러는 사전 학습 중에 인코더와 함께 학습된다.

## 연습문제

**연습문제 1.**
`d_model=768`에서 배치 크기 8, 수열 길이 128인 입력을 넣어 모형을 따라가며 꼴을 좇아라. `encoded`와 `pooled`의 꼴은 무엇인가?

??? success "연습문제 1 풀이"

    - 들임 `x`: `(8, 128)`(토막 번호의 묶음)
    - 부호기 뒤: `encoded`의 꼴은 `(8, 128, 768)`이며 토막마다 768차원 벡터 하나다
    - `encoded[:, 0, :]`: 꼴 `(8, 768)` — 표본마다의 `[CLS]` 토막
    - 모으개 선형 + tanh 뒤: `pooled`의 꼴은 `(8, 768)`이다

    수열 전체 표현은 128개 자리를 모두 지키고, 풀링된 출력은 수열마다 벡터 하나로 눌러 담는다.

---

**연습문제 2.**
BERT가 풀러에 ReLU 같은 다른 활성이나 활성 없음 대신 `tanh`을 쓰는 까닭은 무엇인가? 이 선택이 아래쪽 미세 조정에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    `tanh` 활성은 출력을 $[-1, 1]$으로 묶어 은근한 정규화를 주며 풀링된 표현의 크기가 제멋대로 커지지 않게 한다. 그래서 (무작위로 초기화되는) 아래쪽 머리가 미세 조정 초기에 더 안정된다. ReLU라면 음수를 0으로 만들어 쓸모 있을 수 있는 정보를 잃는다. 활성이 없으면 풀러가 그저 선형 사영이 되어 아래쪽 층의 초기화를 더 조심해야 할 수 있다. 실제로 `tanh` 풀러는 잘 통하는 설계 관례이지만 꼭 있어야 하는 것은 아니다. 어떤 미세 조정 방식은 아예 건너뛴다.

---

**연습문제 3.**
가린 자리의 본디 토큰을 맞히는 가린 언어 모형화 머리로 `BERTModel`을 넓혀라. 그 머리에는 층 정규화와 어휘 토큰마다의 편향이 들어가야 한다.

??? success "연습문제 3 풀이"
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
    가린 언어 모형화 머리는 토큰 표현마다 GELU와 층 정규화를 갖춘 조밀 층을 거치게 한 뒤 어휘 크기로 사영한다. 토큰마다의 편향은 어휘 항목마다 예측 로짓을 세밀하게 다스리게 해 준다.

## 정리하며

**다룬 것** — BERT 모형

`BERTModel` 클래스는 `TransformerEncoder`를 감싸고 풀링 연산을 더한다.

핵심 클래스는 `BERTModel`, `BERTForMLM`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
