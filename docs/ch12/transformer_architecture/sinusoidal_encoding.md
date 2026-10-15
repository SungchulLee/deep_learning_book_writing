# 사인·코사인 위치 인코딩

사인·코사인으로 만든 고정 위치 벡터를 토큰 표현에 더한다. 위치별 행은 학습하지 않으며, 이 모듈만으로 어텐션이나 전체 트랜스포머가 구성되지는 않는다.

## 1. 코드

```python
"""
사인파 위치 인코딩
"""
import torch
import torch.nn as nn
import math

# ========================================================================
# 메인
# ========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    pass
```

## 2. 논의

입력 x는 $(B,L,E)$의 실수 토큰 표현이다. 위치 표는 $(1,\mathrm{max\_len},E)$ 버퍼이고 배치 축으로 브로드캐스트되어 더해진다. 토큰 ID를 직접 입력하는 모듈은 아니다.

짝수·홀수 열에 같은 주파수의 사인·코사인을 배치한다. 이 구현은 E가 홀수이면 코사인 대입의 열 수가 맞지 않으며, L이 max_len을 넘으면 슬라이스가 입력 길이를 충족하지 못한다. 두 조건을 호출 전에 확인해야 한다.

register_buffer로 저장한 표는 학습 매개변수가 아니지만 기본적으로 state_dict와 장치 이동의 대상이다. 이 코드에는 드롭아웃이 없다. 수식상 더 긴 위치를 계산할 수 있다는 사실과 학습 길이를 넘은 모델의 성능은 별개다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 모듈의 학습 매개변수 수와 위치 표의 저장량을 구하라.

</div>

??? success "연습문제 1 풀이"
    학습 매개변수는 0개다. 버퍼는 max_len×E개의 원소를 저장하므로 메모리가 전혀 들지 않는 방식은 아니다. 같은 위치 표를 배치마다 따로 학습하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력 검증에서 확인할 차원과 길이 조건은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    입력은 배치·위치·특징의 세 축을 갖고 마지막 축은 d_model이어야 한다. 이 구현에서는 d_model이 짝수이고 시퀀스 길이가 max_len 이하여야 한다. 입력과 버퍼의 장치·자료형 호환성도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사인·코사인 위치 표현이 긴 입력에서도 좋은 예측을 보장하는가?

</div>

??? success "연습문제 3 풀이"
    새 위치의 벡터를 계산하는 것은 가능하지만 모델이 그 위치 관계를 학습한 것은 아니다. 긴 입력을 허용하도록 버퍼·마스크 등을 확장한 뒤에도 별도 길이별 평가가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

고정 위치 표는 토큰 표현에 순서 정보를 더하며 학습 매개변수와는 구별된다. 이 예제의 짝수 특징 차원과 최대 길이 조건을 지키고, 길이 확장과 예측 품질을 따로 판단한다.
