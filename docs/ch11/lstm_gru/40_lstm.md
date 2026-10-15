# LSTM

LSTM 순환층 두 개와 선형 분류층으로 연속 특징 시퀀스를 분류하는 예제다. 임의 입력으로 형태와 매개변수 수만 확인하며 모델을 학습하지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
LSTM — 장단기 기억 신경망
논문: "Long Short-Term Memory" (1997)
핵심: 순차열의 장기 의존을 붙잡는 문 장치
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 층
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 완전 연결층
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 숨은 상태와 세포 상태 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시각의 숨은 상태를 복호
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    model = LSTMModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 10, 100)  # 배치, 순차열, 특징
    print(f"Input: {x.shape}, Output: {model(x).shape}")
```

**출력:**

```
Parameters: 895,498
Input: torch.Size([32, 10, 100]), Output: torch.Size([32, 10])
```

## 2. 논의

기본 입력은 `(32,10,100)`이며 순환 은닉 크기는 256, 출력 클래스 수는 10이다. 각 호출에서 은닉·셀 상태를 0으로 초기화하고 마지막 배열 위치의 은닉 출력을 분류층에 넣는다. 출력은 `(32,10)`의 로짓이다.

패딩 길이를 받지 않으므로 마지막 위치가 실제 문장 끝인지 확인해야 한다. 초기 상태는 장치만 입력에 맞추고 자료형은 기본값을 사용하므로, 모델·입력의 자료형을 바꿀 때는 상태 자료형도 맞춰야 한다.

`dropout=0.2`는 여러 순환 층 사이에 적용된다. 마지막 분류층 뒤나 같은 층의 시간 순환 연결에 드롭아웃을 넣은 것은 아니다. 층이 하나인 설정에서는 내장 층 사이 드롭아웃이 적용되지 않는다.

출력 예시는 텐서 형태와 모델 크기를 보여 준다. 분류 정확도, 수렴이나 장기 기억 능력을 확인한 결과는 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
LSTMModel의 기본 설정에서 매개변수 수를 계산하라.

</div>

??? success "연습문제 1 풀이"
    첫 순환층은 4×256×(100+256+2)=366592개, 둘째 층은 4×256×(256+256+2)=526336개다. 분류층 256×10+10=2570개를 더하면 총 895498개다. 두 편향을 포함하며 드롭아웃은 매개변수가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력과 마지막 상태의 형태, 자료형을 어떻게 확인하는가?

</div>

??? success "연습문제 2 풀이"
    입력은 `(B,L,100)`의 실수 특징이고 초기 상태의 층·배치·은닉 축이 모델과 맞아야 한다. 장치와 자료형도 일치시킨다. 패딩이 있다면 마지막 배열 위치와 실제 마지막 토큰을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
LSTM과 GRU의 전체 분류 모델 매개변수 비율이 반드시 4:3인가?

</div>

??? success "연습문제 3 풀이"
    같은 차원의 순환층 부분은 같은 편향 규약에서 4:3이다. 하지만 동일한 임베딩이나 분류층을 양쪽에 더하면 전체 모델 비율은 달라진다. 모델 크기의 비율만으로 실행 속도나 정확도 비율을 판단하지 않는다.

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

LSTM 예제의 결과는 구조와 출력 형태를 확인하는 시연이다. 길이·패딩, 초기 상태의 자료형과 층 사이 드롭아웃을 명시적으로 이해한 뒤 학습 코드에 연결해야 한다.
