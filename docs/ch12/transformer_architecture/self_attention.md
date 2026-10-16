# 셀프 어텐션

같은 입력에서 Q·K·V를 투영해 위치별 가중합을 계산하는 셀프 어텐션 예제다. 클래스 이름과 달리 heads 인수에 따른 실제 헤드 분할은 구현되어 있지 않으므로, 단일 분포 계산과 스케일의 관계를 구분해야 한다.

## 1. 코드

```python
"""
자기 주의 구현
"""
import torch
import torch.nn as nn
import math

# ========================================================================
# 메인
# ========================================================================

class SelfAttention(nn.Module):
    """자기 주의 구조의 커널"""
    def __init__(self, embed_size, heads=1):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        N, seq_len, _ = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 스케일 조정 내적 주의
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        
        return self.fc_out(out), attention_weights


if __name__ == "__main__":
    pass
```

## 2. 논의

입력은 $(B,N,E)$이고 세 선형 레이어로 Q·K·V를 만든다. 코드는 헤드 축으로 reshape하지 않아 점수는 항상 $(B,N,N)$ 하나다. heads=1이면 피처 차원 E의 단일 헤드 스케일과 맞지만 heads>1이면 내적은 여전히 E차원인 채 분모만 sqrt(E/heads)로 바뀐다. 일반적인 멀티헤드 어텐션과 다르다.

마스크는 점수에 브로드캐스트할 수 있어야 하고 0인 위치를 제외한다. 모든 키를 가린 행에는 NaN이 생길 수 있다. 출력 형태는 $(B,N,E)$이고 웨이트는 $(B,N,N)$이다. 학습·출력 시연이 없으므로 이 정의만으로 정확도나 학습된 관계를 평가할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
피처 크기 E인 이 클래스의 파라메터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"
    Q·K·V와 출력의 네 Linear(E,E)에 각각 $E^2+E$개가 있어 총 $4E^2+4E$개다. heads 인수는 파라메터 수를 바꾸지 않으며 실제 분할도 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
heads>1일 때 코드가 표준 멀티헤드 연산이 아닌 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    헤드 축으로 나누지 않아 E차원 내적 하나를 계산한다. 분모만 헤드 차원 기준이어서 점수 크기가 달라진다. 표준 구조에는 Q·K·V를 헤드별로 나누고 독립적인 위치 웨이트를 계산해 결합하는 단계가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 코드에 어텐션 드롭아웃을 넣으면 웨이트 합과 평가 모드는 어떻게 달라지는가?

</div>

??? success "연습문제 3 풀이"
    소프트맥스 뒤 일부 웨이트가 0이 되고 나머지는 재스케일되어 학습 중 행 합은 항상 1이 아니다. 평가 모드에서는 끈다. 마스크를 대신하거나 의미 관계를 더 균등하게 학습한다고 보장하지 않는다.

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
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

이 SelfAttention은 실제로 단일 위치 웨이트 행렬을 계산한다. heads 인수의 스케일 변경을 멀티헤드 분할로 해석하지 말고 입력·마스크와 출력 형태를 명확히 확인한다.
