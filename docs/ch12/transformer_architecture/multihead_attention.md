# 멀티헤드 어텐션

Q·K·V를 여러 헤드로 나누어 점수를 계산하고, 헤드별 문맥을 이어 붙이는 구현이다. 쿼리와 키·값을 따로 받아 셀프·크로스 어텐션에 사용할 수 있다.

## 1. 코드

```python
"""
다중 머리 주의
"""
import torch
import torch.nn as nn
import math

# ========================================================================
# 메인
# ========================================================================

class MultiHeadAttention(nn.Module):
    """다중 머리 주의 구조"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output, attention_weights


if __name__ == "__main__":
    pass
```

## 2. 논의

모델 차원 d가 헤드 수 h로 나누어져야 하며, 헤드당 차원은 d/h다. 투영 뒤 텐서는 $(B,h,N,d/h)$로 바뀐다. 쿼리 길이와 키 길이가 다르면 점수는 $(B,h,N_q,N_k)$이고 값의 길이는 키 길이와 같아야 한다.

mask==0 위치를 점수에서 제외하므로 마스크의 축과 브로드캐스트를 맞춘다. 전부 가려진 행의 별도 처리는 없고, 반환 웨이트는 드롭아웃 이후라 학습 중 합이 1이 아닐 수 있다. 출력 투영은 헤드 결과를 모델 차원으로 섞는다. 클래스 정의 자체에는 위치 표현이나 학습 데이터가 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
모델 차원 d인 이 모듈의 바이어스 포함 파라메터 수를 구하라.

</div>

??? success "풀이"
    네 개의 d→d 선형 레이어가 있으므로 $4d^2+4d$개다. 전체 차원을 고정한 채 헤드 수를 바꾸어도 파라메터 수는 같지만 헤드별 웨이트 저장량은 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
드롭아웃 적용 전후의 어텐션 웨이트와 평가 동작을 비교하라.

</div>

??? success "풀이"
    소프트맥스 직후 유효 행은 합이 1이다. 학습 드롭아웃 후에는 유지된 값을 1/(1-p)배 하므로 한 번의 호출에서 합이 1이라는 보장은 없다. eval 모드에서는 드롭아웃이 꺼진다. 마스킹과 드롭아웃은 목적과 적용 단계가 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
쿼리 길이 Nq, 키 길이 Nk에서 웨이트 수와 주된 연산량을 설명하라.

</div>

??? success "풀이"
    배치를 제외한 웨이트는 $hN_qN_k$개다. 전체 피처 차원 d가 고정이면 점수·가중합은 $O(N_qN_kd)$이고 투영 비용은 별도다. 셀프 어텐션은 Nq=Nk일 때 이차가 되며 헤드 수만 늘린다고 이 비용이 줄지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

멀티헤드는 피처 축을 나눈 여러 위치 분포를 결합한다. 키·값 길이, 마스크 축과 드롭아웃 이후 웨이트의 의미를 확인한 뒤 다른 모델에 연결한다.
