# RNN 어텐션 메커니즘

가산 점수의 BahdanauAttention과 내적·일반형 점수의 LuongAttention을 비교한다. 이들은 순환 모델에 연결할 수 있는 어텐션 모듈이며 두 클래스를 합친 완전한 트랜스포머는 아니다.

## 1. 코드

```python
"""
순환 뉴럴 넷트웍을 위한 주의 구조
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class BahdanauAttention(nn.Module):
    """바다나우(더하기) 주의"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        # query: [batch, hidden] - 디코더의 숨은 상태
        # keys: [batch, seq_len, hidden] - 인코더의 출력
        scores = self.V(torch.tanh(
            self.W1(query).unsqueeze(1) + self.W2(keys)
        ))  # [batch, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        return context, attention_weights

class LuongAttention(nn.Module):
    """루옹(곱하기) 주의"""
    def __init__(self, hidden_size, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, keys):
        if self.method == 'dot':
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        elif self.method == 'general':
            scores = torch.bmm(self.W(query).unsqueeze(1), keys.transpose(1, 2))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, keys).squeeze(1)
        return context, attention_weights


# ---------------------------------------------------------------------------
# 가린 소프트맥스 도구
# ---------------------------------------------------------------------------
# 많은 수열 과제에서 입력의 길이가 제각각이다. 주의 점수를 셈할 때
# 실제 수열 길이를 넘는 자리는 -inf로 가려서 소프트맥스가
# 그 자리에 확률 0을 주도록 해야 한다.
# 그러면 채움 토큰에 주의하지 않게 되며, 트랜스포머의 인코더 자기 주의와
# 교차 주의 모두에 꼭 필요하다.

def masked_softmax(X, valid_lens):
    """유효 길이를 넘는 자리를 가리고 소프트맥스를 한다.

    인수:
        X: 꼴이 (batch_size, num_queries, num_keys)인 3차원 텐서
        valid_lens: 1차원 텐서 (batch_size,) 또는 2차원 텐서 (batch_size, num_queries)
            성분마다 그 질의에 유효한 열쇠가 몇 개인지 알려 준다.
    반환값:
        X와 같은 꼴의 소프트맥스 출력이며 가린 자리는 0이다.
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        # 배치 원소마다 모든 질의에 같은 유효 길이를 쓴다
        valid_lens = valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    # 가림을 만든다: valid_len 이상인 자리에 -1e6을 주어 exp(-1e6) ≈ 0이 되게 한다
    X_flat = X.reshape(-1, shape[-1])
    maxlen = X_flat.size(1)
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_lens[:, None]
    X_flat[~mask] = -1e6

    return F.softmax(X_flat.reshape(shape), dim=-1)


if __name__ == "__main__":
    pass
```

## 2. 논의

BahdanauAttention은 쿼리를 $(B,H)$, 키를 $(B,S,H)$로 받아 가산 점수를 계산한다. 웨이트는 $(B,S,1)$이고 키를 값으로도 사용해 $(B,H)$ 문맥을 반환한다. LuongAttention은 $(B,1,S)$의 웨이트와 같은 형태의 문맥을 반환한다. 두 반환 웨이트의 축이 다르다.

Luong의 method는 dot 또는 general을 가정하지만 다른 문자열을 검증하지 않아 점수 변수가 정의되지 않을 수 있다. 두 모듈에는 길이·패딩 마스크 인수가 없다.

별도 masked_softmax는 주로 $(B,Q,K)$ 점수와 길이 정보를 가정한다. reshape한 텐서에 제자리 대입하므로 원래 점수도 변경될 수 있다. 매우 작은 유한 점수로 마스킹하기 때문에 길이 0인 행이 0 웨이트가 아니라 균등 분포가 되는 문제도 있다. 일반적인 임의 차원·빈 입력을 처리하는 함수로 보아서는 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
은닉 H인 BahdanauAttention의 바이어스 포함 파라메터 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    W1과 W2는 각각 $H^2+H$, V는 H+1개이므로 총 $2H^2+3H+1$개다. V의 스칼라 바이어스는 모든 위치에 같이 더해져 소프트맥스에서 상쇄되지만 등록 파라메터에는 포함된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
두 모듈에 패딩 처리를 추가한다면 어느 단계에 무엇이 필요한가?

</div>

??? success "연습문제 2 풀이"
    점수에 유효 위치 마스크를 적용한 뒤 위치 축으로 소프트맥스한다. 반환 웨이트 축이 다르므로 마스크 모양을 각각 맞춘다. 모든 위치가 패딩인 입력은 별도 처리한다. 드롭아웃을 추가하더라도 패딩 마스크를 대신하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
masked_softmax의 유한 음수 마스킹과 제자리 대입의 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"
    전부 가린 행은 모든 점수가 같아 균등 확률이 될 수 있다. reshape가 원래 저장소를 공유하면 호출자의 점수도 수정되며 자동 미분의 제자리 연산 제약을 받을 수 있다. 입력 형태와 길이를 검사하고 전체 행 마스킹의 의미를 정해야 한다.

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

가산·내적 점수는 대안이며 반환 형태와 마스킹을 맞춰야 서로 교체할 수 있다. 별도 masked_softmax의 길이 0과 제자리 변경 동작을 일반적인 안전 처리로 오해하지 않는다.
