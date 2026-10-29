# 시연
무작위 입력에 셀프 어텐션을 적용해 출력과 웨이트의 모양을 살펴보는 데모다. 학습·언어 이해나 번역 품질을 평가하는 예제는 아니다.

!!! note "어느 SelfAttention 인가"
    이 데모는 `SelfAttention(embed_size, heads=1)`로 부른다. `heads`를 받는 것은
    이 장의 [셀프 어텐션 코드](self_attention.md)에 있는 클래스이고,
    [앞 장의 것](../../ch11/attention/self_attention.md)은 그 인수를 받지 않는다.
    이름이 같아도 인터페이스가 다르므로 섞어 쓸 수 없다.

    예전에는 이 쪽이 앞 장의 파일을 `sys.path`로 찾아 읽어 왔는데, 그것은 `heads`를
    받지 않는 쪽이라 애초에 맞지 않는 짝이었다. 지금은 이 장의 클래스를 아래
    프로그램 안에 함께 싣는다.

## 1. 코드

```python
"""
자기 주의 데모
"""
import math

import torch
import torch.nn as nn


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

# ========================================================================
# 메인
# ========================================================================

def demo():
    # 임시 수열을 만든다
    batch_size = 2
    seq_len = 10
    embed_size = 64
    
    x = torch.randn(batch_size, seq_len, embed_size)
    
    # 자기 주의 모듈을 만든다
    attention = SelfAttention(embed_size, heads=1)
    
    # 순전파
    output, weights = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print("\n✓ Self-attention demo complete!")

if __name__ == '__main__':
    demo()
```

**출력:**

```
Input shape: torch.Size([2, 10, 64])
Output shape: torch.Size([2, 10, 64])
Attention weights shape: torch.Size([2, 10, 10])

✓ Self-attention demo complete!
```

## 2. 논의

입력은 $(2,10,64)$의 무작위 실수 텐서다. 이 장의 SelfAttention을 heads=1로 구성하면 출력은 $(2,10,64)$, 웨이트는 $(2,10,10)$이다. 단일 헤드에서 각 쿼리 위치가 열 개 키 위치를 가중합한다.

같은 입력에서 Q·K·V를 각각 투영해도 세 텐서가 같은 값이 되는 것은 아니다. 또한 이 데모는 위치 인코딩·인과 마스크·로스·옵티마이저를 포함하지 않는다. 무작위 초기화의 웨이트 모양을 확인하는 일과 학습된 언어적 관계를 해석하는 일은 구분한다.

앞 장 모듈을 가져오는 안내와 heads=1 호출 사이의 생성자 불일치를 먼저 확인해야 한다. 가져오기 오류와 모델 계산 오류를 나누어 진단한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 입력·출력과 웨이트 모양을 설명하라.

</div>

??? success "풀이"
    이 장의 단일 헤드 구현 기준으로 입력·출력은 (2,10,64), 웨이트는 (2,10,10)이다. 웨이트의 두 위치 축은 각각 쿼리와 키를 뜻하며 마지막 축에서 정규화한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
무작위 어텐션 웨이트가 단어 사이의 의미 관계를 보여 주는가?

</div>

??? success "풀이"
    입력도 웨이트도 학습된 언어 표현이 아니므로 그렇게 해석할 수 없다. 학습 후의 어텐션도 예측의 완전한 인과 설명은 아니다. 여기서는 축과 계산 규약을 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
가져오기 경로를 수정했는데 heads 인수 오류가 나면 무엇을 확인해야 하는가?

</div>

??? success "풀이"
    실제로 가져온 SelfAttention의 생성자와 해당 모듈 경로를 확인한다. 앞 장의 클래스는 d_model과 선택적 d_k·d_v 등을 받고 heads를 받지 않는다. 클래스 이름이 같아도 인터페이스가 같지는 않다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_demo():
        model = Demo(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

이 데모는 셀프 어텐션의 입출력 모양을 살피는 예제다. 먼저 가져오는 클래스의 인터페이스를 맞춰야 하며, 무작위 텐서의 결과를 학습 성능이나 언어적 해석으로 확장하지 않는다.
