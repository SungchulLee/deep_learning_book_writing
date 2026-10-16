# 학습 가능한 위치 인코딩

각 위치에 대응하는 벡터를 임베딩 테이블에서 찾아 토큰 표현에 더한다. 위치 표는 과제의 로스로 학습하며 최대 위치 수를 생성자에서 정한다.

## 1. 코드

```python
"""
학습된 위치 인코딩
"""
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return x + self.position_embeddings(positions)


if __name__ == "__main__":
    pass
```

## 2. 논의

입력은 $(B,L,E)$이며 0부터 L−1까지의 위치 ID를 각 배치에 적용한다. nn.Embedding(max_len,E)의 학습 파라메터 수는 max_len×E다. 학습 중 사용되지 않은 위치 행은 해당 입력의 로스로 학습되지 않는다.

코드 블록은 torch.nn만 가져오고 forward에서 torch.arange를 사용한다. 독립적으로 사용하려면 torch 이름을 가져오는 코드가 추가로 필요하다. 원문 코드는 그대로 두었으므로 완성된 독립 실행 예제로 해석하지 않는다.

위치 ID가 테이블 범위를 넘으면 조회할 수 없다. 매 호출마다 위치가 0에서 다시 시작하므로, 캐시를 사용하는 이어쓰기에는 과거 길이를 반영한 위치 오프셋이 필요하다. 테이블 크기를 늘리거나 보간하는 것만으로 긴 문맥의 성능이 보장되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
max_len=5000, d_model=512이면 위치 임베딩의 파라메터는 몇 개인가?

</div>

??? success "연습문제 1 풀이"
    5000×512=2,560,000개다. 토큰 임베딩이나 다른 트랜스포머 레이어의 파라메터는 이 수에 포함하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 배치의 모든 표본에 같은 위치 ID를 사용해도 되는가?

</div>

??? success "연습문제 2 풀이"
    모든 표본의 첫 토큰이 같은 기준 위치에서 시작한다면 가능하다. 왼쪽 패딩이나 서로 다른 캐시 길이가 있으면 유효 토큰의 위치 규칙을 따로 정해야 한다. 패딩 마스크는 위치 임베딩과 별도로 처리한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 코드를 독립적으로 사용할 때와 긴 입력에 적용할 때 각각 확인할 문제는 무엇인가?

</div>

??? success "연습문제 3 풀이"
    torch 이름의 import가 빠져 있어 먼저 해결해야 한다. 또한 최대 위치 인덱스가 테이블 안에 있어야 하며, 피처 차원과 장치가 입력과 맞아야 한다. 길이를 늘려도 새 위치의 표현이 자동으로 충분히 학습되는 것은 아니다.

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

학습 가능한 위치 임베딩은 유한한 테이블을 조회한다. import 누락, 최대 길이와 위치 시작점의 규약을 확인하고, 새 위치를 허용하는 변경과 실제 성능 개선을 구분한다.
