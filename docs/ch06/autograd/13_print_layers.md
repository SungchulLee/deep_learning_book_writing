# 모델 계층 출력하기

모델의 모듈 구조와 파라미터 모양을 확인하는 방법을 살펴본다. 바로 아래 계층을 보는 경우와 전체 모듈 트리를 보는 경우를 구분한다.

## 1. 코드

```python
"""층을 찍어 본다."""
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def main():
    # 아주 작은 모델: Linear → ReLU → Linear
    model = nn.Sequential(
        nn.Linear(5, 3),  # index 0
        nn.ReLU(),        # index 1
        nn.Linear(3, 1)   # index 2
    )
    # PyTorch의 nn.Linear(in_features, out_features)는 다음을 저장한다:
    #   • 가중치: (out_features, in_features)
    #   • 편향  : (out_features,)
    # 순전파가 계산한다: output = input @ weight.T + bias

    # 0) Sequential이라면 인덱스로 나열하는 것이 가장 빠르다
    if isinstance(model, nn.Sequential):
        print("== Sequential index listing ==")
        for i, layer in enumerate(model):
            print(f"[{i}] {layer}")
        print()

    # 1) 최상위 자식 모듈(어떤 nn.Module에서도 동작한다)
    print("== Top-level children ==")
    for name, layer in model.named_children():
        print(f"{name}: {layer}")
    print()

    # 2) 전체 모듈 트리(중첩된 하위 모듈 포함. 이름 ''은 뿌리이다)
    print("== Full module tree ==")
    for name, layer in model.named_modules():
        print(f"{name or '<root>'}: {layer}")
    print()

    # 3) 필요한 정보만: Linear의 가중치/편향 모양
    print("== Linear shapes ==")
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            W = tuple(layer.weight.shape)  # (out_features, in_features)
            b = tuple(layer.bias.shape) if layer.bias is not None else None
            print(f"{name}: Linear weight {W}, bias {b}")
    print()

if __name__ == "__main__":
    main()
```

**출력:**

```
== Sequential index listing ==
[0] Linear(in_features=5, out_features=3, bias=True)
[1] ReLU()
[2] Linear(in_features=3, out_features=1, bias=True)

== Top-level children ==
0: Linear(in_features=5, out_features=3, bias=True)
1: ReLU()
2: Linear(in_features=3, out_features=1, bias=True)

== Full module tree ==
<root>: Sequential(
  (0): Linear(in_features=5, out_features=3, bias=True)
  (1): ReLU()
  (2): Linear(in_features=3, out_features=1, bias=True)
)
0: Linear(in_features=5, out_features=3, bias=True)
1: ReLU()
2: Linear(in_features=3, out_features=1, bias=True)

== Linear shapes ==
0: Linear weight (3, 5), bias (3,)
2: Linear weight (1, 3), bias (1,)
```

## 2. 논의

### 계층을 조회하는 범위

`nn.Sequential`을 순회하면 순서대로 등록된 계층을 얻는다. `named_children()`은 바로 아래 자식 모듈만 반환하고, `named_modules()`는 루트 자신을 포함한 중첩 모듈을 조회한다. 따라서 빈 이름으로 나타나는 항목은 모델 자체다. 이는 등록된 구조를 보여주며, 모든 모델의 실제 실행 경로를 완전히 설명하는 도구는 아니다.

### Linear의 파라미터 모양

`nn.Linear(in_features, out_features)`의 가중치 모양은 $(\text{out\_features},\text{in\_features})$이다. 예제의 두 선형 계층은 각각 $(3,5)$, $(1,3)$ 가중치를 가진다. 배치 입력은 $XW^\top+b$로 변환되므로, 입력 마지막 차원이 `in_features`와 일치해야 한다.

ReLU처럼 학습할 파라미터가 없는 모듈도 구조에 포함된다. 계층 목록과 파라미터 목록은 같은 개념이 아니며, 중첩 모델을 점검할 때 둘을 함께 확인하면 모양 오류를 찾는 데 도움이 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    r1 = A @ B
    r2 = torch.mm(A, B)
    r3 = torch.matmul(A, B)
    print(torch.allclose(r1, r2) and torch.allclose(r2, r3))  # True
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

</div>

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```

## 정리하며

직접 자식은 `named_children()`, 중첩 구조는 `named_modules()`, 학습 파라미터는 `named_parameters()`로 살펴본다. 선형 계층의 가중치 축 순서도 함께 확인한다.
