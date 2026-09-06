# 층 출력하기

이 스크립트는 모델의 층을 출력하는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""Print layers."""
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
    main()```

## 논의

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

## 연습문제

**연습문제 1.**
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

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


**연습문제 2.**
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


**연습문제 3.**
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```
