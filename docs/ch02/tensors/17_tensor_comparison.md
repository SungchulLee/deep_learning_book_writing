# 텐서 비교 - 원소별 비교와 텐서 비교

이 스크립트는 텐서 비교, 즉 원소별 비교와 텐서 전체의 비교을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""학습 17: 텐서 견주기 - 원소별 견줌과 텐서 견줌"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Element-wise Comparison")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    print(f"a = {a}\nb = {b}\n")
    print(f"a > b: {a > b}")
    print(f"a >= b: {a >= b}")
    print(f"a < b: {a < b}")
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    
    header("2. Tensor Equality - torch.equal()")
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2, 3])
    z = torch.tensor([1, 2, 4])
    print(f"x = {x}\ny = {y}\nz = {z}\n")
    print(f"torch.equal(x, y): {torch.equal(x, y)}")  # True
    print(f"torch.equal(x, z): {torch.equal(x, z)}")  # False
    print("\nNote: equal() requires EXACT match")
    
    header("3. Approximate Equality - torch.allclose()")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0001, 2.0001, 3.0001])
    print(f"a = {a}\nb = {b}\n")
    print(f"equal(): {torch.equal(a, b)}")  # False
    print(f"allclose() default: {torch.allclose(a, b)}")  # True
    print(f"allclose(atol=1e-5): {torch.allclose(a, b, atol=1e-5)}")  # False
    print(f"allclose(atol=1e-3): {torch.allclose(a, b, atol=1e-3)}")  # True
    
    header("4. Finding Matches - torch.eq()")
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[1, 0, 3], [4, 0, 6]])
    print(f"a =\n{a}\nb =\n{b}\n")
    matches = torch.eq(a, b)
    print(f"Element-wise equality:\n{matches}")
    num_matches = matches.sum().item()
    print(f"Number of matching elements: {num_matches}")
    
    header("5. Top-k and Sorting")
    scores = torch.tensor([3.2, 1.5, 4.7, 2.1, 5.3])
    print(f"Scores: {scores}")
    top_k_values, top_k_indices = torch.topk(scores, k=3)
    print(f"Top 3 values: {top_k_values}")
    print(f"Top 3 indices: {top_k_indices}")
    sorted_values, sorted_indices = torch.sort(scores, descending=True)
    print(f"\nSorted (descending): {sorted_values}")
    print(f"Sorted indices: {sorted_indices}")
    
    header("6. Element-wise Max/Min")
    a = torch.tensor([1, 5, 3])
    b = torch.tensor([2, 4, 6])
    print(f"a = {a}\nb = {b}\n")
    max_elem = torch.max(a, b)
    min_elem = torch.min(a, b)
    print(f"Element-wise max: {max_elem}")
    print(f"Element-wise min: {min_elem}")
    
    header("7. Practical: Finding Best Predictions")
    logits = torch.randn(5, 10)  # 5 samples, 10 classes
    print(f"Logits shape: {logits.shape}")
    predictions = torch.argmax(logits, dim=1)
    print(f"Predicted classes: {predictions}")
    max_scores, _ = torch.max(logits, dim=1)
    print(f"Max scores: {max_scores}")

if __name__ == "__main__":
    main()```

## 논의

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

`sum()`, `mean()`, `min()`, `max()` 같은 축약 연산은 텐서의 하나 이상의 차원을 접는다. `dim` 매개변수가 어느 차원을 축약할지 지정하며, `keepdim=True`는 축약된 차원을 크기 1로 남겨 둔다. 이는 이후 연산에서 올바른 브로드캐스팅을 위해 필수적이다.

## 연습문제

**연습문제 1.**
`torch.topk`를 사용하여 텐서 `[3, 1, 4, 1, 5, 9, 2, 6, 5]`에서 상위 3개의 값과 그 인덱스를 찾아라.

??? success "연습문제 1 풀이"
    ```python
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    values, indices = torch.topk(x, k=3)
    print(f'Top 3 values: {values}')   # tensor([9, 6, 5])
    print(f'Top 3 indices: {indices}') # tensor([5, 7, 4])
    ```

---


**연습문제 2.**
$(3, 4)$ 행렬을 각 행을 따라(dim=1) 정렬하고 반환된 인덱스가 무엇을 나타내는지 설명하라.

??? success "연습문제 2 풀이"
    ```python
    m = torch.randint(0, 10, (3, 4))
    values, indices = torch.sort(m, dim=1)
    # `indices[i, j]`는 원래 i번 행의 어느 열이 정렬 후
    # j번 위치에 오게 되었는지를 알려준다.
    # 이 인덱스를 gather()와 함께 쓰면 복원할 수 있다.
    ```

---


**연습문제 3.**
상위 k 정확도를 구현하라. 모양 $(N, C)$인 로짓과 모양 $(N,)$인 레이블이 주어졌을 때, 참 레이블이 상위 3개 예측에 들어 있는 비율을 계산하라.

??? success "연습문제 3 풀이"
    ```python
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    _, top3 = torch.topk(logits, k=3, dim=1)
    correct = (top3 == labels.unsqueeze(1)).any(dim=1)
    accuracy = correct.float().mean()
    print(f'Top-3 accuracy: {accuracy:.2%}')
    ```
