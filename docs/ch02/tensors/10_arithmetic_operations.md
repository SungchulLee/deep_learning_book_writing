# 산술 연산

이 스크립트는 텐서의 산술 연산을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""
튜토리얼 10: 셈 연산
===================================

PyTorch에서 원소별 셈과 텐서 셈을 익힌다.

핵심 개념:
- 원소별 셈(+, -, *, /, **)
- 제자리 셈(add_, mul_ 따위)
- 수학 함수(sqrt, exp, log 따위)
- 모으기와 원소별 셈 견주기
- 펴 맞추기 기초
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # -------------------------------------------------------------------------
    # 1. 기본 원소별 산술
    # -------------------------------------------------------------------------
    header("1. Basic Element-wise Arithmetic")
    
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    # 덧셈
    c_add = a + b
    print(f"a + b = {c_add}")  # tensor([6., 8., 10., 12.])
    
    # 또한: torch.add(a, b)
    c_add_fn = torch.add(a, b)
    print(f"torch.add(a, b) = {c_add_fn}")
    
    # 뺄셈
    c_sub = a - b
    print(f"\na - b = {c_sub}")  # tensor([-4., -4., -4., -4.])
    
    # 곱셈(원소별이며 행렬 곱이 아니다)
    c_mul = a * b
    print(f"a * b = {c_mul}")  # tensor([5., 12., 21., 32.])
    
    # 나눗셈
    c_div = b / a
    print(f"b / a = {c_div}")  # tensor([5., 3., 2.3333, 2.])
    
    # 내림 나눗셈
    c_floordiv = b // a
    print(f"b // a = {c_floordiv}")  # tensor([5., 3., 2., 2.])
    
    # 나머지 연산
    c_mod = b % a
    print(f"b % a = {c_mod}")  # tensor([0., 0., 1., 0.])
    
    # 거듭제곱
    c_pow = a ** 2
    print(f"a ** 2 = {c_pow}")  # tensor([1., 4., 9., 16.])
    
    # -------------------------------------------------------------------------
    # 2. 제자리 연산(메모리에서 텐서를 직접 수정)
    # -------------------------------------------------------------------------
    header("2. In-place Operations")
    
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Original x = {x}")
    print(f"Memory address: {id(x)}")
    
    # 제자리 연산은 밑줄(_)로 끝난다
    x.add_(10)  # x = x + 10
    print(f"After x.add_(10) = {x}")
    print(f"Memory address: {id(x)}")  # Same address!
    
    x.mul_(2)  # x = x * 2
    print(f"After x.mul_(2) = {x}")
    
    x.div_(4)  # x = x / 4
    print(f"After x.div_(4) = {x}")
    
    # 주의: 경사가 필요한 텐서에 제자리 연산을 하면 오류가 날 수 있다!
    # y = torch.tensor([1.0], requires_grad=True)
    # y.add_(1)  # RuntimeError: 경사를 가진 잎 변수에는 제자리 연산을 할 수 없다
    
    # -------------------------------------------------------------------------
    # 3. 스칼라 연산
    # -------------------------------------------------------------------------
    header("3. Scalar Operations")
    
    vec = torch.tensor([1, 2, 3, 4, 5])
    print(f"vec = {vec}")
    
    # 스칼라는 자동으로 브로드캐스팅된다
    vec_plus_10 = vec + 10
    print(f"vec + 10 = {vec_plus_10}")
    
    vec_times_2 = vec * 2
    print(f"vec * 2 = {vec_times_2}")
    
    vec_pow_2 = vec ** 2
    print(f"vec ** 2 = {vec_pow_2}")
    
    # -------------------------------------------------------------------------
    # 4. 수학 함수
    # -------------------------------------------------------------------------
    header("4. Mathematical Functions")
    
    x = torch.tensor([0.0, 1.0, 4.0, 9.0])
    print(f"x = {x}\n")
    
    # 제곱근
    sqrt_x = torch.sqrt(x)
    print(f"sqrt(x) = {sqrt_x}")
    
    # 지수함수
    exp_x = torch.exp(x)
    print(f"exp(x) = {exp_x}")
    
    # 자연로그(밑이 e인 로그)
    x_pos = torch.tensor([1.0, 2.718, 7.389])
    log_x = torch.log(x_pos)
    print(f"\nlog({x_pos}) = {log_x}")
    
    # 밑이 10인 로그
    log10_x = torch.log10(x_pos)
    print(f"log10({x_pos}) = {log10_x}")
    
    # 절댓값
    x_neg = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
    abs_x = torch.abs(x_neg)
    print(f"\nabs({x_neg}) = {abs_x}")
    
    # 부호 함수
    sign_x = torch.sign(x_neg)
    print(f"sign({x_neg}) = {sign_x}")
    
    # 반올림 연산
    x_float = torch.tensor([1.2, 2.5, -3.7, 4.9])
    print(f"\nx = {x_float}")
    print(f"round(x) = {torch.round(x_float)}")
    print(f"floor(x) = {torch.floor(x_float)}")
    print(f"ceil(x) = {torch.ceil(x_float)}")
    print(f"trunc(x) = {torch.trunc(x_float)}")  # Remove decimal part
    
    # -------------------------------------------------------------------------
    # 5. 삼각함수
    # -------------------------------------------------------------------------
    header("5. Trigonometric Functions")
    
    angles = torch.tensor([0.0, torch.pi/4, torch.pi/2, torch.pi])
    print(f"angles = {angles}")
    
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    tan_angles = torch.tan(angles)
    
    print(f"sin(angles) = {sin_angles}")
    print(f"cos(angles) = {cos_angles}")
    print(f"tan(angles) = {tan_angles}")
    
    # 역삼각함수
    values = torch.tensor([0.0, 0.5, 1.0])
    print(f"\nvalues = {values}")
    print(f"arcsin(values) = {torch.asin(values)}")
    print(f"arccos(values) = {torch.acos(values)}")
    print(f"arctan(values) = {torch.atan(values)}")
    
    # -------------------------------------------------------------------------
    # 6. 자르기와 범위 제한
    # -------------------------------------------------------------------------
    header("6. Clipping and Clamping")
    
    x = torch.tensor([-5.0, -2.0, 0.0, 3.0, 10.0])
    print(f"x = {x}")
    
    # 값을 [min, max] 범위로 제한
    clamped = torch.clamp(x, min=-3.0, max=5.0)
    print(f"clamp(x, -3, 5) = {clamped}")  # [-3., -2., 0., 3., 5.]
    
    # 최솟값만
    clamped_min = torch.clamp(x, min=0.0)
    print(f"clamp(x, min=0) = {clamped_min}")  # ReLU-like behavior
    
    # 최댓값만
    clamped_max = torch.clamp(x, max=2.0)
    print(f"clamp(x, max=2) = {clamped_max}")
    
    # -------------------------------------------------------------------------
    # 7. 비교 연산
    # -------------------------------------------------------------------------
    header("7. Comparison Operations")
    
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    
    print(f"a = {a}")
    print(f"b = {b}\n")
    
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    print(f"a > b: {a > b}")
    print(f"a >= b: {a >= b}")
    print(f"a < b: {a < b}")
    print(f"a <= b: {a <= b}")
    
    # 원소별 최댓값/최솟값
    print(f"\ntorch.max(a, b) (element-wise): {torch.max(a, b)}")
    print(f"torch.min(a, b) (element-wise): {torch.min(a, b)}")
    
    # -------------------------------------------------------------------------
    # 8. 행렬 연산(2차원 텐서)
    # -------------------------------------------------------------------------
    header("8. Matrix Operations")
    
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print(f"A =\n{A}\n")
    print(f"B =\n{B}\n")
    
    # 원소별 곱
    C_elem = A * B
    print(f"A * B (element-wise) =\n{C_elem}")
    
    # 행렬 곱
    C_matmul = A @ B  # or torch.matmul(A, B)
    print(f"\nA @ B (matrix multiplication) =\n{C_matmul}")
    
    # 또한: 2차원 행렬 곱에는 torch.mm()
    C_mm = torch.mm(A, B)
    print(f"torch.mm(A, B) =\n{C_mm}")
    
    # -------------------------------------------------------------------------
    # 9. 축약 연산
    # -------------------------------------------------------------------------
    header("9. Reduction Operations")
    
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    print(f"x =\n{x}\n")
    
    # 모든 원소의 합
    total = torch.sum(x)
    print(f"sum(x) = {total}")
    
    # 0번 차원을 따라 합(행을 접는다)
    sum_dim0 = torch.sum(x, dim=0)
    print(f"sum(x, dim=0) = {sum_dim0}")  # [5., 7., 9.]
    
    # 1번 차원을 따라 합(열을 접는다)
    sum_dim1 = torch.sum(x, dim=1)
    print(f"sum(x, dim=1) = {sum_dim1}")  # [6., 15.]
    
    # 평균
    mean_all = torch.mean(x)
    print(f"\nmean(x) = {mean_all}")
    
    mean_dim0 = torch.mean(x, dim=0)
    print(f"mean(x, dim=0) = {mean_dim0}")
    
    # 최솟값과 최댓값
    print(f"\nmin(x) = {torch.min(x)}")
    print(f"max(x) = {torch.max(x)}")
    
    # argmin과 argmax(인덱스를 반환)
    print(f"argmin(x) = {torch.argmin(x)}")  # Flattened index
    print(f"argmax(x) = {torch.argmax(x)}")
    
    # -------------------------------------------------------------------------
    # 10. 흔한 패턴과 요령
    # -------------------------------------------------------------------------
    header("10. Common Patterns and Tips")
    
    print("""
    핵심 학습:
    
    1. **원소별 셈**
       - 셈 기호(+, -, *, /)는 대개 원소별로 움직인다
       - 행렬 곱에는 @이나 torch.matmul()을 써라
    
    2. **제자리 셈**
       - 밑줄로 끝난다: add_(), mul_() 따위
       - 기억 자리에서 텐서를 고친다(새 텐서를 만들지 않는다)
       - requires_grad=True인 텐서에는 쓸 수 없다
    
    3. **Broadcasting**
       - 홑값은 텐서 꼴에 절로 펴 맞춰진다
       - 자세한 펴 맞추기 규칙은 튜토리얼 11을 보아라
    
    4. **함수와 방법 견주기**
       - torch.add(a, b) == a.add(b) == a + b
       - 코드가 가장 읽기 좋은 것을 써라
    
    5. **Performance**
       - 제자리 셈은 기억 자리를 아끼지만 기울기에 조심하라
       - 더 잘 다듬어질 수 있도록 torch.* 함수를 써라
    """)
    
    # -------------------------------------------------------------------------
    # 연습 문제
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    다음을 해 보아라.
    
    1. x = [0, 1, 2, 3, 4]에 대해 (x^2 + 2*x + 1)을 셈하여라
    2. 값을 [0, 1] 범위로 맞추어라: (x - min) / (max - min)
    3. 벡터의 L2 노름(유클리드 길이)을 셈하여라
    4. 텐서 셋의 원소별 최댓값
    5. 시그모이드 함수: 1 / (1 + exp(-x))
    """)
    
    # 해답
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    ex1 = x**2 + 2*x + 1
    print(f"\n1. (x^2 + 2*x + 1) = {ex1}")
    
    x2 = torch.tensor([3.0, 5.0, 1.0, 9.0])
    ex2 = (x2 - x2.min()) / (x2.max() - x2.min())
    print(f"2. Normalized = {ex2}")
    
    vec = torch.tensor([3.0, 4.0])
    ex3 = torch.sqrt(torch.sum(vec ** 2))
    print(f"3. L2 norm = {ex3}")
    
    t1 = torch.tensor([1, 5, 3])
    t2 = torch.tensor([2, 4, 6])
    t3 = torch.tensor([3, 3, 3])
    ex4 = torch.max(torch.max(t1, t2), t3)
    print(f"4. Element-wise max = {ex4}")
    
    x5 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    ex5 = 1 / (1 + torch.exp(-x5))
    print(f"5. Sigmoid = {ex5}")


if __name__ == "__main__":
    main()```

## 2. 논의

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

## 연습문제

**연습문제 1.**
모양이 $(3, 1, 5)$와 $(4, 5)$인 텐서가 주어졌을 때 덧셈 후의 출력 모양을 구하고 PyTorch로 확인하라.

??? success "연습문제 1 풀이"
    ```python
    import torch
    a = torch.randn(3, 1, 5)
    b = torch.randn(4, 5)
    c = a + b
    print(c.shape)  # torch.Size([3, 4, 5])
    # b는 암묵적으로 (1, 4, 5)가 되고, 그다음 둘 다 (3, 4, 5)로 브로드캐스팅된다.
    ```

---


**연습문제 2.**
모양이 $(3, 4)$와 $(5,)$인 텐서의 덧셈이 실패하는 이유를 설명하라. 이 불일치를 고치는 서로 다른 두 가지 방법을 제안하라.

??? success "연습문제 2 풀이"
    마지막 차원이 4와 5인데 서로 같지도 않고 1도 아니다. 해결 1: 벡터를 열에 맞도록 바꾼다. 예를 들어 $(4,)$ 벡터를 쓴다. 해결 2: 벡터를 $(5, 1)$로 바꾸고 행렬을 $(5, 4)$로 바꾸어 올바른 축을 따라 브로드캐스팅이 되게 한다.

---


**연습문제 3.**
`torch.outer`를 쓰지 않고 브로드캐스팅으로 두 벡터 $a = [1, 2, 3]$과 $b = [4, 5]$의 외적을 계산하라.

??? success "연습문제 3 풀이"
    ```python
    a = torch.tensor([1., 2., 3.]).unsqueeze(1)  # (3, 1)
    b = torch.tensor([4., 5.]).unsqueeze(0)       # (1, 2)
    outer = a * b  # (3, 2)
    print(outer)
    # tensor([[ 4.,  5.],
    #         [ 8., 10.],
    #         [12., 15.]])
    ```

## 정리하며

**다룬 것** — 산술 연산

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
