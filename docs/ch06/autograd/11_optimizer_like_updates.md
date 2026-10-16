# 옵티마이저처럼 파라미터 업데이트하기

옵티마이저 없이 텐서를 직접 업데이트하며 `no_grad()`의 역할을 확인한다. 경사를 계산하는 단계와 경사로 값을 바꾸는 단계를 구분한다.

## 1. 코드

```python
"""최적화기처럼 고치기."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    w = torch.randn(3, requires_grad=True)
    lr = 0.1
    for step in range(3):
        # ---------------- 순전파 ----------------
        # 간단한 이차 로스 = 제곱합.
        # w.requires_grad=True이므로 이 로스 텐서도 경사를 필요로 한다
        # grad_fn(<SumBackward0>)을 가지므로 autograd가 backward()를 호출할 때
        # ∂loss/∂w를 계산할 줄 안다.
        loss = (w ** 2).sum()

        # ---------------- 경사 초기화 ----------------
        # 기본적으로 PyTorch는 .grad 버퍼에 경사를 누적한다.
        # 따라서 backward()를 다시 호출하기 전에 명시적으로 지워야 한다.
        # w.grad.zero_()는 안전하다:
        #   - .grad는 버퍼 텐서일 뿐 계산 그래프의 일부가 아니다.
        #   - .grad에 대한 제자리 연산은 autograd를 혼란시키지 않는다.
        #   - 여기서는 torch.no_grad()가 필요 없다.
        if w.grad is not None:
            w.grad.zero_()

        # ---------------- 역전파 ----------------
        # d(loss)/dw를 계산하여 결과를 w.grad에 저장한다.
        # 내부적으로는 loss.grad_fn을 통해 그래프를 훑으며 이루어진다.
        loss.backward()

        # ---------------- 파라메터 갱신 ----------------
        # ❌ 잘못됨: w -= lr * w.grad   (no_grad 밖에서)
        #   - 그래프의 일부로 기록될 것이다(grad_fn=<SubBackward0>).
        #   - w가 "잎" 텐서가 아니게 되고 .grad가 더 이상 올바르게 갱신되지 않는다.
        #   - 그래프가 매 단계 커진다 → 메모리 누수.
        #
        # ✅ 올바름: torch.no_grad()로 감싼다
        #   - 경사 추적을 일시적으로 끈다.
        #   - 제자리 갱신이 일어나지만 그래프에서는 제외된다.
        #   - w는 requires_grad=True인 잎 텐서로 남는다.
        with torch.no_grad():
            w -= lr * w.grad

        # 형식 지정(:.6f)이 동작하도록 스칼라 로스에는 .item()을 쓴다.
        print(f"step {step} | loss={loss.item():.6f} | w={w}")

    # ---------------- 학습 후 참고 ----------------
    # - w.requires_grad는 여전히 True이다. w는 여전히 학습 가능하다.
    # - no_grad() 때문에 autograd가 그 갱신들을 건너뛰었다.
    # - torch.optim.SGD와 Adam이 .step()을 구현하는 방식이 정확히 이것이다.
    print("Final w.requires_grad (still True, updates not tracked):", w.requires_grad)

if __name__ == "__main__":
    main()
```

**출력:**

```
step 0 | loss=4.378094 | w=tensor([ 1.3805, -0.2067,  0.9239], requires_grad=True)
step 1 | loss=2.801980 | w=tensor([ 1.1044, -0.1653,  0.7391], requires_grad=True)
step 2 | loss=1.793267 | w=tensor([ 0.8835, -0.1323,  0.5913], requires_grad=True)
Final w.requires_grad (still True, updates not tracked): True
```

## 2. 논의

### 제곱합에 대한 업데이트

로스 $L=\sum_i w_i^2$의 경사는 $2w$이다. 러닝레이트가 0.1이므로 한 번의 업데이트는 $w\leftarrow w-0.1(2w)=0.8w$가 된다. 이 예제에서는 각 성분이 0에 가까워지는 방향으로 이동한다.

### 업데이트 연산을 기록하지 않는 이유

`w`는 경사를 추적하는 잎 텐서이므로, 일반적인 추적 상태에서 직접 제자리 수정하면 오류가 발생할 수 있다. 예제는 `torch.no_grad()` 안에서 값을 바꾸어 업데이트 자체를 미분 경로에 넣지 않는다. `w.requires_grad`는 그대로 유지되어 다음 반복의 로스는 다시 미분할 수 있다.

매 반복에서 로스를 새로 계산하고 이전 경사를 지운다. 출력에 나오는 로스는 업데이트 전 값이고 `w`는 업데이트 후 값이다. 일반적인 `torch.optim` 옵티마이저도 경사 계산과 별도로 파라미터 업데이트를 담당한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
로스가 제곱합이고 러닝레이트가 0.1일 때 한 업데이트에서 파라미터가 어떻게 바뀌는지 유도하라.

</div>

??? success "연습문제 2 풀이"
    경사가 $2w$이므로 새 값은 $w-0.1(2w)=0.8w$다. 예제는 이 연산을 `no_grad()` 안에서 수행한다. `w.requires_grad`는 유지되므로 다음 로스는 다시 미분할 수 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```

## 정리하며

경사는 추적 상태에서 계산하고, 보통의 학습 업데이트는 기록을 끈 상태에서 수행한다. 다음 반복에는 갱신한 파라미터로 로스를 새로 만든다.
