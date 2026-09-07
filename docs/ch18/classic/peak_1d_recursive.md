# 1차원 봉우리 — 되돌이
```python
def compare(left, center, right):
    if left > center: # 왼쪽이 중앙보다 크면 왼쪽 영역 (left 포함) 에서 찿는다.
        return 'left'
    elif right > center: # 오른쪽이 중앙보다 크면 오른쪽 영역 (right 포함) 에서 찿는다.
        return 'right'
    else: # 왼쪽과 오른쪽 모두 중앙보다 작거나 같으면 center가 찿고자하는 1D Peak 이다.
        return 'center'
```

```python
def peak(lst):
    
    # 바닥 경우
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    elif (len(lst) == 2) and (lst[0] >= lst[1]):
        return lst[0] 
    elif len(lst) == 2:
        return lst[1] 
    
    # 되돌이
    i = int(len(lst)/2) # 가운데 번호
    left = lst[i-1]
    center = lst[i]
    right = lst[i+1]
    result = compare(left,center, right)
    
    if result == 'left':
        return peak(lst[:i]) 
    elif result == 'right':
        return peak(lst[i+1:])
    else:
        return center
```

```python
#lst = [1,2,3,4,5]
#lst = [5,4,3,2,1]
lst = [1,2,3,4,5,4,3,2,1]
peak(lst)
```

**출력:**
```
5
```

# 참고 문헌

[1. Algorithmic Thinking, Peak Finding](https://www.youtube.com/watch?v=HtSuA80QTyo&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=1)

## 연습문제

**연습문제 1.**
1차원 배열의 봉우리 찾기 문제를 정의하고 봉우리가 늘 있는 까닭을 밝혀라.

??? success "연습문제 1 풀이"
    1차원 배열 $A[0..n-1]$에서 봉우리란 $A[i] \geq A[i-1]$이고 $A[i] \geq A[i+1]$인 원소 $A[i]$을 말한다(가장자리 조건은 $A[-1] = A[n] = -\infty$). 봉우리는 늘 있다. 온 배열에서 가장 큰 원소를 보면 두 이웃보다 작지 않으므로 봉우리다. 좀 더 엄밀히 말하면, 마디 있는 배열에는 반드시 가장 큰 값이 있고 그 값이 봉우리 조건을 채운다.

---

**연습문제 2.**
이분 찾기가 어떻게 $O(\log n)$ 때에 봉우리를 찾는지 밝혀라. 가운데 자리를 견주는 것만으로 어느 쪽 반에 봉우리가 있는지 정할 수 있는 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    $A[\text{mid}]$을 그 이웃과 견준다. $A[\text{mid}] < A[\text{mid}+1]$이면 오른쪽 반($\text{mid}+1$을 넣어)에 봉우리가 반드시 있다. $A[\text{mid}+1]$이 봉우리이거나, 값이 계속 오르다가 마침내 가장자리에서 내려가면서 봉우리를 이루기 때문이다. 마찬가지로 $A[\text{mid}] < A[\text{mid}-1]$이면 왼쪽 반에 봉우리가 있다. 두 이웃이 모두 크지 않으면 $A[\text{mid}]$ 자신이 봉우리다. 견줌마다 찾을 자리가 반으로 주므로 $O(\log n)$ 때가 든다.

---

**연습문제 3.**
봉우리 찾기 알고리즘을 2차원으로 넓혀라. 나누어 이기기 방식의 시간 복잡도는 무엇인가?

??? success "연습문제 3 풀이"
    $n \times m$ 행렬에서는 가운데 열의 가장 큰 값을 찾고($O(n)$ 때) 좌우 이웃과 견준다. 이웃이 더 크면 그쪽 절반의 열에서 되부른다. 열의 가장 큰 값을 잡으면 2차원 봉우리 쪽으로 나아감이 보장된다. 되돌이 식은 $T(n, m) = T(n, m/2) + O(n)$이므로 $O(n \log m)$이다. 아니면 행 찾기와 열 찾기를 번갈아 하면 꾀에 따라 $O(n + m)$이나 $O(\max(n,m))$이 된다.

---

**연습문제 4.**
봉우리 찾기 문제에 이분 찾기를 짜고 모서리 경우(테두리의 봉우리, 평평한 곳)를 올바로 다루는지 확인하여라.

??? success "연습문제 4 풀이"
    ```python
    def find_peak(arr):
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < arr[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # 시험
    assert find_peak([1, 3, 2]) == 1
    assert find_peak([5, 4, 3, 2, 1]) == 0  # 왼쪽 테두리의 봉우리
    assert find_peak([1, 2, 3, 4, 5]) == 4  # 오른쪽 테두리의 봉우리
    assert find_peak([1, 2, 2, 2, 1]) in [1, 2, 3]  # 평평한 곳
    ```
    가장자리 조건 $A[-1] = A[n] = -\infty$ 덕에 알고리즘은 가장자리 봉우리도 옳게 다룬다.
