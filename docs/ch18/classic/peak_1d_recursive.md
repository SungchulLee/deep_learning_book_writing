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
    A peak in a 1D array $A[0..n-1]$ is an element $A[i]$ such that $A[i] \geq A[i-1]$ and $A[i] \geq A[i+1]$ (with boundary conditions $A[-1] = A[n] = -\infty$). A peak always exists because: consider the global maximum element. It is at least as large as both its neighbors, so it is a peak. More formally, any finite array has a maximum, and the maximum satisfies the peak condition.

---

**연습문제 2.**
Explain how binary search finds a peak in $O(\log n)$ time. Why is the midpoint comparison sufficient to decide which half contains a peak?

??? success "연습문제 2 풀이"
    Compare $A[\text{mid}]$ with its neighbors. If $A[\text{mid}] < A[\text{mid}+1]$, the right half (including $\text{mid}+1$) must contain a peak: either $A[\text{mid}+1]$ is a peak, or values keep increasing until they must eventually decrease (at the boundary), creating a peak. Similarly, if $A[\text{mid}] < A[\text{mid}-1]$, the left half contains a peak. If neither neighbor is larger, $A[\text{mid}]$ itself is a peak. Each comparison halves the search space, giving $O(\log n)$ time.

---

**연습문제 3.**
봉우리 찾기 알고리즘을 2차원으로 넓혀라. 나누어 이기기 방식의 시간 복잡도는 무엇인가?

??? success "연습문제 3 풀이"
    For an $n \times m$ matrix, find the maximum in the middle column ($O(n)$ time), then compare with horizontal neighbors. If a neighbor is larger, recurse on that half of the columns. Finding the column max ensures we move toward a 2D peak. Recurrence: $T(n, m) = T(n, m/2) + O(n)$, giving $O(n \log m)$. Alternatively, alternating between row and column searches gives $O(n + m)$ or $O(\max(n,m))$ depending on the strategy.

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
    The algorithm correctly handles boundary peaks by the boundary condition $A[-1] = A[n] = -\infty$.
