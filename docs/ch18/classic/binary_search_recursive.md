# 이분 찾기 — 되돌이

[되풀이 이분 찾기](binary_search.md)는 `while` 되풀이로 찾을 자리를 좁힌다. 이와 같은 **되돌이** 세우기는 같은 논리를, 더 작은 부분 배열에 스스로를 부르는 함수로 나타낸다. 되돌이 판은 이분 찾기의 나누어 이기기 짜임을 드러낸다. 곧 부름마다 배열을 반으로 나누고, 한쪽 반에 되돌이해 이기며, 결과를 그대로 돌려주어 아우른다.

이 쪽에서는 되돌이 세우기를 내놓고, 짜임 귀납법으로 옳음을 증명하며, 시간과 공간 복잡도를 살피고, 되풀이 판과 견준다.

---

## 1. 되돌이로 세우기

되돌이 이분 찾기는 배열 $A$, 찾는 값 $x$, 지금의 테두리 $l$과 $r$을 매개변수로 받는다.

### 의사 코드

```
RECURSIVE-BINARY-SEARCH(A, x, l, r):
    if l > r:
        return NOT-FOUND
    m = floor((l + r) / 2)
    if A[m] == x:
        return m
    else if A[m] < x:
        return RECURSIVE-BINARY-SEARCH(A, x, m + 1, r)
    else:
        return RECURSIVE-BINARY-SEARCH(A, x, l, m - 1)
```

첫 부름은 `RECURSIVE-BINARY-SEARCH(A, x, 0, n - 1)`이다.

### 파이썬 구현

```python
def binary_search_recursive(arr, target, left=None, right=None):
    """
    정렬된 배열에서 되돌이로 값 찾기.

    매개변수
    ----------
    arr : list
        견줄 수 있는 원소를 정렬한 목록.
    target : comparable
        찾을 값.
    left : int, optional
        찾을 범위의 왼쪽 테두리(붙박이: 0).
    right : int, optional
        찾을 범위의 오른쪽 테두리(붙박이: len(arr) - 1).

    반환값
    -------
    int or None
        찾으면 그 번호, 없으면 None.
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1

    if left > right:
        return None

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

---

## 2. 짜임 귀납법으로 보는 옳음

찾을 자리의 크기 $s = r - l + 1$에 대한 강한 귀납법으로 옳음을 증명한다.

**밑 자리**($s \le 0$). $l > r$이면 찾을 자리가 비어 있다. $x$이 $A[l \,..\, r]$ 안에 있었다면 이 잔배열이 비어 있지 않았을 터이므로 `NOT-FOUND`을 돌려주는 것이 옳다.

**미루어 나아가는 걸음.** 크기가 $s$보다 작은 모든 찾을 자리에 대해 알고리즘이 옳다고 하자. 찾을 자리의 크기가 $s = r - l + 1 > 0$인 부름을 보자. $m = \lfloor (l + r) / 2 \rfloor$을 셈한다.

- $A[m] = x$이면 $m$을 돌려주는 것이 옳다.
- $A[m] < x$이면 $A$이 줄 세워져 있으므로 $x \notin A[l \,..\, m]$이다. $A[m+1 \,..\, r]$에 대한 되부름의 찾을 자리 크기는 $r - m \le s - 1 < s$이다. 미루어 세운 가정에 따라 이 부름은 옳은 답을 돌려준다.
- $A[m] > x$이면 대칭으로, $A[l \,..\, m-1]$에 대한 되부름의 찾을 자리 크기는 $m - l \le s - 1 < s$이고 미루어 세운 가정에 따라 옳다.

어느 자리에서나 알고리즘은 옳은 결과를 돌려준다. $\square$

---

## 3. 복잡도 분석

### 시간 복잡도

되돌이 이분 찾기는 다음 되돌이 관계식을 채운다

$$
T(n) = T\!\left(\frac{n}{2}\right) + O(1)
$$

밑 자리는 $T(0) = O(1)$이다. 으뜸 정리($a = 1$, $b = 2$, $f(n) = O(1) = O(n^0)$, 둘째 갈래)에 따라

$$
T(n) = O(\log n)
$$

이는 되풀이 판과 정확히 같다.

### 공간 복잡도

되부름마다 부름 쌓개에 틀이 하나씩 쌓인다. 되부름의 깊이가 $O(\log n)$이므로(부름마다 찾을 자리가 반으로 준다) 자리 복잡도는 다음과 같다.

$$
S(n) = O(\log n)
$$

이것이 덧붙은 자리를 $O(1)$만큼만 쓰는 되돌이 갈래와의 고갱이 다름이다. 참으로는 $O(\log n)$의 쌓개 깊이가 작아서($\log_2 10^9 \approx 30$ 틀 따위) 덤이 걸리는 일은 드물다.

!!! note "꼬리 부름 다듬기"
    되부름이 **꼬리 자리**에 있다. 곧 함수가 돌아가기 앞의 마지막 셈이다. 스킴이나 다듬기 깃발을 켠 어떤 C 엮개처럼 꼬리 부름 다듬기(TCO)를 받치는 말은 이 되부름을 되돌이로 바꾸어 쌓개 덤을 아주 없앨 수 있다. 파이썬은 TCO을 받치지 않으므로 $O(\log n)$의 쌓개가 그대로 든다.

---

## 4. 견줌: 되풀이와 되돌이

| 성질 | 되풀이 | 되돌이 |
|---|---|---|
| 때 복잡도 | $O(\log n)$ | $O(\log n)$ |
| 자리 복잡도 | $O(1)$ | $O(\log n)$ |
| 나누어 이기기 짜임 | 숨어 있음 | 드러남 |
| 꼬리 부름 가능 | 해당 없음 | 가능 |
| 쌓개 넘침 걱정 | 없음 | 이론상 있음($n = 10^9$일 때 깊이 $\approx 30$) |

두 판 모두 옳고 시간 복잡도가 같다. 실전 코드에서는 공간을 $O(1)$만 쓰는 되풀이 판을 대개 더 낫게 여긴다. 되돌이 판은 나누어 이기기 짜임을 이해하는 데 값지며 더 복잡한 되돌이 알고리즘의 틀이 된다.

---

## 5. 풀이 예제

$A = [3, 7, 12, 19, 25, 31, 42]$($n = 7$)에서 $x = 12$을 찾는다:

| 부름 | $l$ | $r$ | $m$ | $A[m]$ | 하는 일 |
|---|---|---|---|---|---|
| 1 | 0 | 6 | 3 | 19 | $19 > 12$, $[0, 2]$에 되돌이 |
| 2 | 0 | 2 | 1 | 7 | $7 < 12$, $[2, 2]$에 되돌이 |
| 3 | 2 | 2 | 2 | 12 | $12 = 12$, $2$을 돌려줌 |

찾는 값을 부름 3번 만에 번호 2에서 찾는다. 되돌이가 풀리며 결과 $2$이 틀마다 거슬러 올라간다.

---

## 연습문제

**연습문제 1.**
되돌이 이분 찾기의 되돌이 관계식을 쓰고 풀어라.

??? success "연습문제 1 풀이"
    $T(n) = T(n/2) + O(1)$이고 밑 자리는 $T(1) = O(1)$이다. 으뜸 정리(둘째 갈래, $a=1, b=2, k=0$)에 따라 $T(n) = O(\log n)$이다. 아니면 풀어 써도 된다. $T(n) = T(n/2) + c = T(n/4) + 2c = \cdots = T(1) + c\log_2 n = O(\log n)$. $\square$

---

**연습문제 2.**
원소가 $10^6$개인 배열에서 되부르는 이분 찾기의 가장 깊은 되부름 깊이는 얼마인가?

??? success "연습문제 2 풀이"
    되부름 깊이는 $= \lceil \log_2(10^6) \rceil = \lceil 19.93 \rceil = 20$이다. 파이썬의 기본 되부름 한도 1000에 한참 못 미친다. 원소가 $10^9$개라도 깊이는 $\lceil \log_2(10^9) \rceil = 30$뿐이다. 되부르는 이분 찾기는 참으로 쓰는 어떤 크기의 들임에서도 쌓개 넘침 걱정이 없다. $\square$

---

**연습문제 3.**
공간 복잡도와 실전 성능으로 되풀이 이분 찾기와 되돌이 이분 찾기를 견주어라.

??? success "연습문제 3 풀이"
    **되돌이**: 덧붙은 자리가 $O(1)$이다(되돌이 변수뿐). 함수를 부르는 덤도 없다. **되부름**: 되부름 틀에 쌓개 자리를 $O(\log n)$만큼 쓴다. 틀마다 함수를 부르는 덤(매개변수 넘기기, 돌아갈 자리)이 든다. 참으로는 부르는 덤이 없어 되돌이가 조금 더 빠르다. 둘 다 때는 $O(\log n)$이다. 참으로 굴리는 코드에서는 흔히 되돌이를 쓰고, 가르칠 때는 되부름이 더 또렷하다. $\square$

---

**연습문제 4.**
찾는 값이 없으면 가장 가까운 원소의 번호를 돌려주도록 되돌이 이분 찾기를 고쳐라.

??? success "연습문제 4 풀이"
    바탕 경우 `low > high` 뒤에 (쓸 수 있으면) `arr[low]`과 `arr[high]`을 찾는 값과 견주어 더 가까운 쪽의 번호를 돌려준다:

    ```python
    def closest_search(arr, target, low, high):
        if low > high:
            if low >= len(arr): return high
            if high < 0: return low
            return low if abs(arr[low] - target) <= abs(arr[high] - target) else high
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return closest_search(arr, target, mid + 1, high)
        else:
            return closest_search(arr, target, low, mid - 1)
    ```
    $\square$

## 정리하며

되부르는 이분 찾기는 나누어 다스리기 얼개를 드러내 놓고 보인다. 부름마다 찾을 자리를 반으로 나누고 한쪽에서 되부른 뒤 그 결과를 그대로 돌려준다. 때 복잡도는 되돌이 갈래와 같은 $O(\log n)$이지만 쌓개 자리를 $O(\log n)$만큼 쓴다. 옳음 밝히기는 찾을 자리의 크기에 대한 센 미루어 나아가기로 이루어진다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 2장. MIT Press.
