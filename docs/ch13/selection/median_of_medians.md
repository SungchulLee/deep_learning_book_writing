# 중앙값의 중앙값

빠른 고르기의 최악의 경우 $O(n^2)$ 은 축을 잘못 고르는 데서 온다. 축이 늘 극단적인 순위에 떨어지면 원소가 거의 다 다음 되돌이 부름까지 살아남는다. **중앙값의 중앙값**은 고른 축의 순위가 양 끝에서 일정 몫만큼 떨어져 있음을 보장하는 축 고르기 기법이며, 그 덕분에 최악의 경우에도 $O(n)$ 고르기가 된다. 1973년 Blum, Floyd, Pratt, Rivest, Tarjan이 들여왔다.

## 1. 축을 어떻게 고를 것인가

**정의 1.** [중앙값의 중앙값 축]

원소 $n$ 개의 배열 $A$ 에서 다음 절차로 고른 값을 **중앙값의 중앙값** 축이라 한다.

1. **묶기** — $A$ 를 원소 $5$ 개씩 $\lceil n/5\rceil$ 개 무리로 나눈다(마지막 무리는 더 작을 수 있다).
2. **무리마다 정렬** — 끼워넣기 정렬을 쓴다(원소 $5$ 개에 많아야 $6$ 번 견준다).
3. **중앙값 뽑기** — 정렬한 무리마다 가운데 원소를 가져온다. 중앙값이 $\lceil n/5\rceil$ 개 나온다.
4. **되돌이** — 고르기 알고리즘을 되돌이로 써서 이 중앙값들의 중앙값을 찾는다.

### 정리 1. 축의 질 보장 — 순위가 양 끝에서 $3n/10$ 만큼 떨어져 있다 { .thm }

중앙값의 중앙값으로 고른 축 $m$ 에 대해, $m$ 이하인 원소와 $m$ 이상인 원소가 각각 적어도

$$
3\left\lceil \frac12\left\lceil \frac n5\right\rceil\right\rceil \;\geq\; \frac{3n}{10} - 6
$$

개이다. 따라서 $m$ 으로 나눈 뒤 더 큰 쪽에 남는 원소는 많아야 $\dfrac{7n}{10} + 6$ 개이다.

??? proof "증명"

    무리 중앙값이 $\lceil n/5\rceil$ 개 있고 $m$ 은 그들의 중앙값이므로, 무리 중앙값 가운데 적어도 절반이 $m$ 이하이다. 그런 무리 중앙값의 수는 $\bigl\lceil \tfrac12\lceil n/5\rceil\bigr\rceil$ 이다.

    무리 중앙값 $g$ 가 $m$ 이하이면, $g$ 가 속한 무리에서 $g$ 자신과 $g$ 보다 작은 원소 $2$ 개, 곧 적어도 $3$ 개가 $m$ 이하이다. 무리가 $5$ 개짜리로 정렬되어 있기 때문이다.

    그런 무리들은 서로 겹치지 않으므로 $m$ 이하인 원소는 적어도 $3\bigl\lceil \tfrac12\lceil n/5\rceil\bigr\rceil \geq \tfrac{3n}{10}-6$ 개다. $m$ 이상인 쪽도 대칭으로 같다. 남는 쪽은 전체에서 이만큼을 뺀 것이므로 많아야 $\tfrac{7n}{10}+6$ 개다.

!!! note "쓰임새"
    보장은 **최악의 경우**에 대한 것이다. 마구잡이 축은 기댓값으로는 좋지만 어떤 입력에서는 늘 나쁘게 떨어질 수 있다. 중앙값의 중앙값은 그 가능성을 아예 없앤다.

**보기 1.** <span class="diff easy" title="쉬움"></span> $n = 100$ 일 때 정리 1이 보장하는 양쪽 크기를 구하시오.

??? success "풀이"

    무리는 $\lceil 100/5\rceil = 20$ 개, 그중 절반인 $10$ 개의 무리 중앙값이 $m$ 이하이다. 무리마다 $3$ 개씩이므로 $m$ 이하인 원소는 적어도 $30$ 개다.

    대칭으로 $m$ 이상도 적어도 $30$ 개이므로, 나눈 뒤 남는 쪽은 많아야 $70$ 개다. 곧 되돌이 부름마다 적어도 $30\%$ 가 걸러진다.

## 2. 선형 시간 점화식

### 정리 2. 점화식의 풀이 — 최악의 경우에도 $T(n) = O(n)$ { .thm }

정리 1의 보장 아래 전체 일의 양은

$$
T(n) \leq T\!\left(\left\lceil \frac n5\right\rceil\right) + T\!\left(\frac{7n}{10}+6\right) + O(n)
$$

를 만족하고, 이 점화식의 풀이는 $T(n) = O(n)$ 이다.

??? proof "증명"

    $O(n)$ 항을 $an$ 이라 하고, 어떤 상수 $c$ 에 대해 더 작은 모든 입력에서 $T(\cdot)\leq c\,(\cdot)$ 이라고 놓자(대입법). 그러면

    $$
    T(n) \leq \frac{cn}{5} + c\left(\frac{7n}{10}+6\right) + an
    = cn\left(\frac15+\frac7{10}\right) + 6c + an
    = \frac{9cn}{10} + 6c + an
    $$

    이다. 이것이 $cn$ 이하가 되려면

    $$
    \frac{9cn}{10} + 6c + an \leq cn
    \iff
    an + 6c \leq \frac{cn}{10}
    $$

    이면 된다. $c \geq 10a$ 로 잡고 $n \geq 60$ 이면 $an \leq cn/10 - 6c$ 가 되어 부등식이 성립한다. $n < 60$ 은 상수 개이므로 $c$ 를 키워 덮으면 된다.

!!! note "쓰임새"
    핵심은 두 되돌이 부분 문제의 크기 합이 $\tfrac15 + \tfrac7{10} = \tfrac9{10} < 1$ 이라는 점이다. 합이 $1$ 미만이면 각 켜에서 하는 일이 등비로 줄어 전체가 선형이 된다. 합이 정확히 $1$ 이면 $\Theta(n\log n)$ 이 된다.

**보기 2.** <span class="diff easy" title="쉬움"></span> 되돌이 켜마다 하는 일이 등비수열을 이룸을 확인하시오.

??? success "풀이"

    맨 위 켜에서 $an$ 만큼 일한다. 다음 켜에서는 크기가 $n/5$ 와 $7n/10$ 인 두 문제이므로 합이 $\tfrac9{10}n$ 이고, 일은 $a\cdot\tfrac9{10}n$ 이다. 그다음 켜는 $a\left(\tfrac9{10}\right)^2 n$ 이다.

    전체는

    $$
    an\sum_{i\ge 0}\left(\frac9{10}\right)^i = an\cdot\frac{1}{1-9/10} = 10an = O(n)
    $$

    이다. 공비가 $1$ 보다 작다는 것이 선형성의 전부다.

## 3. 왜 다섯씩 묶는가

### 정리 3. 무리 크기의 조건 — 홀수 $g > 3$ 이어야 선형이다 { .thm }

무리 크기를 홀수 $g$ 로 두면 정리 1과 같은 논증이 양쪽에 적어도

$$
\frac{g+1}{4g}\,n
$$

개를 보장하고, 점화식은 $T(n/g) + T\bigl(n - \tfrac{g+1}{4g}n\bigr) + O(n)$ 이 된다. 이 점화식이 선형이 되는 것과

$$
\frac1g + 1 - \frac{g+1}{4g} < 1
\quad\Longleftrightarrow\quad
g > 3
$$

인 것은 같은 말이다.

??? proof "증명"

    무리는 $n/g$ 개이고 그중 절반의 중앙값이 $m$ 이하이며, 그런 무리마다 $\lceil g/2\rceil = \tfrac{g+1}{2}$ 개가 $m$ 이하이다. 따라서 보장되는 수는

    $$
    \frac{g+1}{2}\cdot\frac12\cdot\frac ng = \frac{g+1}{4g}\,n
    $$

    이다. 두 부분 문제 크기의 합을 $n$ 으로 나눈 값이 $1$ 보다 작아야 선형이므로 조건은

    $$
    \frac1g + \left(1 - \frac{g+1}{4g}\right) < 1
    \iff \frac1g < \frac{g+1}{4g}
    \iff 4 < g+1
    \iff g > 3
    $$

    이다.

| 무리 크기 $g$ | 걸러지는 비율 $\frac{g+1}{4g}$ | 부분 문제 크기의 합 | 결과 |
|---|---|---|---|
| $3$ | $1/3$ | $\tfrac13 + \tfrac23 = 1$ | $\Theta(n\log n)$ |
| $5$ | $3/10$ | $\tfrac15 + \tfrac7{10} = \tfrac9{10}$ | $\Theta(n)$ |
| $7$ | $2/7$ | $\tfrac17 + \tfrac57 = \tfrac67$ | $\Theta(n)$ |
| $9$ | $5/18$ | $\tfrac19 + \tfrac{13}{18} = \tfrac{15}{18}$ | $\Theta(n)$ |

!!! note "쓰임새"
    $g=3$ 은 합이 정확히 $1$ 이라 아슬아슬하게 선형이 되지 못한다. $g \geq 7$ 은 선형이지만 무리를 정렬하는 상수 비용이 커진다. $g=5$ 가 선형이 되는 가장 작은 홀수여서 쓰인다.

**문제 1.** <span class="diff med" title="중간"></span> 무리 크기를 짝수로 잡으면 무엇이 달라지는지 설명하시오.

??? success "풀이"

    짝수 무리에는 가운데 원소가 하나로 정해지지 않아 두 후보 가운데 하나를 골라야 한다. 아래쪽을 고르면 그 무리에서 $m$ 이하가 보장되는 수가 $g/2$ 로 홀수일 때의 $\tfrac{g+1}{2}$ 보다 하나 적다.

    $g=6$ 이면 보장 비율이 $\tfrac{g}{4g}=\tfrac14$ 이고 합은 $\tfrac16+\tfrac34=\tfrac{11}{12}<1$ 로 여전히 선형이다. 다만 같은 비용으로 $g=5$ 보다 보장이 나쁘므로 쓸 까닭이 없다. 홀수를 쓰는 것은 이런 손해를 피하기 위함이다.

## 4. 구현

세 갈래 나눔을 쓰면 축과 같은 값이 여럿일 때도 한 번에 걸러진다.

```python
"""
최악의 경우 선형 고르기를 위한 중앙값의 중앙값 축 고르기.

배열을 5개씩 무리로 나누고 무리마다 중앙값을 찾은 뒤,
그 중앙값들의 중앙값을 되돌이로 고른다. 그러면 적어도 원소의 30%을
걸러 내는 축이 보장된다.
"""

# === 작은 무리 정렬 ===

def sort5(arr: list, lo: int, hi: int) -> None:
    """끼워넣기 정렬로 arr[lo..hi]을 정렬한다(5개 이하 무리용)."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# === 중앙값의 중앙값 ===

def median_of_medians(arr: list, lo: int, hi: int) -> int:
    """중앙값의 중앙값으로 축을 고른다. 축의 값을 되돌린다."""
    n = hi - lo + 1
    if n <= 5:
        sort5(arr, lo, hi)
        return arr[lo + n // 2]

    # 5개짜리 무리마다 중앙값 찾기
    num_groups = (n + 4) // 5
    for i in range(num_groups):
        group_lo = lo + i * 5
        group_hi = min(group_lo + 4, hi)
        sort5(arr, group_lo, group_hi)
        # 무리 중앙값을 배열 앞으로 옮기기
        median_idx = group_lo + (group_hi - group_lo) // 2
        arr[lo + i], arr[median_idx] = arr[median_idx], arr[lo + i]

    # 무리 중앙값의 중앙값을 되돌이로 찾기
    return median_of_medians(arr, lo, lo + num_groups - 1)

def select_mom(arr: list, k: int):
    """최악의 경우 O(n)에 k번째로 작은 것(0부터 세는)을 찾는다."""
    data = arr.copy()
    return _select(data, 0, len(data) - 1, k)

def _select(arr: list, lo: int, hi: int, k: int):
    """중앙값의 중앙값 축을 쓴 되돌이 고르기."""
    if lo == hi:
        return arr[lo]

    pivot = median_of_medians(arr, lo, hi)

    # 세 갈래 나눔
    lt, gt = lo, hi
    i = lo
    while i <= gt:
        if arr[i] < pivot:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    if k < lt:
        return _select(arr, lo, lt - 1, k)
    elif k > gt:
        return _select(arr, gt + 1, hi, k)
    else:
        return arr[k]

# === 시연 ===

if __name__ == "__main__":
    data = [31, 12, 5, 23, 7, 19, 42, 3, 15, 8, 27, 35, 1, 10, 20]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [0, 4, 7, 11, 14]:
        result = select_mom(data, k)
        expected = sorted(data)[k]
        status = "OK" if result == expected else "MISMATCH"
        print(f"k={k:2d}: got {result:3d}, expected {expected:3d} [{status}]")
```

**출력:**

```
Array:  [31, 12, 5, 23, 7, 19, 42, 3, 15, 8, 27, 35, 1, 10, 20]
Sorted: [1, 3, 5, 7, 8, 10, 12, 15, 19, 20, 23, 27, 31, 35, 42]

k= 0: got   1, expected   1 [OK]
k= 4: got   8, expected   8 [OK]
k= 7: got  15, expected  15 [OK]
k=11: got  27, expected  27 [OK]
k=14: got  42, expected  42 [OK]
```

## 연습문제

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 중앙값의 중앙값이 보장하는 것이 무엇이며, 그것이 왜 최악의 경우 선형성을 낳는지 밝히시오.

??? success "풀이"

    보장하는 것은 정리 1의 부등식, 곧 축의 순위가 양 끝에서 적어도 $3n/10$ 만큼 떨어져 있다는 사실이다. 그래서 나눈 뒤 남는 쪽이 많아야 $7n/10$ 이다.

    여기에 축을 고르는 되돌이 비용 $T(n/5)$ 를 더해도 두 부분 문제 크기의 합이 $\tfrac9{10}n < n$ 이므로, 정리 2의 대입법으로 $T(n)=O(n)$ 이 나온다. 마구잡이 축은 기댓값으로만 이런 성질을 가지므로 최악의 경우 $O(n^2)$ 를 막지 못한다.

**연습문제 2.** <span class="diff easy" title="쉬움"></span> 배열 $[31, 12, 5, 23, 7, 19, 42, 3, 15, 8]$ 에서 중앙값의 중앙값 축을 손으로 구하시오.

??? success "풀이"

    $5$ 개씩 두 무리로 나누어 정렬한다.

    - 무리 1: $[31,12,5,23,7] \to [5,7,12,23,31]$, 중앙값 $12$
    - 무리 2: $[19,42,3,15,8] \to [3,8,15,19,42]$, 중앙값 $15$

    중앙값이 $\{12, 15\}$ 두 개이므로 그 중앙값을 되돌이로 찾는다. 원소가 $5$ 개 이하이므로 정렬해 가운데를 가져오면 $15$ 이다(구현에서는 $n//2$ 번째를 쓴다).

    축 $15$ 로 나누면 아래쪽에 $\{3,5,7,8,12\}$, 위쪽에 $\{19,23,31,42\}$ 가 놓여 양쪽이 고르게 갈린다.

**연습문제 3.** <span class="diff med" title="중간"></span> 축을 고르는 되돌이 부름을 없애고 그냥 무리 중앙값들의 **평균**을 쓰면 어떻게 되는지 밝히시오.

??? success "풀이"

    평균은 순위를 보장하지 못한다. 값 몇 개가 아주 크면 평균이 그쪽으로 끌려가 축이 배열의 끝 가까이 떨어질 수 있고, 그러면 정리 1의 부등식이 무너진다.

    예를 들어 원소 대부분이 $0$ 이고 하나가 $10^9$ 이면 평균은 크지만 순위로는 거의 최댓값이다. 축이 극단으로 가면 나눈 뒤 거의 모든 원소가 살아남아 최악의 경우 $O(n^2)$ 로 되돌아간다. **순위에 대한 진술**이 필요하므로 중앙값이라야 한다.

**연습문제 4.** <span class="diff med" title="중간"></span> 실전에서 중앙값의 중앙값 대신 마구잡이 축을 더 즐겨 쓰는 까닭을 밝히고, 둘을 섞는 방법을 서술하시오.

??? success "풀이"

    중앙값의 중앙값은 상수 인자가 크다. 무리마다 정렬하고 축을 고르는 되돌이 부름까지 있어 실제 걸리는 때가 마구잡이 축보다 몇 배 크다. 마구잡이 축은 기대 시간이 $O(n)$ 이고 상수가 작아 거의 언제나 더 빠르다.

    섞는 방법이 **인트로셀렉트**다. 마구잡이 축으로 시작하되 되돌이 깊이가 $c\log n$ 을 넘으면 중앙값의 중앙값으로 갈아탄다. 그러면 보통은 마구잡이의 빠르기를 누리면서 최악의 경우 $O(n)$ 보장도 잃지 않는다. C++ 표준 라이브러리의 `nth_element` 가 이 방식이다.

**연습문제 5.** <span class="diff hard" title="어려움"></span> 정리 1의 $-6$ 항이 어디서 나오는지 밝히고, $n$ 이 $10$ 의 배수일 때 이 항이 필요한지 따지시오.

??? success "풀이"

    $-6$ 은 두 곳의 어긋남을 덮는다. 첫째, 마지막 무리가 $5$ 개가 안 될 수 있어 그 무리의 중앙값이 $3$ 개를 보장하지 못한다. 둘째, 축 $m$ 이 속한 무리 자체는 양쪽 어느 쪽으로도 온전히 세지 못한다. 이 둘을 넉넉히 덮으려고 무리 $2$ 개 몫인 $2\times 3 = 6$ 을 뺀다.

    $n$ 이 $10$ 의 배수여도 이 항은 필요하다. 마지막 무리가 꽉 차더라도 축이 든 무리를 빼는 몫은 남기 때문이다. 다만 점근으로는 상수이므로 $O(n)$ 결론에는 영향이 없다.

## 정리하며

중앙값의 중앙값은 **축의 순위에 대한 최악의 경우 보장**을 사는 기법이다.

1. $5$ 개씩 묶어 무리 중앙값을 구하고 그 중앙값을 되돌이로 찾으면, 축의 순위가 양 끝에서 적어도 $3n/10$ 만큼 떨어져 있다(정리 1).
2. 그 덕분에 부분 문제 크기의 합이 $\tfrac9{10}n$ 으로 $n$ 보다 작아지고, 대입법으로 $T(n)=O(n)$ 이 나온다(정리 2).
3. 무리 크기는 홀수 $g>3$ 이어야 하며, 그 가운데 가장 작은 $g=5$ 를 쓴다(정리 3). $g=3$ 은 합이 정확히 $1$ 이라 $\Theta(n\log n)$ 에 머문다.

값은 최악의 경우 보장이지만 상수 인자가 커서, 실전에서는 마구잡이 축으로 시작해 깊이가 깊어질 때만 갈아타는 인트로셀렉트를 쓴다. 마구잡이 축 자체는 「[빠른 고르기](quickselect.md)」에서, 선형 고르기의 다른 길은 「[선형 시간 고르기](linear.md)」에서 다룬다.

**참고 문헌**

- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 9장. MIT Press.
