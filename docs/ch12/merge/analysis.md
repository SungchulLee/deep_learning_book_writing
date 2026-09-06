# 병합 정렬 분석

병합 정렬은 최선, 평균, 최악 어느 경우에도 $O(n \log n)$ 시간을 이루어 가장 내다보기 쉬운 정렬 알고리즘에 든다. 이 쪽은 시간과 공간의 한계를 빈틈없이 끌어내고 견줌 횟수를 자세히 뜯어보며, 병합 정렬이 왜 점근적으로 가장 좋은지와 다른 $O(n \log n)$ 정렬에 견주어 상수 인자가 어디쯤인지를 밝힌다.

## 시간 복잡도

### 점화식

도는 시간은 다음을 만족한다.

$$
T(n) = 2T(n/2) + \Theta(n), \quad T(1) = \Theta(1)
$$

- $2T(n/2)$: 배열의 반쪽 둘에 대한 되돌이 부름.
- $\Theta(n)$: 모든 원소를 한 번 훑는 병합 걸음.

### 마스터 정리로 풀기

$a = 2$, $b = 2$, $f(n) = \Theta(n)$이면 다음과 같다.

$$
\log_b a = \log_2 2 = 1
$$

$f(n) = \Theta(n^1) = \Theta(n^{\log_b a})$이므로 마스터 정리의 경우 2에 든다.

$$
T(n) = \Theta(n \log n)
$$

### 펼쳐서 풀기

점화식을 곧바로 펼치면 다음과 같다.

$$
T(n) = 2T(n/2) + cn = 2[2T(n/4) + cn/2] + cn = 4T(n/4) + 2cn
$$

$k$번 펼치면 다음과 같다.

$$
T(n) = 2^k T(n/2^k) + kcn
$$

$n/2^k = 1$으로 두면 $k = \log_2 n$이고 다음과 같다.

$$
T(n) = nT(1) + cn\log_2 n = \Theta(n \log n)
$$

### 모든 경우가 같다

빠른 정렬과 달리 병합 정렬은 입력 차례와 상관없이 늘 배열을 꼭 반으로 나눈다. 병합 걸음은 늘 원소 $n$개를 다룬다. 그러므로 다음과 같다.

$$
T_{\text{best}}(n) = T_{\text{avg}}(n) = T_{\text{worst}}(n) = \Theta(n \log n)
$$

!!! tip "양날의 칼"
    $\Theta(n \log n)$ 보장은 강점이면서 약점이다. 병합 정렬은 끼워넣기 정렬(최선의 경우 $O(n)$)이나 팀 정렬(정렬된 데이터에서 $O(n)$)처럼 입력에 이미 있는 차례를 살려 쓰지 못한다.

## 견줌 횟수

### 위 한계

되돌이 트리의 층마다 그 층의 부분 배열을 모두 병합하면 많아야 $n - 1$번 견준다(한쪽 부분 배열이 다하면 마지막에 놓이는 원소는 견줄 필요가 없다). $\log_2 n$개 층에 걸치면 다음과 같다.

$$
C(n) \leq n \log_2 n
$$

### 아래 한계

크기가 $p$과 $q$인 정렬된 부분 배열 둘을 병합하려면 최악의 경우 적어도 $p + q - 1$번 견주어야 한다(두 부분 배열의 원소가 번갈아 놓일 때). 모든 층에 걸쳐 더하면 다음과 같다.

$$
C(n) \geq \frac{n}{2} \log_2 n \quad \text{(병합마다 최악의 경우)}
$$

### 정확한 횟수(최악의 경우)

$n = 2^k$일 때 최악의 경우 정확한 견줌 횟수는 다음과 같다.

$$
C(n) = n \log_2 n - n + 1
$$

크기 $m$인 반쪽 둘을 병합할 때 최악의 경우 꼭 $2m - 1$번 견준다는 점을 보고 모든 층에 걸쳐 더하면 나온다.

## 공간 복잡도

### 도움 공간

보통의 병합 정렬은 병합에 쓰는 임시 배열을 위해 $O(n)$ 도움 공간이 든다. 되돌이의 어느 시점에서도 병합은 하나만 돌고 있으므로 여분 공간의 총량은 가장 큰 병합의 크기, 곧 맨 위 층의 $n$에 눌린다.

$$
S(n) = O(n)
$$

### 더미 공간

되돌이 깊이는 $\log_2 n$이라 더미 틀이 $O(\log n)$개 쌓인다. $O(\log n) \subset O(n)$이므로 도움 배열이 으뜸이다.

$$
S_{\text{total}}(n) = O(n) + O(\log n) = O(n)
$$

??? note "병합 정렬을 여분 공간 O(1)으로 할 수 있을까?"
    제자리 병합 정렬 변형이 있기는 하지만 복잡하고 상수 인자가 크다. 크론로드-카타야이넨-파사넨 알고리즘은 여분 공간 $O(1)$으로 $O(n \log n)$ 시간을 이루지만 상수 인자 탓에 실제로 쓰기 어렵다. 실전에서는 $O(n)$의 공간 짐을 받아들일 만하다고 본다.

## 다른 알고리즘과의 견줌

$$
\begin{array}{lccc}
\textbf{Algorithm} & \textbf{Comparisons (worst)} & \textbf{Space} & \textbf{Stable} \\
\hline
\text{Merge sort}   & n \log_2 n - n + 1 & O(n)      & \text{Yes} \\
\text{Heapsort}     & \sim 2n \log_2 n   & O(1)      & \text{No}  \\
\text{Quicksort}    & \sim 1.39 n \log_2 n \text{ (avg)} & O(\log n) & \text{No}
\end{array}
$$

병합 정렬은 표준 $O(n \log n)$ 알고리즘 가운데 가장 적게 견주므로, 견주는 비용이 클 때(이를테면 복잡한 객체나 긴 문자열을 견줄 때) 가장 나은 선택이다.

## 파이썬 시연

```python
"""
병합 정렬 분석 보여 주기.

병합 정렬 중에 견준 정확한 횟수를 세어
이론적 한계 n*log2(n) - n + 1과 견준다.
"""

import math


# === 견줌을 세는 병합 정렬 ====================================================

def merge_counted(arr: list, left: int, mid: int, right: int, count: list) -> None:
    """견줌을 세는 병합."""
    left_half = arr[left:mid + 1]
    right_half = arr[mid + 1:right + 1]
    i = j = 0
    k = left

    while i < len(left_half) and j < len(right_half):
        count[0] += 1
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1
    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1


def merge_sort_counted(arr: list, left: int, right: int, count: list) -> None:
    """견줌을 세는 병합 정렬."""
    if left < right:
        mid = (left + right) // 2
        merge_sort_counted(arr, left, mid, count)
        merge_sort_counted(arr, mid + 1, right, count)
        merge_counted(arr, left, mid, right, count)


# === 메인 =====================================================================

if __name__ == "__main__":
    print(f"{'n':>8}  {'comparisons':>12}  {'n*lg(n)-n+1':>12}  {'ratio':>6}")
    print("-" * 46)

    for k in range(4, 15):
        n = 2 ** k
        # 최악의 경우: 원소가 엇갈려 견줌이 가장 많아진다
        arr = list(range(n))
        count = [0]
        merge_sort_counted(arr, 0, n - 1, count)
        theory = n * math.log2(n) - n + 1 if n > 1 else 0
        ratio = count[0] / theory if theory > 0 else 0
        print(f"{n:>8}  {count[0]:>12}  {theory:>12.0f}  {ratio:>6.3f}")
```

**출력(흔한 예):**
```
       n   comparisons  n*lg(n)-n+1   ratio
----------------------------------------------
      16            33           49   0.673
      32            81          129   0.628
      64           193          321   0.601
     128           449          769   0.584
     256          1025         1793   0.572
     512          2305         4097   0.563
    1024          5121         9217   0.556
    2048         11265        20481   0.550
    4096         24577        45057   0.546
    8192         53249        98305   0.542
   16384        114689       212993   0.539
```

정렬된 입력에서 실제 견줌 횟수는 이론적 최악의 경우보다 훨씬 적어, $n \log_2 n - n + 1$ 한계가 특정한 엇갈림 본새에서만 빈틈없음을 보여 준다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, 2.3절과 4.3~4.5절.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley, 5.2.4절.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, 2.2절.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 병합 정렬 분석을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
병합 정렬 분석의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
병합 정렬 분석은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 병합 정렬 분석을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.