# 빠른 정렬 분석

빠른 정렬은 실전에서 가장 빠른 두루 쓰이는 비교 정렬이지만, 최악의 경우는 $O(n^2)$으로 힙 정렬이나 병합 정렬보다 나쁘다. 빠른 정렬의 분석을 이해하려면 세 경우(최선, 최악, 평균)를 뜯어보아야 하고, 무엇보다 무작위 입력에서 **기대** 도는 시간이 $O(n \log n)$임을 증명해야 한다. 이 쪽은 CLRS의 이름난 지시 확률 변수 증명까지 아우른 온전한 분석을 내보인다.

## 최악의 경우 분석

최악의 경우는 나눌 때마다 가장 치우친 쪼갬이 나올 때 일어난다. 곧 크기 $n - 1$인 부분 배열 하나와 크기 0인 부분 배열 하나이다.

$$
T(n) = T(n - 1) + T(0) + \Theta(n) = T(n - 1) + \Theta(n)
$$

점화식을 펼치면 다음과 같다.

$$
T(n) = \sum_{k=1}^{n} \Theta(k) = \Theta\!\left(\frac{n(n+1)}{2}\right) = \Theta(n^2)
$$

배열이 이미 정렬되어 있거나(첫 원소나 마지막 원소를 축으로 쓸 때) 모든 원소가 같을 때(로무토 나눔에서) 일어난다.

## 최선의 경우 분석

최선의 경우는 나눌 때마다 배열이 꼭 반으로 쪼개질 때 일어난다.

$$
T(n) = 2T(n/2) + \Theta(n) = \Theta(n \log n)
$$

마스터 정리(경우 2, $a = 2$, $b = 2$)로 얻는다.

## 치우쳤어도 쓸 만한 쪼갬

9:1처럼 크게 치우친 쪼갬이라도 여전히 $O(n \log n)$이 나온다.

$$
T(n) = T(n/10) + T(9n/10) + \Theta(n)
$$

되돌이 트리의 가장 긴 길은 깊이가 $\log_{10/9} n = O(\log n)$이고 층마다 많아야 $cn$의 일을 낸다.

$$
T(n) = O(n \log n)
$$

핵심 통찰은 **일정한 비율**로만 쪼개져도 $O(n \log n)$이 된다는 것이다. 최악의 경우가 되려면 나눌 때마다 한쪽에 원소가 $O(1)$개만 남아야 한다.

## 평균의 경우 분석

### 준비

입력이 무작위 순열이고 축이 늘 마지막 원소(로무토)라고 하자. 기대 견줌 횟수 $C(n)$을 뜯어본다.

원소를 정렬한 차례를 $z_1 < z_2 < \cdots < z_n$이라 하고 다음을 정의한다.

$$
X_{ij} = \begin{cases} 1 & (\text{정렬 도중 } z_i \text{과 } z_j \text{이 견주어짐}) \\ 0 & (\text{그 밖의 경우}) \end{cases}
$$

견줌의 총 횟수는 다음과 같다.

$$
C(n) = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} X_{ij}
$$

### 핵심 관찰

두 원소 $z_i$과 $z_j$은 범위 $\{z_{i+1}, \ldots, z_{j-1}\}$의 어떤 원소보다 먼저 둘 가운데 하나가 축으로 뽑힐 **때에만** 견주어진다. $z_i$과 $z_j$ 사이의 원소가 축으로 뽑히는 순간 둘은 다른 부분 배열로 갈라져 다시는 견주어지지 않는다.

### 확률 셈하기

원소 $j - i + 1$개 $\{z_i, z_{i+1}, \ldots, z_j\}$ 가운데 어느 것이든 축으로 처음 뽑힐 가능성이 같다. $z_i$이나 $z_j$이 먼저 뽑힐 확률은 다음과 같다.

$$
\Pr[X_{ij} = 1] = \frac{2}{j - i + 1}
$$

### 기대 견줌 횟수

$$
\mathbb{E}[C(n)] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j - i + 1}
$$

$k = j - i$으로 바꾸면 다음과 같다.

$$
\mathbb{E}[C(n)] = \sum_{i=1}^{n-1} \sum_{k=1}^{n-i} \frac{2}{k + 1} \leq \sum_{i=1}^{n-1} \sum_{k=1}^{n} \frac{2}{k+1}
$$

조화급수 한계 $\sum_{k=1}^{n} \frac{1}{k+1} < \ln n$을 쓰면 다음과 같다.

$$
\mathbb{E}[C(n)] < 2(n - 1) \ln n = 2n \ln n - 2\ln n
$$

밑을 2로 바꾸면 다음과 같다.

$$
\mathbb{E}[C(n)] < 2n \ln n \approx 1.39 n \log_2 n
$$

??? note "정확한 결과"
    정확한 기대 견줌 횟수는 $2(n+1)H_n - 4n$이며, 여기서 $H_n = \sum_{k=1}^{n} 1/k$은 $n$번째 조화수이다. $\gamma \approx 0.5772$이 오일러-마스케로니 상수일 때 $H_n = \ln n + \gamma + O(1/n)$이므로 $\mathbb{E}[C(n)] = 2n \ln n + O(n)$임이 확인된다.

## 복잡도 간추림

$$
\begin{array}{lcc}
\textbf{Case} & \textbf{Time} & \textbf{When} \\
\hline
\text{Best} & \Theta(n \log n) & \text{Balanced partitions} \\
\text{Average} & O(n \log n) & \text{Random permutation} \\
\text{Worst} & \Theta(n^2) & \text{Sorted input, fixed pivot}
\end{array}
$$

**공간 복잡도**: 기대값으로 $O(\log n)$(고르게 나뉠 때의 더미 깊이)이고 최악의 경우 $O(n)$이다. 더 큰 부분 배열에 꼬리 부름 손질을 하면 더미 공간 $O(\log n)$이 보장된다.

## 파이썬 시연

```python
"""
빠른 정렬 분석 보여 주기.

무작위 순열에서 빠른 정렬의 견줌을 세어
이론적 한계 2n*ln(n)과 견준다.
"""

import math
import random


# === 견줌을 세는 빠른 정렬 ====================================================

def quicksort_counted(arr: list, left: int, right: int, count: list) -> None:
    """견줌을 세는, 로무토 나눔을 쓴 빠른 정렬."""
    if left < right:
        pivot = arr[right]
        i = left
        for j in range(left, right):
            count[0] += 1
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[right] = arr[right], arr[i]
        quicksort_counted(arr, left, i - 1, count)
        quicksort_counted(arr, i + 1, right, count)


# === 메인 =====================================================================

if __name__ == "__main__":
    print(f"{'n':>8}  {'avg comps':>10}  {'2n*ln(n)':>10}  {'ratio':>6}")
    print("-" * 40)

    for n in [100, 500, 1000, 5000]:
        trials = 50
        total_comps = 0
        for _ in range(trials):
            arr = list(range(n))
            random.shuffle(arr)
            count = [0]
            quicksort_counted(arr, 0, n - 1, count)
            total_comps += count[0]
        avg = total_comps / trials
        theory = 2 * n * math.log(n)
        print(f"{n:>8}  {avg:>10.0f}  {theory:>10.0f}  {avg/theory:>6.3f}")
```

**출력(흔한 예):**
```
       n    avg comps    2n*ln(n)   ratio
----------------------------------------
     100        816         921   0.886
     500       5679        6215   0.914
    1000      12710       13816   0.920
    5000      78521       85162   0.922
```

실험에서 견줌 횟수가 이론이 내다본 $2n \ln n$에 가깝게 머물러 평균의 경우 분석을 뒷받침한다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, 7.4절.
- Hoare, C. A. R. (1962). Quicksort. *The Computer Journal*, 5(1), 10-16.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley, 5.2.2절.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 빠른 정렬 분석을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
빠른 정렬 분석의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
빠른 정렬 분석은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 빠른 정렬 분석을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.