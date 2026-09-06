# 나누기, 이기기, 아우르기

나누어 이기기 알고리즘마다 세 단계 무늬를 따른다. 곧 들임을 쪼개고, 조각을 풀고, 결과를 어울린다. [전략](strategy.md) 쪽에서 이 무늬를 큰 틀로 소개했다면, 이 쪽에서는 단계마다 자세히 살펴 각 단계의 꾸밈 고름이 옳음과 효율을 어떻게 정하는지 보인다.

같은 문제도 나누기, 이기기, 아우르기 전략을 달리해 풀 수 있고 저마다 다른 되돌이 관계식과 다른 도는 시간으로 이어지므로, 단계마다의 얼개를 아는 것이 꼭 필요하다.

## 나누기 단계

**나누기** 단계는 들임을 같은 갈래의 더 작은 문제로 나눈다. 목표는 되돌이 켜마다 문제 크기를 상수 배로 줄이는 것이다.

### 쪼개기 전략

The most common approach splits the input in half, producing two subproblems of size $\lfloor n/2 \rfloor$ and $\lceil n/2 \rceil$. This balanced split is the default for array-based problems such as merge sort and binary search.

더 널리 보면 나누기 단계는 크기 $n/b$인 작은 문제 $a$개를 낳을 수 있다:

- **둘로 쪼개기**($a = 2$, $b = 2$): 어울러 정렬, 가장 가까운 점 짝.
- **하나만 남기기**($a = 1$, $b = 2$): 이분 찾기는 들임의 반을 버린다.
- **여러 갈래로 쪼개기**($a > 2$): 카라추바는 둘로 쪼갠 뒤($b = 2$) 작은 문제 $a = 3$개를 쓴다.

!!! warning "고르지 않은 쪼개기는 성능을 해친다"
    If the divide step produces subproblems of sizes $n - 1$ and $1$ (as in naive quicksort on a sorted array), the recursion depth becomes $O(n)$ and the total work is often $O(n^2)$. Balanced splits keep the recursion depth at $O(\log n)$.

### 나누는 값

나누기 단계 자체는 $D(n)$의 시간이 든다. 여러 알고리즘에서 나누기는 아무것도 아니다:

- **Merge sort**: compute $\text{mid} = \lfloor (l + r) / 2 \rfloor$ in $O(1)$.
- **이분 찾기**: 가운뎃점을 $O(1)$에 셈한다.
- **가장 가까운 짝**: 점을 자리표로 정렬하거나 가운뎃값에서 쪼갠다. $O(n)$이 들고 미리 정렬해 두었으면 $O(1)$이다.

나누는 값은 되돌이 관계식 $T(n) = aT(n/b) + f(n)$의 $f(n)$ 항에 보태진다.

## 이기기 단계

**이기기** 단계는 더 작은 들임에 같은 알고리즘을 써서 작은 문제마다 되돌이로 푼다. 들임이 곧바로 풀 만큼 작은 **바탕 경우**에 이를 때까지 되돌이가 이어진다.

### 바탕 경우

잘 고른 바탕 경우는 옳음과 효율 모두에 결정적이다.

| 알고리즘 | 바탕 경우 | 곧바른 풀이 |
|---|---|---|
| Merge sort | $n \le 1$ | A single element is already sorted |
| 이분 찾기 | $l > r$ | 찾는 값이 배열에 없다 |
| 카라추바 | $n = 1$ | 한 자리 곱셈 |
| 슈트라센 | $n = 1$ | 홑값 곱셈 |

### 섞인 바탕 경우

In practice, switching to a simpler algorithm below a threshold $n_0$ reduces constant-factor overhead. For example, merge sort implementations typically switch to insertion sort when $n \le 16$, because insertion sort's lower overhead makes it faster on small arrays despite its $O(n^2)$ worst case.

문턱값 $n_0$은 점근 복잡도를 바꾸지 않지만 실제 성능을 상수 배 낫게 할 수 있다.

### 작은 문제끼리 안 얽힘

나누어 이기기를 규정하는 성질은 작은 문제끼리 **얽히지 않는다**는 것이다. 곧 하나를 푸는 데 다른 것의 결과가 필요 없다. 이 안 얽힘이 작은 문제가 겹치고 풀이를 나눠 갖는 동적 계획과 나누어 이기기를 가른다.

안 얽힘 덕분에 나누어 이기기 알고리즘은 **나란히 하기**에 자연스레 어울린다. 곧 어느 켜에서든 작은 문제 $a$개를 한꺼번에 풀 수 있다.

## 아우르기 단계

**아우르기** 단계는 작은 문제의 풀이를 본디 문제의 풀이로 어울린다. 흔히 알고리즘으로 가장 재미있는 단계이자 전체 복잡도를 정하는 단계이다.

### 아우르기 단계의 보기

**어울러 정렬.** 아우르기 단계는 정렬된 두 반쪽을 정렬된 배열 하나로 어울린다. 어울리기는 두 반쪽을 한꺼번에 훑어 $O(n)$ 시간에 내놓는다:

```python
def merge(left, right):
    """정렬된 두 목록을 정렬된 하나로 어울리기."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**이분 찾기.** 아우르기 단계는 아무것도 아니다. 곧 되돌이 부름 하나의 답이 본디 문제의 답이다. 아우르는 값은 $O(1)$이다.

**최대 부분 배열(나누어 이기기).** 아우르기 단계는 가운뎃점을 걸치는 최대 부분 배열을 찾는다. 왼쪽과 오른쪽 반쪽을 한 줄로 훑어야 하므로 $O(n)$이 든다.

### 아우르는 값과 그 영향

아우르는 값 $C(n)$은 되돌이 관계식의 덧짐 함수 $f(n) = D(n) + C(n)$의 한 몫이다. $f(n)$과 작은 문제의 품 사이 관계가 마스터 정리를 거쳐 전체 복잡도를 정한다:

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

- If $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$, the subproblem work dominates: $T(n) = \Theta(n^{\log_b a})$.
- If $f(n) = \Theta(n^{\log_b a})$, the work is evenly distributed: $T(n) = \Theta(n^{\log_b a} \log n)$.
- If $f(n) = \Omega(n^{\log_b a + \epsilon})$ and the regularity condition holds, the combine work dominates: $T(n) = \Theta(f(n))$.

이 되돌이 관계식을 푸는 자세한 이야기는 [되돌이 관계 살피기](recurrence.md) 쪽을 보라.

## 모두 모아 보기: 어울러 정렬

어울러 정렬은 세 단계를 모두 깔끔하게 보여 준다.

**Divide.** Split the array at the midpoint: $\text{mid} = \lfloor (l + r) / 2 \rfloor$. Cost: $O(1)$.

**Conquer.** Recursively sort the left half $A[l \,..\, \text{mid}]$ and the right half $A[\text{mid}+1 \,..\, r]$.

**아우르기.** 정렬된 두 반쪽을 어울린다. 값: $O(n)$.

되돌이 관계식은 다음과 같다

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n)
$$

By the Master Theorem (case 2, with $a = 2$, $b = 2$, $f(n) = \Theta(n)$, and $\log_b a = 1$), the solution is

$$
T(n) = \Theta(n \log n)
$$

```python
def merge_sort(arr):
    """어울러 정렬 알고리즘으로 배열 정렬하기."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])      # 왼쪽 정복하기
    right = merge_sort(arr[mid:])     # 오른쪽 정복하기
    return merge(left, right)          # 합치기
```

## 흔히 빠지는 함정

!!! danger "아우르기 단계를 잊음"
    올바른 나누어 이기기 알고리즘은 작은 문제의 풀이를 아울러야 한다. 아우르지 않고 나누고 이기기만 하면 작은 문제의 풀이만 나올 뿐 본디 문제의 풀이는 나오지 않는다.

!!! warning "겹치는 작은 문제"
    나누기 단계가 아래 짜임을 함께 갖는 작은 문제를 낳으면(보기로 피보나치 수를 되돌이로 셈할 때) 같은 품을 지수적으로 여러 번 되풀이하게 된다. 그런 경우에는 적어 두기를 곁들인 **동적 계획**이 알맞은 틀이다.

## 요약

나누어 이기기의 세 단계, 곧 나누기·이기기·아우르기는 알고리즘 꾸미기의 온전한 요령이 된다. 나누기 단계는 문제 크기를 줄이고, 이기기 단계는 되돌이와 바탕 경우를 다루며, 아우르기 단계는 마지막 답을 짜맞춘다. 도는 시간은 되돌이 관계식 $T(n) = aT(n/b) + f(n)$이 담아내며, 여기서 $f(n)$에는 나누는 값과 아우르는 값이 모두 든다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.

## 연습문제

**연습문제 1.**
나누기, 이기기, 아우르기의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Divide, Conquer, Combine applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
나누기, 이기기, 아우르기의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
나누기, 이기기, 아우르기이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
나누기, 이기기, 아우르기의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
