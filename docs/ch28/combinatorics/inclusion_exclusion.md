# 넣고 빼기 원리

넣고 빼기 원리는 교집합의 크기를 번갈아 더하고 빼서 합집합의 크기를 셈한다. 어려운 "합집합의 원소 세기" 문제를 더 쉬운 "교집합의 원소 세기" 아래 문제로 바꾸어, 어긋난 자리바꿈 세기, 오일러 파이 함수, 체 바탕 알고리즘에 없어서는 안 될 도구가 된다.

---

## 1. 직관

$|A| + |B|$을 더하면 $A \cap B$의 원소를 두 번 세므로 $|A \cap B|$을 뺀다. 모임이 셋이면 짝별 교집합을 모두 빼는 것이 지나치므로 세 겹 교집합을 다시 더한다. 이 번갈아 나타나는 무늬는 모임이 몇 개이든 넓혀진다.

---

## 2. 모임 둘의 공식

유한 모임 $A$과 $B$에 대해:

$$
|A \cup B| = |A| + |B| - |A \cap B|
$$

---

## 3. 모임 셋의 공식

유한 모임 $A$, $B$, $C$에 대해:

$$
|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|
$$

---

## 4. 일반 공식

유한 모임 $A_1, A_2, \ldots, A_n$에 대해:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \sum_{1 \le i_1 < \cdots < i_k \le n} |A_{i_1} \cap \cdots \cap A_{i_k}|
$$

같은 말로 $S = \{1, 2, \ldots, n\}$이라 적으면:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{\emptyset \ne T \subseteq S} (-1)^{|T|+1} \left|\bigcap_{i \in T} A_i\right|
$$

---

## 5. 귀납으로 밝히기

**바탕 경우($n = 1$).** $|A_1| = |A_1|$이다. 하찮게 참이다.

**귀납 걸음.** 모임 $n-1$개에 대해 공식이 성립한다고 하자. 다음과 같이 적는다:

$$
\bigcup_{i=1}^{n} A_i = \left(\bigcup_{i=1}^{n-1} A_i\right) \cup A_n
$$

모임 둘의 공식에 따라:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \left|\bigcup_{i=1}^{n-1} A_i\right| + |A_n| - \left|\left(\bigcup_{i=1}^{n-1} A_i\right) \cap A_n\right|
$$

$\left(\bigcup_{i=1}^{n-1} A_i\right) \cap A_n = \bigcup_{i=1}^{n-1} (A_i \cap A_n)$이므로 모임 $n-1$개의 두 합집합에 귀납 가정을 쓸 수 있다. 부호에 조심하며 펼쳐 항을 모으면 모임 $n$개의 공식을 얻는다.

??? note "다른 밝힘(두 번 세기)"
    $\bigcup A_i$의 아무 원소 $x$을 잡고 $x$이 모임 가운데 꼭 $m$개에 든다고 하자($m \ge 1$). 오른쪽은 $x$을 꼭 다음만큼 센다:

    $$
    \binom{m}{1} - \binom{m}{2} + \binom{m}{3} - \cdots + (-1)^{m+1}\binom{m}{m}
    $$

    $x = -1$인 이항 정리에 따라:

    $$
    \sum_{k=0}^{m} \binom{m}{k}(-1)^k = (1 - 1)^m = 0
    $$

    따라서 $\sum_{k=1}^{m} (-1)^{k+1}\binom{m}{k} = 1$이다. 합집합의 모든 원소가 꼭 한 번씩 세어진다.

---

## 6. 보기: 2, 3, 5으로 나누어떨어지는 정수 세기

**문제.** $\{1, 2, \ldots, 100\}$ 가운데 2, 3, 5으로 나누어떨어지는 것은 몇 개인가?

$A_2$, $A_3$, $A_5$을 각각 2, 3, 5의 배수 모임이라 하자.

$$
|A_2| = 50, \quad |A_3| = 33, \quad |A_5| = 20
$$

$$
|A_2 \cap A_3| = |A_6| = 16, \quad |A_2 \cap A_5| = |A_{10}| = 10, \quad |A_3 \cap A_5| = |A_{15}| = 6
$$

$$
|A_2 \cap A_3 \cap A_5| = |A_{30}| = 3
$$

넣고 빼기에 따라:

$$
|A_2 \cup A_3 \cup A_5| = 50 + 33 + 20 - 16 - 10 - 6 + 3 = 74
$$

---

## 7. 쓰임새: 어긋난 자리바꿈

**어긋난 자리바꿈**은 붙박이점이 없는 자리바꿈이다. $A_i$을 원소 $i$을 붙박이로 두는 $[n]$의 자리바꿈 모임이라 하자. 붙박이점이 적어도 하나 있는 자리바꿈의 개수는 $|\bigcup A_i|$이다.

$|A_{i_1} \cap \cdots \cap A_{i_k}| = (n-k)!$이고(남은 $n-k$개는 자유롭게 자리를 바꾼다) 번호 $k$개를 고르는 방법이 $\binom{n}{k}$가지이므로:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \binom{n}{k}(n-k)!
$$

어긋난 자리바꿈의 개수 $D_n = n! - |\bigcup A_i|$은:

$$
D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!} \approx \frac{n!}{e}
$$

---

## 8. 구현

```python
from itertools import combinations

def inclusion_exclusion(
    universe_size: int,
    sets: list[set],
) -> int:
    """넣고 빼기로 모임의 합집합의 원소를 센다.

    인수:
        universe_size: 곧바로 쓰지는 않는다. 모임이 원소를 담는다.
        sets: 합집합의 크기를 알고 싶은 모임의 목록.

    반환값:
        합집합의 크기.
    """
    n = len(sets)
    total = 0
    for k in range(1, n + 1):
        sign = (-1) ** (k + 1)
        for combo in combinations(range(n), k):
            intersection = sets[combo[0]]
            for idx in combo[1:]:
                intersection = intersection & sets[idx]
            total += sign * len(intersection)
    return total

def count_derangements(n: int) -> int:
    """넣고 빼기 식으로 [n]의 완전 어긋난 차례 수를 센다."""
    result = 0
    factorial = 1
    for k in range(n + 1):
        if k > 0:
            factorial *= k
        # 쓰지 않음: 공식에서 곧바로 셈한다
    # 곧바른 셈하기
    from math import factorial as fact
    return sum((-1) ** k * fact(n) // fact(k) for k in range(n + 1))

if __name__ == "__main__":
    # === 보기: {1..100}에서 2, 3, 5의 배수 ===
    A2 = {i for i in range(1, 101) if i % 2 == 0}
    A3 = {i for i in range(1, 101) if i % 3 == 0}
    A5 = {i for i in range(1, 101) if i % 5 == 0}
    result = inclusion_exclusion(100, [A2, A3, A5])
    print(f"|A2 ∪ A3 ∪ A5| = {result}")  # 74

    # === 어긋난 자리바꿈 ===
    for n in range(1, 8):
        print(f"D_{n} = {count_derangements(n)}")
```

---

## 9. 복잡도

일반 넣고 빼기 공식은 모임 $n$개의 비어 있지 않은 부분 모임 $2^n - 1$가지 모두에 대해 더해야 한다. 이 지수 비용은 가장 나쁜 경우에 피할 수 없지만 $n$이 작으면(보기로 나누어떨어짐 조건이 몇 개로 붙박여 있으면) 흔히 받아들일 만하다.

---

## 연습문제

**연습문제 1.**
넣고 빼기로 1부터 100까지 정수 가운데 2, 3, 5으로 나누어떨어지는 개수를 세어라.

??? success "연습문제 1 풀이"
    $A_2, A_3, A_5$을 배수 모임이라 하자. $|A_2| = 50, |A_3| = 33, |A_5| = 20$이다. $|A_2 \cap A_3| = |A_6| = 16, |A_2 \cap A_5| = |A_{10}| = 10, |A_3 \cap A_5| = |A_{15}| = 6$이다. $|A_2 \cap A_3 \cap A_5| = |A_{30}| = 3$이다. 넣고 빼기에 따라 $|A_2 \cup A_3 \cup A_5| = 50 + 33 + 20 - 16 - 10 - 6 + 3 = 74$이다.

---

**연습문제 2.**
넣고 빼기로 어긋난 자리바꿈의 개수 $D_n$의 공식을 이끌어 내라.

??? success "연습문제 2 풀이"
    $A_i$을 원소 $i$이 붙박인 자리바꿈이라 하자. 넣고 빼기에 따라 어긋난 자리바꿈은 붙박이점이 없는 자리바꿈이므로 $D_n = n! - |A_1 \cup \cdots \cup A_n|$이다. $|A_{i_1} \cap \cdots \cap A_{i_k}| = (n-k)!$이다($k$개를 붙박이로 두고 나머지를 자리바꿈한다). 그런 교집합이 $\binom{n}{k}$개이다. 넣고 빼기에 따라 $D_n = \sum_{k=0}^{n}(-1)^k \binom{n}{k}(n-k)! = n!\sum_{k=0}^{n}\frac{(-1)^k}{k!}$이다.

---

**연습문제 3.**
넣고 빼기로 크기 $m$인 모임에서 크기 $n$인 모임 위로 가는 위로 함수의 개수를 세어라.

??? success "연습문제 3 풀이"
    $A_j$을 공역에서 원소 $j$을 빠뜨리는 함수라 하자. 위로 함수는 모든 $A_j$을 피한다. $|A_{j_1} \cap \cdots \cap A_{j_k}| = (n-k)^m$이다($m$개가 저마다 남은 $n - k$개의 과녁으로 간다). 위로 함수의 개수: $\sum_{k=0}^{n}(-1)^k\binom{n}{k}(n-k)^m$이다. $m < n$이면 0이고 $m \geq n$이면 양수이다.

---

**연습문제 4.**
에라토스테네스의 체가 소수 세기에 넣고 빼기를 쓴 특별한 경우임을 밝혀라.

??? success "연습문제 4 풀이"
    $N$ 이하의 소수를 세려면 각 소수 $p \leq \sqrt{N}$에 대해 $A_p$을 $\{2, \ldots, N\}$에서 $p$의 배수라 하자. $(\sqrt{N}, N]$의 소수는 어떤 $A_p$에도 들지 않는 것들이다. 넣고 빼기에 따라 개수 $= |U| - |A_{p_1} \cup \cdots \cup A_{p_k}| = \sum_{S \subseteq \{p_1,\ldots,p_k\}} (-1)^{|S|} \lfloor N / \prod_{p \in S} p \rfloor$이다. 에라토스테네스의 체는 배수를 되풀이해 지워 이를 짜며, 이는 소수를 하나씩 넣고 빼기 합에 적용하는 것과 같다. 르장드르 체가 이 이음을 엄밀히 적는다.

## 정리하며

이 마당은 직관、모임 둘의 공식、모임 셋의 공식、일반 공식을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.). Addison-Wesley. Chapter 4.
