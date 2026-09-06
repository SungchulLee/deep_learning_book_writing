# NC 복잡도 갈래

어떤 문제는 나란한 셈틀로 크게 빨라진다. 수 $n$개를 줄 세우기는 차례로 하면
$O(n \log n)$이 들지만 셈틀 $n$개로는 나란히 $O(\log n)$ 때면 된다.
복잡도 갈래 **NC**(닉의 갈래)는 "효율 좋게 나란히 할 수 있는" 문제,
곧 셈틀을 다항식만큼 써서 로그의 다항식 때에 풀 수 있는 문제를
담는다.

## 엄밀한 정의

NC을 뜻매김하는 여느 모형은 **PRAM**(나란한 아무 닿기 기계)이며, 공통 기억을
함께 쓰는 맞춘 셈틀 여럿으로 이루어진다. 셈틀마다 한 걸음에 공유 기억의 어느
칸이든 읽고 쓸 수 있고 모든 셈틀이 발맞추어 명령을
돌린다.

어떤 상수 $k$과 $c$에 대해 $n$이 들임 크기일 때 PRAM에서 다음처럼 풀 수 있으면

$$
O(\log^k n) \text{ time using } O(n^c) \text{ processors}
$$

그 결정 문제는 **NC**에 든다.

### NC의 켜

NC은 로그 때 가둠의 지수에 따라 켜를 이룬다:

$$
\text{NC}^1 \subseteq \text{NC}^2 \subseteq \cdots \subseteq \text{NC}^k \subseteq \cdots \subseteq \text{NC}
$$

여기서 $\text{NC}^k$은 셈틀을 다항식만큼 써서 $O(\log^k n)$ 때에 풀 수 있는
문제로 이루어진다.

!!! note "회로로 나타내기"
    같은 뜻으로 $\text{NC}^k$은 크기가 다항식이고 깊이가 $O(\log^k n)$이며
    들임 가지 수가 가둬진 불 회로로 판정할 수 있는 문제의 갈래이다.
    깊이가 나란한 때에 해당한다.

## P과의 관계

모든 NC 문제는 P에 든다. 나란한 셈을 차례로 흉내 낼 수 있기 때문이다.
셈틀 $O(n^c)$개가 $O(\log^k n)$걸음 도는 동안 하는 연산은 많아야
$O(n^c \log^k n)$개이며 이는 다항식이다. 그러므로:

$$
\text{NC} \subseteq \text{P}
$$

그 역, 곧 $\text{P} = \text{NC}$인지는 큰 미해결 물음이다.
$\text{P} \ne \text{NC}$이면 다항식 때에 풀리지만 효율 좋게 나란히 할 수 없는
문제가 있다는 뜻이다.

## P완전 문제

NP완전 문제가 NP에서 가장 어려운 문제를 나타내듯, **P완전** 문제(NC 줄이기 아래)는
P에서 나란히 하기가 가장 어려운 문제를 나타낸다. P의 모든 문제가 NC에서 셈할 수 있는
줄이기로 그 문제에 줄어들면 그것이 P완전이다. 어떤 P완전 문제라도 NC에 들면
$\text{P} = \text{NC}$이
된다.

그래서 P완전 문제는 **본디 차례로만 할 수 있다**고 여겨진다. 알려진 로그의 다항식 때
나란한 알고리즘이 없고, 그것이 나란히 된다면 켜 전체가
무너진다.

| P완전 문제 | 밝힘 |
|---|---|
| 회로 값 문제 | 주어진 들임에서 불 회로를 따진다 |
| 호른절 만족 | 호른절의 만족 가능성 |
| 선형 계획(가능성) | 선형 부등식 체계가 풀 수 있는지 판정 |
| 최대 흐름(일반) | 그물의 최대 흐름 찾기 |
| 문맥 자유 문법 속함 | 문자열이 문맥 자유 말에 드는지 판정 |

## NC에 드는 문제

다음 표는 잘 알려진 NC 결과를 적는다. 때와 셈틀 수는 특정 PRAM 알고리즘의
것이며 회로 깊이로 나눈 갈래는 다를 수 있다(보기로 줄 세우기 그물은 깊이
$O(\log n)$을 이루어 회로 모형에서 줄 세우기를 $\text{NC}^1$에
넣는다).

| 문제 | NC 켜 | PRAM 때 | PRAM 셈틀 |
|---|---|---|---|
| 홀짝 | $\text{NC}^1$ | $O(\log n)$ | $O(n)$ |
| 정수 더하기 | $\text{NC}^1$ | $O(\log n)$ | $O(n)$ |
| 정수 곱하기 | $\text{NC}^1$ | $O(\log n)$ | $O(n \log n \log \log n)$ |
| 줄 세우기(콜의 합침 정렬) | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n)$ |
| 행렬 곱하기 | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n^3)$ |
| 이어진 조각 | $\text{NC}^2$ | $O(\log^2 n)$ | $O(n^2)$ |

## 보기: 나란한 앞자락 합

앞자락 합 문제는 셈이 짜임 있는 쓸기 무늬로 어떻게 나란히 $O(\log n)$ 때를
이루는지 보여 주며 이를 $\text{NC}^1$에 또렷이 놓는다. 블렐로크 알고리즘은
**위로 쓸기**(줄임 마당) 뒤에 **아래로 쓸기**(나눔 마당)를 하며 저마다
나란한 걸음 $O(\log n)$번과 온 일 $O(n)$이
든다.

```python
"""
블렐로크 알고리즘으로 하는 나란한 앞자락 합(훑기)(흉내).

Parallel time : O(log n)
Work (total ops): O(n)

유의: 이 짜기는 두 갈래 나무 번호 매기기가 어떤 들임 크기에서도
맞도록 들임을 다음 2의 거듭제곱까지 채운다.
"""

# === 도우미 ===

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# === 나란한 앞자락 합(흉내) ===

def parallel_prefix_sum(arr: list[int]) -> list[int]:
    """블렐로크 알고리즘으로 자기를 담는 앞자락 합을 셈한다.

    위로 쓸기와 아래로 쓸기의 번호 매기기가 모든 원소를 덮도록
    들임을 속에서 2의 거듭제곱 길이로 채운다.

    인수:
        arr: 정수의 목록.

    반환값:
        arr과 길이가 같은 자기를 담는 앞자락 합의 목록.
    """
    n = len(arr)
    if n == 0:
        return []

    # 2의 거듭제곱까지 채운다
    m = next_power_of_two(n)
    x = list(arr) + [0] * (m - n)

    # 위로 쓸기(줄임): 아래에서 위로 부분 합을 쌓는다
    step = 1
    while step < m:
        for i in range(2 * step - 1, m, 2 * step):
            x[i] += x[i - step]
        step *= 2

    # 아래로 쓸기: 위에서 아래로 부분 합을 나눈다
    x[m - 1] = 0
    step = m // 2
    while step >= 1:
        for i in range(2 * step - 1, m, 2 * step):
            temp = x[i - step]
            x[i - step] = x[i]
            x[i] += temp
        step //= 2

    # 자기를 뺀 앞자락 합을 담는 판으로 바꾼다
    result = [x[i] + arr[i] for i in range(n)]
    return result


# === 시연 ===

if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    prefix = parallel_prefix_sum(data)
    print(f"Input:      {data}")
    print(f"Prefix sum: {prefix}")

    # 단순한 차례 훑기와 맞대어 확인한다
    expected = []
    s = 0
    for v in data:
        s += v
        expected.append(s)
    print(f"Expected:   {expected}")
    assert prefix == expected

    # 2의 거듭제곱이 아닌 길이를 시험한다
    data2 = [1, 2, 3, 4, 5]
    prefix2 = parallel_prefix_sum(data2)
    print(f"\nInput:      {data2}")
    print(f"Prefix sum: {prefix2}")
    assert prefix2 == [1, 3, 6, 10, 15]
    print("All tests passed.")
```

**출력:**

```
Input:      [3, 1, 4, 1, 5, 9, 2, 6]
Prefix sum: [3, 4, 8, 9, 14, 23, 25, 31]
Expected:   [3, 4, 8, 9, 14, 23, 25, 31]

Input:      [1, 2, 3, 4, 5]
Prefix sum: [1, 3, 6, 10, 15]
모든 시험을 지났다.
```

## 복잡도 지형

더 넓은 복잡도 갈래의 켜에서 NC이 어디에 있는지 보면 그 뜻이 또렷해진다:

$$
\text{NC}^1 \subseteq \text{L} \subseteq \text{NL} \subseteq \text{NC}^2 \subseteq \text{P} \subseteq \text{NP}
$$

여기서 L과 NL은 로그 공간 갈래이다. 이 포함 가운데 어느 것이 참으로 좁은지는
아직 열려 있지만 모두 그러하리라 믿어진다.

!!! warning "NC과 실제 나란히 하기"
    NC은 셈틀 수에 제한이 없다고 보고 이론으로 나란히 될 수 있음을 잰다.
    실제로는 셈틀 수가 붙박여 있고 주고받기 비용, 기억 대역, 맞추기 덧짐이
    판을 친다. NC에 드는 문제라도 실제 기계에서 효율 좋게 나란히 하기는
    어려울 수
    있다.

## 참고 문헌

- Greenlaw, R., Hoover, H. J., & Ruzzo, W. L. *Limits to Parallel
  Computation: P-Completeness Theory*. Oxford University Press, 1995.
- JaJa, J. *An Introduction to Parallel Algorithms*. Addison-Wesley, 1992.


## 연습문제

**연습문제 1.**
복잡도 갈래 NC을 뜻매김하고 어떤 문제를 담는지 밝혀라.

??? success "연습문제 1 풀이"
    NC(닉의 갈래)은 PRAM에서 셈틀을 다항식만큼($O(n^c)$) 써서 로그의 다항식 때 $O(\log^k n)$에 풀 수 있는 문제의 갈래이다. NC은 '효율 좋게 나란히 할 수 있는' 문제를 담는다. $\text{NC} = \bigcup_{k \geq 1} \text{NC}^k$이며 $\text{NC}^k$은 $O(\log^k n)$ 때를 요구한다. NC의 보기: 줄 세우기, 행렬 곱하기, 이어진 조각, 최대 짝짓기. $\text{NC} \stackrel{?}{=} \text{P}$은 다항식 때 문제가 모두 효율 좋게 나란히 되는지를 묻는다.

---

**연습문제 2.**
행렬 곱하기가 NC에 듦을 보여라.

??? success "연습문제 2 풀이"
    $n \times n$ 행렬 둘: $C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$. 셈틀 $n^3$개를 쓰면 저마다 곱 $A_{ik} B_{kj}$ 하나를 $O(1)$ 때에 셈한다. 그다음 $(i,j)$마다 값 $n$개를 나란한 앞자락 합으로 $O(\log n)$ 때에 더한다. 모두: 셈틀 $O(n^3)$개로 $O(\log n)$ 때. 이는 $\text{NC}^1$에 든다.

---

**연습문제 3.**
P완전 문제란 무엇이며 왜 나란히 하기 어렵다고 여겨지는가?

??? success "연습문제 3 풀이"
    P에 들면서 P의 모든 문제가 NC 줄이기(또는 로그 공간 줄이기)로 그것에 줄어들면 그 문제는 P완전이다. 어떤 P완전 문제라도 NC에 들면 P = NC이 된다. P완전 문제는 P에서 나란히 하기 '가장 어려운' 문제다. 보기: 회로 값 문제(불 회로 따지기), 선형 계획, 최대 흐름. 이 문제들은 본디 차례로만 되는 듯 보인다. 걸음마다 앞 걸음에 매여 나란히 하기를 버틴다.

---

**연습문제 4.**
NC의 켜는 깊은 배움의 실제 나란한 셈과 어떻게 이어지는가?

??? success "연습문제 4 풀이"
    행렬 곱하기와 겹말기(NC)는 GPU에서 매우 잘 나란해져 이 연산들이 거의 최고 처리량을 내는 까닭이 된다. 되돌이 신경망(때 걸음 사이의 차례 매임)은 P완전 문제에 가까워, 나란한 기계에서 트랜스포머보다 익히기가 느린 까닭이 된다. 트랜스포머는 차례 되돌이를 나란한 스스로 눈길(사실상 행렬 곱하기)로 바꾸어 셈을 'P완전 같은 것'에서 'NC 같은 것'으로 옮겨 엄청난 나란함을 가능하게 했다.