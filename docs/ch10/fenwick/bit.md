# 이진 색인 트리

점 갱신도 받는 배열에서 접두사 합을 되풀이해 셈해야 하는 응용이 많다. 순진한 방법은 갱신마다 접두사 합을 $O(n)$에 다시 셈하고, 납작한 접두사 합 배열은 뒤따르는 항목을 모두 고쳐야 하므로 갱신이 $O(n)$이다. **이진 색인 트리**(BIT)는 피터 펜윅의 1994년 논문을 따라 **펜윅 트리**라고도 하며, 색인의 이진 표현을 이용해 두 연산 모두를 $O(\log n)$에 해낸다.

## 핵심 통찰 — 가장 낮은 켜진 비트

BIT의 짜임 전체가 비트를 다루는 요령 하나에 놓여 있다. 어떤 양의 정수 $i$에 대해서도 다음 식이

$$
\text{lowbit}(i) = i \;\&\; (-i)
$$

$i$의 가장 낮은 켜진 비트를 뽑아낸다. 2의 보수 산술에서 $-i$은 $i$의 비트 보수에 1을 더한 것이므로, AND 연산이 가장 오른쪽의 `1`만 빼고 모든 비트를 0으로 만든다.

!!! example "가장 낮은 켜진 비트의 보기"
    | $i$ (십진) | $i$ (이진) | $i \;\&\; (-i)$ | 결과 |
    |:---:|:---:|:---:|:---:|
    | 6 | `110` | `010` | 2 |
    | 12 | `1100` | `0100` | 4 |
    | 7 | `0111` | `0001` | 1 |

## 트리의 짜임

BIT는 (1부터 세는) 납작한 배열 `tree[1..n]`에 담긴다. 자리 $i$마다 본디 배열의 한 범위를 맡는다.

$$
\texttt{tree}[i] = \sum_{j = i - \text{lowbit}(i) + 1}^{i} a[j]
$$

달리 말해 `tree[i]`은 자리 $i$에서 끝나는 꼭 $\text{lowbit}(i)$개 원소의 합을 담는다. 곧 다음과 같다.

- `tree[1]`은 `a[1]`을 덮는다 ($\text{lowbit}(1)=1$이므로 원소 1개).
- `tree[2]`은 `a[1..2]`을 덮는다 ($\text{lowbit}(2)=2$이므로 원소 2개).
- `tree[3]`은 `a[3]`을 덮는다 (원소 1개).
- `tree[4]`은 `a[1..4]`을 덮는다 ($\text{lowbit}(4)=4$이므로 원소 4개).

색인마다 비트가 많아야 $\lfloor \log_2 n \rfloor$개이므로 이 숨은 트리의 층은 $O(\log n)$개이다.

## 연산

### 접두사 질의

$\text{prefix}(i) = \sum_{j=1}^{i} a[j]$을 셈하려면 가장 낮은 켜진 비트를 없애 가며 `tree` 값을 쌓는다.

$$
\text{prefix}(i) = \texttt{tree}[i] + \texttt{tree}[i - \text{lowbit}(i)] + \cdots
$$

단계마다 $i$의 비트가 적어도 하나 줄므로 고리가 많아야 $\lfloor \log_2 n \rfloor$번 돈다.

### 점 갱신

자리 $i$에 값 $\delta$을 더하려면 가장 낮은 켜진 비트를 **더해** 가며 $i$을 범위에 담는 모든 트리 노드를 고친다.

$$
i \;\leftarrow\; i + \text{lowbit}(i)
$$

$i > n$이 될 때까지 숨은 트리를 거슬러 올라간다.

## 구현

```python
"""
이진 색인 트리 (펜윅 트리).

가장 낮은 켜진 비트 요령으로 1부터 세는 배열에서
O(log n)의 점 갱신과 접두사 합 질의를 준다.
"""


# === 펜윅 트리 클래스 ===

class FenwickTree:
    """점 갱신과 접두사 질의를 받쳐 주는 이진 색인 트리."""

    def __init__(self, n: int):
        """크기 n의 BIT를 모두 0으로 시작한다."""
        self.n = n
        self.tree = [0] * (n + 1)  # 1부터 센다

    def update(self, i: int, delta: int) -> None:
        """자리 i에 delta를 더한다. 모든 조상으로 퍼진다."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 부모로 옮긴다

    def query(self, i: int) -> int:
        """색인 1부터 i까지의 접두사 합을 돌려준다."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)  # 가장 낮은 켜진 비트를 뗀다
        return s

    def range_query(self, l: int, r: int) -> int:
        """색인 l부터 r까지(둘 다 포함)의 합을 돌려준다."""
        return self.query(r) - self.query(l - 1)

    def build(self, arr: list) -> None:
        """0부터 세는 배열로 BIT를 O(n) 시간에 세운다."""
        for i, v in enumerate(arr, 1):
            self.update(i, v)


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    ft = FenwickTree(len(data))
    ft.build(data)

    print(f"Array: {data}")
    print(f"Prefix sum [1..3]: {ft.query(3)}")
    print(f"Range sum [2..5]:  {ft.range_query(2, 5)}")

    # 점 갱신: 자리 3에 10을 더한다
    ft.update(3, 10)
    print(f"After adding 10 to position 3:")
    print(f"Prefix sum [1..3]: {ft.query(3)}")
    print(f"Range sum [2..5]:  {ft.range_query(2, 5)}")
```

**출력:**
```
Array: [1, 3, 5, 7, 9]
Prefix sum [1..3]: 9
Range sum [2..5]:  24
After adding 10 to position 3:
Prefix sum [1..3]: 19
Range sum [2..5]:  34
```

## 복잡도 분석

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 세우기 | $O(n \log n)$ | $O(n)$ |
| 점 갱신 | $O(\log n)$ | $O(1)$ |
| 접두사 질의 | $O(\log n)$ | $O(1)$ |
| 범위 질의 | $O(\log n)$ | $O(1)$ |

원소마다 바로 위 부모로 한 번에 아래에서 위로 퍼뜨리면 $O(n)$에 세울 수도 있다.

!!! tip "언제 BIT를 쓰는가"
    1차원 배열에서 **점 갱신**과 **접두사 합 질의**가 필요할 때 BIT가 알맞다. 구간 트리의 절반쯤 되는 기억을 쓰고 상수 배도 작다. 다만 범위 갱신이나 (범위 최솟값처럼) 되돌릴 수 없는 연산이 필요하면 게으른 전파를 갖춘 구간 트리가 더 알맞다.

## 참고 문헌

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
이진 색인 트리의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 이진 색인 트리를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
이진 색인 트리가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.