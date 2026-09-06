# 나란한 앞자락 합

**앞자락 합**(*훑기*라고도 한다)은 배열의 부분 합을 모두 셈한다. $[a_0, a_1, \ldots, a_{n-1}]$이 주어지면 $[a_0, a_0 + a_1, \ldots, a_0 + a_1 + \cdots + a_{n-1}]$을 낸다. 이 연산은 짐 고르기와 흐름 다지기부터 줄 세우기와 그래프 알고리즘까지 나란한 셈 어디에나 나온다. 차례 훑기는 $O(n)$ 때가 들지만 나란한 앞자락 합은 일 $O(n)$으로 뻗음 $O(\log n)$을 이루어 가장 중요한 나란한 원소 가운데 하나가 된다.

## 문제의 정의

배열 $A = [a_0, a_1, \ldots, a_{n-1}]$과 결합이 되는 두 자리 연산자 $\oplus$이 주어질 때 **자기를 담는 앞자락 합**은 다음을 낸다:

$$
S[i] = \bigoplus_{k=0}^{i} a_k = a_0 \oplus a_1 \oplus \cdots \oplus a_i
$$

**자기를 빼는 앞자락 합**은 결과를 한 자리 옮긴다:

$$
S[i] = \bigoplus_{k=0}^{i-1} a_k
$$

$S[0]$은 $\oplus$의 항등원으로 둔다.

## 블렐로크 알고리즘(일 효율 좋은 훑기)

블렐로크 알고리즘은 배열 위에 세운 고른 두 갈래 나무 얼개에서 두 마당으로 앞자락 합을 셈한다.

### 1마당: 위로 쓸기(줄임)

잎에서 뿌리로 나무를 훑으며 안쪽 마디마다 부분 합을 셈한다. (잎에서 센) 켜 $d$에서 마디마다 제 부분 나무의 합을 담는다:

$$
\text{tree}[i] \leftarrow \text{tree}[i] \oplus \text{tree}[i - 2^d]
$$

위로 쓸기가 끝나면 뿌리가 온 합을 가진다.

### 2마당: 아래로 쓸기(나눔)

뿌리를 항등원으로 두고 뿌리에서 잎으로 훑는다. 마디마다 앞자락 합을 아래로 퍼뜨린다:

$$
\text{left} \leftarrow \text{parent}, \quad
\text{right} \leftarrow \text{parent} \oplus \text{old\_left}
$$

아래로 쓸기가 끝나면 자리마다 자기를 뺀 앞자락 합을 가진다.

### 일-뻗음 살피기

- **일**: 마당마다 연산을 $O(n)$번 한다. 모두: $T_1 = O(n)$.
- **뻗음**: 마당마다 켜가 $O(\log n)$개다. 모두: $T_\infty = O(\log n)$.
- **나란함**: $P = O(n / \log n)$.

## 구현

```python
"""
Parallel prefix sum (Blelloch's work-efficient scan).

Simulates the up-sweep (reduce) and down-sweep (distribute)
나란한 앞자락 합 알고리즘의 마당.
"""

import math

# ===================================================================
# 블렐로크의 일 효율 좋은 훑기
# ===================================================================

def parallel_prefix_sum(arr):
    """Compute exclusive prefix sum using Blelloch's algorithm.

    인수:
        arr: 수의 들임 배열

    반환값:
        자기를 뺀 앞자락 합 배열
    """
    n = len(arr)
    # 다음 2의 거듭제곱까지 채운다
    size = 1 << math.ceil(math.log2(max(n, 2)))
    tree = list(arr) + [0] * (size - n)

    # 위로 쓸기(줄임 마당)
    stride = 1
    while stride < size:
        for i in range(2 * stride - 1, size, 2 * stride):
            tree[i] += tree[i - stride]
        stride *= 2

    # 자기를 뺀 훑기를 위해 뿌리를 0으로 둔다
    tree[size - 1] = 0

    # 아래로 쓸기(나눔 마당)
    stride = size // 2
    while stride >= 1:
        for i in range(2 * stride - 1, size, 2 * stride):
            temp = tree[i - stride]
            tree[i - stride] = tree[i]
            tree[i] += temp
        stride //= 2

    return tree[:n]


def inclusive_prefix_sum(arr):
    """자기를 담는 앞자락 합을 셈한다.

    인수:
        arr: 수의 들임 배열

    반환값:
        자기를 담는 앞자락 합 배열
    """
    exclusive = parallel_prefix_sum(arr)
    return [exclusive[i] + arr[i] for i in range(len(arr))]

# ===================================================================
# 차례 앞자락 합(견주기용)
# ===================================================================

def sequential_prefix_sum(arr):
    """자기를 담는 앞자락 합을 차례로 셈한다.

    인수:
        arr: 수의 들임 배열

    반환값:
        자기를 담는 앞자락 합 배열
    """
    result = []
    running = 0
    for x in arr:
        running += x
        result.append(running)
    return result

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    arr = [3, 1, 7, 0, 4, 1, 6, 3]

    exclusive = parallel_prefix_sum(arr)
    inclusive = inclusive_prefix_sum(arr)
    sequential = sequential_prefix_sum(arr)

    print(f"Input:            {arr}")
    print(f"Exclusive prefix: {exclusive}")
    print(f"Inclusive prefix:  {inclusive}")
    print(f"Sequential check: {sequential}")
    print(f"Match: {inclusive == sequential}")

    # 일-뻗음 살피기
    n = len(arr)
    work = 2 * n
    span = 2 * math.ceil(math.log2(n))
    print(f"\nn = {n}")
    print(f"Work O(n) ~ {work}")
    print(f"Span O(log n) ~ {span}")
    print(f"Parallelism ~ {work / span:.1f}")
```

**출력:**
```
Input:            [3, 1, 7, 0, 4, 1, 6, 3]
Exclusive prefix: [0, 3, 4, 11, 11, 15, 16, 22]
Inclusive prefix:  [3, 4, 11, 11, 15, 16, 22, 25]
Sequential check: [3, 4, 11, 11, 15, 16, 22, 25]
맞는가: True

n = 8
일 O(n) ~ 16
뻗음 O(log n) ~ 6
나란함 ~ 2.7
```

## 힐리스-스틸 알고리즘

논리는 더 단순하지만 일이 더 많은 다른 길:

걸음 $d = 0, 1, \ldots, \lceil \log_2 n \rceil - 1$마다 $i \ge 2^d$인 번호마다:

$$
a_i^{(d+1)} = a_i^{(d)} + a_{i - 2^d}^{(d)}
$$

- **일**: $T_1 = O(n \log n)$(일 효율이 좋지 않다).
- **뻗음**: $T_\infty = O(\log n)$.
- **나란함**: $O(n)$.

!!! tip "알고리즘 고르기"
    블렐로크 알고리즘은 일 효율이 좋고(일 $O(n)$) 셈틀 수가 한정될 때 낫다. 힐리스-스틸은 온 일이 더 많지만 맞추기가 단순해 나란함이 넉넉한 GPU 얼개에 알맞다.

## 응용

- **흐름 다지기**: 조건을 만족하는 원소를 거른다. 앞자락 합이 남길 원소마다 내놓기 자리를 셈한다.
- **기수 정렬**: 자릿수마다의 세기 정렬이 앞자락 합으로 내놓기 자리를 나란히 셈한다.
- **짐 고르기**: 일감 크기의 앞자락 합을 셈해 일을 고르게 나눈다.
- **나란한 너비 우선 찾기**: 다음 켜의 앞자락 자리 옮김을 셈한다.

## 복잡도 요약

| 알고리즘 | 일 $T_1$ | 뻗음 $T_\infty$ | 나란함 |
|---|---|---|---|
| 차례 훑기 | $O(n)$ | $O(n)$ | $O(1)$ |
| 블렐로크(일 효율 좋음) | $O(n)$ | $O(\log n)$ | $O(n / \log n)$ |
| 힐리스-스틸 | $O(n \log n)$ | $O(\log n)$ | $O(n)$ |

## 참고 문헌

- Blelloch, G. E. (1990). "Prefix sums and their applications." *Tech Report CMU-CS-90-190*.
- Hillis, W. D. and Steele, G. L. (1986). "Data parallel algorithms." *Communications of the ACM*, 29(12), 1170--1183.


## 연습문제

**연습문제 1.**
나란한 앞자락 합(훑기) 연산을 뜻매김하고 그 일-뻗음 살피기를 보여라.

??? success "연습문제 1 풀이"
    앞자락 합: $[a_1, \ldots, a_n]$이 주어지면 $[a_1, a_1+a_2, \ldots, \sum_{i=1}^n a_i]$을 셈한다. 블렐로크 훑기는 두 마당을 쓴다. (1) 위로 쓸기(줄임): 부분 합의 두 갈래 나무를 세운다. 일 $O(n)$, 깊이 $O(\log n)$. (2) 아래로 쓸기: 앞자락을 나무 아래로 퍼뜨린다. 일 $O(n)$, 깊이 $O(\log n)$. 모두: $W = O(n)$, $D = O(\log n)$. 이는 일 효율이 좋다(차례의 $O(n)$과 맞는다).

---

**연습문제 2.**
알고리즘에서 나란한 앞자락 합의 쓰임새 셋을 들어라.

??? success "연습문제 2 풀이"
    (1) **나란한 다지기/거르기**: 참거짓 가리개가 주어지면 앞자락 합으로 내놓기 자리를 정한 뒤 흩뿌린다. 일 $O(n)$, 깊이 $O(\log n)$. (2) **나란한 기수 정렬**: 비트 자리마다 앞자락 합으로 0비트와 1비트가 갈 곳을 셈한다. (3) **다항식 따지기**: 호너 방법은 차례지만 앞자락 합으로 여러 점에서 나란히 따질 수 있다. 그 밖의 쓰임: 매김 셈하기, 막대 그림 짓기, 짐 고르기.

---

**연습문제 3.**
앞자락 합이 왜 GPU 셈의 근본 원소로 여겨지는지 밝혀라.

??? success "연습문제 3 풀이"
    앞자락 합은 이어 쌓는 값을 가진 되돌이의 GPU 판이다. '원소마다 이어 쌓는 값을 고친다' 꼴의 차례 되돌이는 거의 모두 훑기로 나란히 할 수 있다. GPU 기계는 다듬은 훑기 짜기를 준다(CUB, Thrust). 깊이가 $O(\log n)$이라 코어가 수천 개인 GPU에서 엄청난 나란함이 나온다. 훑기는 거의 모든 GPU 알고리즘 속에 쓰인다. 줄 세우기, 찾기, 성긴 행렬 연산, 막대 그림, 흐름 다지기, 기억 나누기가 그것이다.

---

**연습문제 4.**
트랜스포머의 눈길 셈에 앞자락 합이 어떻게 쓰이는지 밝혀라.

??? success "연습문제 4 풀이"
    선형 눈길 변형(보기로 퍼포머, 선형 트랜스포머)에서는 $O(n^2)$인 소프트맥스를 피하려 눈길 얼개를 다시 적는다. 대신 특징을 쏘아 앞자락 합으로 쌓는다. $o_i = \sum_{j \leq i} \phi(q_i)^T \phi(k_j) v_j$을 바깥 곱 $\phi(k_j) v_j^T$에 대한 앞자락 합으로 적을 수 있다. 이는 나란한 훑기로 눈길을 $O(n^2)$에서 $O(n)$으로 줄여 선형 때 차례 나타내기를 가능하게 한다. 앞자락 합은 쌓인 손실 셈하기와 고르게 맞추기의 이어지는 통계에도 나온다.