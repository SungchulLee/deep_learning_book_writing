# 정렬 망

보통의 정렬 알고리즘은 데이터에 기대어 판단한다. 다음에 어느 원소를 견줄지가 앞선 견줌의 결과에 달렸다. **정렬 망**은 컴파일 때 견줌의 차례를 붙박아 이 기댐을 없앤다. 입력 값과 상관없이 같은 견주고 맞바꾸기 연산을 밟는다. 그래서 같은 "깊이"의 비교기를 모두 한꺼번에 돌릴 수 있는 병렬 정렬과 하드웨어 정렬에 정렬 망이 자연스러운 추상이 된다.

---

## 1. 비교기

정렬 망의 기본 벽돌은 **비교기**이며, 입력 둘을 받아 정렬된 차례로 내보내는 장치이다.

$$
\text{comparator}(a, b) = (\min(a, b),\; \max(a, b))
$$

줄 $i$과 $j$($i < j$)을 잇는 비교기는 줄 $i$의 값이 줄 $j$의 값보다 크면 두 값을 맞바꾼다.

---

## 2. 정렬 망의 정의

원소 $n$개를 위한 정렬 망은 짝 $(n, S)$이며, $S$은 $0 \le i_k < j_k < n$인 비교기의 열 $\{(i_1, j_1), (i_2, j_2), \dots, (i_m, j_m)\}$이다.

줄 세우기 그물을 가르는 고갱이 자는 둘이다.

| 지표 | 정의 |
|--------|-----------|
| **크기** | 비교기의 총 개수 $m$ |
| **깊이** | 병렬 걸음의 수(서로 겹치지 않는 줄의 비교기는 한꺼번에 돌 수 있다) |

깊이가 나란한 시간 복잡도를 정하고 크기가 온 일감을 정한다.

---

## 3. 0-1 원리

**0-1 원리**는 정렬 망이 옳음을 증명하는 근본 도구이다.

**정리.** 비교기 망이 0과 1로 이루어진 모든 입력 수열을 옳게 정렬하면, 아무 값으로 이루어진 모든 입력 수열도 옳게 정렬한다.

*증명 얼개.* 망이 어떤 입력 $a_0, \dots, a_{n-1}$에서 무너진다고 하자. 그러면 출력에서 $a_i > a_j$인 첨자 $i < j$이 있다. $f(x) = \mathbf{1}[x \ge a_j]$이라 하자. $f$이 단조이므로 입력마다 $f$을 씌워도 견주고 맞바꾸는 거동이 그대로 지켜진다. 그런데 $\{0, 1\}$ 입력 $f(a_0), \dots, f(a_{n-1})$의 출력에는 1이 0보다 앞서 나와, 모든 $\{0,1\}$ 입력이 옳게 정렬된다는 가정에 어긋난다. $\square$

0-1 원리는 확인해야 할 입력을 무한히 많은 것에서 이진 입력 $2^n$개로 줄인다.

---

## 4. 흔한 정렬 망

### 끼워넣기 정렬 망

끼워넣기 정렬을 흉내 낸 비교기를 쓴다. 원소 $i$마다 원소 $i-1, i-2, \dots, 0$과 견준다.

- **깊이:** $O(n)$
- **크기:** $O(n^2)$

### 홀짝 자리바꿈 정렬

홀짝 쌍 $(1,2), (3,4), \dots$과 짝홀 쌍 $(0,1), (2,3), \dots$을 견주기를 $n$번 번갈아 한다.

- **깊이:** $O(n)$
- **크기:** $O(n^2)$

### 바이토닉 정렬 망

바이토닉 병합의 짜임을 쓴다.

- **깊이:** $O(\log^2 n)$
- **크기:** $O(n \log^2 n)$

### AKS 망(가장 좋은 깊이)

아이타이, 코믈로시, 세메레디는 다음을 지닌 줄 세우기 그물이 있음을 밝혔다.

- **깊이:** $O(\log n)$
- **크기:** $O(n \log n)$

이는 점근적으로 가장 좋지만 상수 인자가 엄청나서 실제로 쓸 수 없다. 그래도 정렬 분야의 위대한 이론적 결과 가운데 하나로 남아 있다.

---

## 5. 아래 한계

원소 $n$개를 다루는 어떤 줄 세우기 그물도 다음을 지녀야 한다.

$$
\text{depth} \ge \lceil \log_2 n \rceil
$$

병렬 걸음마다 가려낼 수 있는 서로 다른 차례의 수가 많아야 두 배가 되기 때문이다.

$$
\text{size} \ge \lceil \log_2(n!) \rceil \approx n \log_2 n - 1.44n
$$

이는 정보 이론의 아래 한계에서 따라 나온다. 가능한 순열이 $n!$개이면 적어도 $\log_2(n!)$번의 이진 판단이 필요하다.

---

## 6. 구현

```python
"""
정렬 망 — 병렬 정렬을 위한 붙박이 견줌 차례.

정렬 망 여럿을 구현하고 0-1 원리로
그것들을 확인한다.
시간:  망에 달렸다(낱낱의 분석을 보아라)
공간: O(n) — 맞바꾸기로 제자리에서
"""

# === 비교기 =================================================================

def compare_and_swap(arr: list, i: int, j: int) -> None:
    """arr[i] > arr[j]이면 arr[i]과 arr[j]을 맞바꾼다."""
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]

# === 홀짝 자리바꿈 정렬 망 ===================================================

def odd_even_sort_network(n: int) -> list[list[tuple[int, int]]]:
    """홀짝 자리바꿈 정렬의 비교기 차례를 만든다.

    서로 겹치지 않는 비교기를 담은 병렬 단계의 목록을 되돌린다.
    """
    stages: list[list[tuple[int, int]]] = []
    for phase in range(n):
        stage: list[tuple[int, int]] = []
        if phase % 2 == 0:
            # 짝수 단계: (0,1), (2,3), ... 견주기
            for i in range(0, n - 1, 2):
                stage.append((i, i + 1))
        else:
            # 홀수 단계: (1,2), (3,4), ... 견주기
            for i in range(1, n - 1, 2):
                stage.append((i, i + 1))
        if stage:
            stages.append(stage)
    return stages

# === 정렬 망 쓰기 ===========================================================

def apply_network(
    arr: list[int], network: list[list[tuple[int, int]]]
) -> list[int]:
    """*arr*에 정렬 망을 쓰고 정렬된 결과를 되돌린다."""
    result = list(arr)
    for stage in network:
        for i, j in stage:
            compare_and_swap(result, i, j)
    return result

# === 0-1 원리로 확인하기 =====================================================

def verify_network(n: int, network: list[list[tuple[int, int]]]) -> bool:
    """0-1 원리로 정렬 망을 확인한다.

    2^n개의 이진 입력을 모두 시험한다.
    """
    for bits in range(1 << n):
        arr = [(bits >> i) & 1 for i in range(n)]
        result = apply_network(arr, network)
        if result != sorted(arr):
            return False
    return True

# === 시연 ===================================================================

if __name__ == "__main__":
    n = 8
    network = odd_even_sort_network(n)
    print(f"Odd-even transposition sort for n={n}:")
    print(f"  Depth (parallel stages): {len(network)}")
    print(f"  Size (total comparators): {sum(len(s) for s in network)}")
    print(f"  Verified (0-1 principle): {verify_network(n, network)}")

    # 보기 입력으로 시험하기
    data = [3, 7, 4, 8, 6, 2, 1, 5]
    sorted_data = apply_network(data, network)
    print(f"\n  Input:  {data}")
    print(f"  Output: {sorted_data}")

    # 확인을 위한 더 작은 보기
    n_small = 6
    net_small = odd_even_sort_network(n_small)
    print(f"\nOdd-even for n={n_small}:")
    print(f"  Verified: {verify_network(n_small, net_small)}")
```

**출력:**
```
Odd-even transposition sort for n=8:
  Depth (parallel stages): 8
  Size (total comparators): 28
  Verified (0-1 principle): True

  Input:  [3, 7, 4, 8, 6, 2, 1, 5]
  Output: [1, 2, 3, 4, 5, 6, 7, 8]

Odd-even for n=6:
  Verified: True
```

---

## 7. 알려진 정렬 망 간추림

| 망 | 깊이 | 크기 | 실전에 쓰나? |
|---------|-------|------|-----------|
| 끼워넣기 | $O(n)$ | $O(n^2)$ | 작은 $n$에서만 |
| 홀짝 자리바꿈 | $O(n)$ | $O(n^2)$ | 단순함, 작은 $n$ |
| 바이토닉(배처) | $O(\log^2 n)$ | $O(n \log^2 n)$ | 예 — GPU |
| 홀짝 병합(배처) | $O(\log^2 n)$ | $O(n \log^2 n)$ | 예 |
| AKS | $O(\log n)$ | $O(n \log n)$ | 아니오 — 상수가 엄청나다 |
| 가장 좋은 것(굿리치) | $O(\log n)$ | $O(n \log n)$ | 이론상으로만 |

---

## 연습문제

**연습문제 1.**
정렬 망의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 정렬 망을 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 정렬 망이 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
정렬 망이 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.

## 정리하며

이 마당은 비교기、정렬 망의 정의、0-1 원리、흔한 정렬 망을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 27장. MIT Press.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.), 5.3.4절. Addison-Wesley.
