# PRAM 모형

차례 알고리즘 살피기에 RAM 모형이 바탕이 되듯, 나란한 알고리즘을 살피려면
엄밀한 셈 모형이 필요하다. **나란한 아무 닿기 기계(PRAM)**는 RAM 모형을
공통 기억을 함께 쓰는 셈틀 여럿으로 넓혀, 나란한 때와 일 복잡도를 따지는
깔끔한 추상을
준다.

## 정의

PRAM은 다음으로 이루어진다:

- 저마다 제 자리 기억을 가진 셈틀 $p$개 $P_1, P_2, \dots, P_p$.
- 크기에 제한이 없는 공유 온 자리 기억.
- 모든 셈틀을 맞추는 공통 시계.

때 걸음마다 셈틀마다 다음을 할 수 있다:

1. 공유 기억에서 칸 하나를 읽는다.
2. 그 자리 셈을 한 번 한다.
3. 공유 기억에 칸 하나를 적는다.

모든 셈틀이 같은 프로그램을 돌리되 자기 번호에 따라 갈라진다.

## PRAM의 변형

여러 셈틀이 같은 기억 칸에 한꺼번에 닿을 때 움직임은 PRAM 변형에
매인다:

| 변형 | 함께 읽기 | 함께 적기 | 힘 |
|---|---|---|---|
| **EREW**(홀로 읽기, 홀로 적기) | 아니오 | 아니오 | 가장 약함 |
| **CREW**(함께 읽기, 홀로 적기) | 예 | 아니오 | 보통 |
| **CRCW**(함께 읽기, 함께 적기) | 예 | 예 | 가장 셈 |

### CRCW의 적기 다툼 풀기

CRCW에서 여러 셈틀이 같은 칸에 적을 때:

- **공통:** 적는 셈틀이 모두 같은 값을 적어야 한다.
- **아무거나:** 적는 이 가운데 아무 하나가 이긴다.
- **앞섬:** 번호가 가장 작은 셈틀이 이긴다.

!!! note "서로의 힘"
    CRCW PRAM에서 $T$ 때에 도는 어떤 알고리즘도 EREW PRAM에서 $O(T \log p)$ 때에
    흉내 낼 수 있다. 그러니 변형끼리는 많아야 로그 인수만큼
    다르다.

## 복잡도 잣대

셈틀 $p$개를 쓰는 PRAM 알고리즘에서:

- **나란한 때** $T_p$: 맞춘 걸음의 수.
- **일** $W = p \cdot T_p$: 모든 셈틀과 모든 걸음을 통틀은 온 연산
  개수.
- **비용** $C = p \cdot T_p$: 일과 같다(바꿔 쓴다).

$T_1$이 그 문제의 가장 좋은 차례 때일 때 $W = O(T_1)$이면 알고리즘이
**일 효율이 좋다**고 한다.

### 브렌트 정리

**정리(브렌트).** 나란한 때가 $T$이고 일이 $W$인 어떤 PRAM 알고리즘도 셈틀
$p$개에서 다음 때에 돌릴 수 있다

$$
T_p = O\!\left(\frac{W}{p} + T\right)
$$

이 가둠은 일 효율이 좋은 알고리즘($W = T_1$)이 $p \le T_1 / T$일 때 거의
가장 좋은 빨라짐을 이룸을 보인다.

## 보기: 나란한 합

수 $n$개의 합을 셈하는 일이 PRAM 알고리즘 설계를 잘 보여 준다.

### 두 갈래 나무 줄임

원소를 짝지어 나란히 더하며 걸음마다 문제 크기를 반으로 줄인다:

$$
T = O(\log n), \quad p = n/2, \quad W = O(n)
$$

차례 합이 $O(n)$이므로 이는 일 효율이 좋다.

```python
"""
두 갈래 나무 줄임을 쓰는 PRAM 나란한 합 흉내내기.

나란히 셈 때: O(log n)
일: O(n) — 일 효율이 좋다
"""

import math


# === 나란한 합(흉내) ===
def parallel_sum(arr: list[int]) -> int:
    """PRAM에서 두 갈래 나무 줄임을 흉내 낸다."""
    n = len(arr)
    if n == 0:
        return 0

    # 2의 거듭제곱까지 채운다
    size = 1 << math.ceil(math.log2(max(n, 1)))
    data = list(arr) + [0] * (size - n)

    steps = 0
    stride = 1
    while stride < size:
        # "셈틀"마다 더하기 하나를 셈한다
        new_data = list(data)
        for i in range(0, size, 2 * stride):
            new_data[i] = data[i] + data[i + stride]
        data = new_data
        stride *= 2
        steps += 1

    return data[0]


# === 나란한 최대 ===
def parallel_max(arr: list[int]) -> int:
    """두 갈래 나무 줄임으로 PRAM 나란한 최대를 흉내 낸다."""
    n = len(arr)
    if n == 0:
        return float("-inf")

    size = 1 << math.ceil(math.log2(max(n, 1)))
    data = list(arr) + [float("-inf")] * (size - n)

    stride = 1
    while stride < size:
        new_data = list(data)
        for i in range(0, size, 2 * stride):
            new_data[i] = max(data[i], data[i + stride])
        data = new_data
        stride *= 2

    return data[0]


# === 보기 ===
if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Input: {data}")
    print(f"Parallel sum: {parallel_sum(data)}")
    print(f"Parallel max: {parallel_max(data)}")
    print(f"Steps: {math.ceil(math.log2(len(data)))}")
```

## PRAM과 다른 모형

| 모형 | 공유 기억 | 맞춤 | 주고받기 비용 |
|---|---|---|---|
| PRAM | 예 | 예 | 공짜(낱덩이 비용) |
| BSP | 예(큰 걸음으로) | 울타리로 맞춤 | 드러남 |
| LogP | 아니오(쪽지 건네기) | 아니오 | 지체 + 대역 |
| 일-뻗음 | 숨은 방향 있고 돌이 없는 그래프 | 예(갈라짐/합침) | 공짜 |

!!! warning "PRAM의 한계"
    PRAM은 셈틀 수와 상관없이 공유 기억 닿기가 낱덩이 비용이라고 여긴다.
    실제 기계는 기억 다툼, 두름 일관성 덧짐, 고르지 않은 닿기 지체(NUMA)를
    마주한다. PRAM 살피기는 나란한 때의 쓸모 있는 아래 가둠을 주지만 실제
    빨라짐을 크게
    볼 수 있다.

## 핵심 문제와 그 PRAM 복잡도

| 문제 | 때 | 셈틀 | 일 효율이 좋은가? |
|---|---|---|---|
| 합/줄임 | $O(\log n)$ | $O(n)$ | 예 |
| 앞자락 합 | $O(\log n)$ | $O(n)$ | 예 |
| 줄 세우기(콜) | $O(\log n)$ | $O(n)$ | 예 |
| 목록 매기기 | $O(\log n)$ | $O(n)$ | 예 |
| 행렬 곱하기 | $O(\log n)$ | $O(n^3)$ | 예 |
| 이어진 조각 | $O(\log^2 n)$ | $O(n^2)$ | 아니오 |

## 참고 문헌

- JaJa, J. *An Introduction to Parallel Algorithms*. Addison-Wesley, 1992.
- Karp, R. M. & Ramachandran, V. "Parallel Algorithms for Shared-Memory
  Machines." *Handbook of Theoretical Computer Science*, Vol. A, 1990.


## 연습문제

**연습문제 1.**
PRAM 모형과 그 변형(EREW, CREW, CRCW)을 뜻매김하여라.

??? success "연습문제 1 풀이"
    나란한 아무 닿기 기계(PRAM)는 온 자리 기억을 함께 쓰는 셈틀 $p$개를 가진다. 걸음마다 셈틀마다 기억에서 읽고 셈하고 적는다. 변형은 함께 닿기를 다룬다. EREW(홀로 읽기 홀로 적기): 같은 칸에 한꺼번에 닿을 수 없다. CREW(함께 읽기 홀로 적기): 여럿이 읽어도 되지만 적기는 홀로. CRCW(함께 읽기 함께 적기): 둘 다 함께 되며 적기 다툼 풀기(앞섬, 공통, 아무거나)를 둔다. EREW이 가장 약하고 CRCW이 가장 세다.

---

**연습문제 2.**
$T$ 때에 도는 어떤 CRCW-PRAM 알고리즘도 EREW-PRAM에서 $O(T \log p)$ 때에 흉내 낼 수 있음을 밝혀라.

??? success "연습문제 2 풀이"
    CRCW의 함께 읽기와 적기는 줄 세우기로 EREW에서 흉내 낼 수 있다. 칸 $x$을 함께 읽는 경우: (1) $x$이 필요한 셈틀마다 요청을 배열에 적고, (2) 요청을 칸 주소로 줄 세우고(EREW에서 $O(\log p)$), (3) 칸마다 첫 셈틀이 실제로 읽어 두 갈래 나무로 널리 알린다($O(\log p)$). 함께 적기: 칸으로 줄 세우고 다툼 풀기를 쓴 뒤 적는다. CRCW 걸음마다 EREW 걸음 $O(\log p)$번이 든다. $T$걸음이면 $O(T \log p)$이다.

---

**연습문제 3.**
브렌트 정리는 무엇이며 일과 때를 어떻게 잇는가?

??? success "연습문제 3 풀이"
    브렌트 정리: 일이 $W$이고 깊이(뻗음)가 $D$인 나란한 알고리즘은 셈틀 $p$개에서 때 $T \leq W/p + D$에 돌릴 수 있다. 일 $W$ = 온 연산, 깊이 $D$ = 가장 긴 차례 매임 사슬. 이는 가장 좋은 값이다. $W/p$(온 일을 나눔)도 $D$(핵심 길)도 넘어설 수 없다. $p = W/D$이면 $T = O(D)$이고 선형 빨라짐을 이룬다. 브렌트 정리가 일-뻗음 모형을 쓸모 있는 추상으로 뒷받침한다.

---

**연습문제 4.**
PRAM 모형과 요즘 GPU 얼개를 견주어라. PRAM은 무엇을 담고 무엇을 놓치는가?

??? success "연습문제 4 풀이"
    PRAM이 담는 것: (1) 셈틀 여럿이 한꺼번에 일함, (2) 공유 기억 닿기, (3) 맞춘 돌림(모든 셈틀이 발맞춤). PRAM이 놓치는 것: (1) 기억의 켜(L1/L2/L3 두름, 온 자리 기억, 공유 기억 --- GPU 성능은 기억 닿기 무늬에 크게 매인다), (2) 기억 대역의 한계(PRAM은 낱덩이 비용 닿기를 여긴다), (3) 다발 단위 돌림(GPU의 실은 32개씩 묶여 돌고 갈래가 갈라지면 차례로 돌아간다), (4) 실 덩이 사이의 주고받기 비용. 요즘 GPU 알고리즘은 기억 뭉치기, 채움률, 뱅크 다툼을 다듬어야 한다.