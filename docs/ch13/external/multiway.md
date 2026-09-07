# 여러 갈래 병합

두 갈래 바깥 병합 정렬은 런을 쌍으로 병합하여 데이터를 $O(\log_2(N/M))$번 훑어야 한다. 훑을 때마다 원소를 모두 읽고 쓰므로 훑기의 수를 줄이면 입출력 비용이 곧바로 줄어든다. **여러 갈래 병합**(또는 $k$갈래 병합)은 최소 힙(우선순위 큐)으로 런 $k$개를 한꺼번에 병합하여 로그의 밑을 2에서 $k$로 올리고, 큰 데이터셋에서 훑기의 수를 크게 줄인다.

## 여러 갈래 병합이 중요한 까닭

기억이 $M$이고 블록 크기가 $B$이면 입력 버퍼를 $k = \lfloor M/B \rfloor - 1$개 둘 수 있다(하나는 출력용으로 남긴다). 입력 버퍼마다 서로 다른 런의 블록 하나를 담는다. 런 $k$개를 한꺼번에 병합하면 병합 훑기의 수가 $\lceil \log_2 (N/M) \rceil$에서 $\lceil \log_k (N/M) \rceil$으로 줄어든다.

**보기:** 기억 1GB와 4KB 블록으로 1TB를 정렬하면 $k \approx 250{,}000$이다. 두 갈래 병합은 훑기가 20번쯤 필요하지만 여러 갈래 병합은 한두 번이면 된다.

## 알고리즘 훑어보기

1. **런 만들기**(두 갈래와 같다): 정렬된 런 $\lceil N/M \rceil$개를 만든다.
2. **여러 갈래 병합 훑기:**
    - 읽을 런 $k$개를 열고 저마다 블록 하나씩을 기억으로 불러온다.
    - 블록마다 첫 원소를 크기 $k$의 최소 힙에 넣는다.
    - 힙에서 가장 작은 것을 거듭 꺼내 출력 버퍼에 쓰고, 같은 런에서 다음 원소를 넣는다.
    - 입력 버퍼가 다하면 그 런에서 다음 블록을 읽는다.
    - 출력 버퍼가 차면 디스크에 쓴다.
3. 런이 $\lceil N/M \rceil / k$개 남을 때까지 되풀이한 다음, 필요하면 다시 병합한다.

## 입출력 복잡도

병합 훑기마다 원소 $N$개를 모두 읽고 쓴다. 곧 $2 \lceil N/B \rceil$번의 입출력이다. 병합 인자가 $k = \lfloor M/B \rfloor - 1$이면 훑기의 수는 $\lceil \log_k (N/M) \rceil$이다.

$$
\text{I/O}(N, M, B) = O\!\left(\frac{N}{B} \log_{M/B} \frac{N}{M}\right)
$$

이는 바깥 기억 모형에서 **점근적으로 가장 좋다**고 알려져 있다. 비교 기반 바깥 정렬 알고리즘 가운데 이보다 나은 것은 없다.

병합 걸음마다 기억 안에서 드는 비용은 힙 연산의 $O(\log k)$이며, 전체 견줌 복잡도는 다음과 같다.

$$
T(N, k) = O\!\left(N \log_k \frac{N}{M} \cdot \log k\right) = O(N \log N)
$$

견줌 횟수가 안쪽 정렬과 같아, 병목이 셈이 아니라 입출력임을 확인해 준다.

## 풀이 예제

$M = 3{,}000$, $B = 1{,}000$으로 원소 $N = 12{,}000$개를 정렬해 보자.

**런 만들기:** 원소 3,000개짜리 줄 세운 런 4개. 값: 입출력 $2 \times 12 = 24$ 번.

**합치기:** 들임 버퍼 $k = \lfloor 3{,}000/1{,}000 \rfloor - 1 = 2$개.

- $k = 2$이면 2 번 훑으며 입출력 $2 \times 24 = 48$ 번이 든다. 모두 입출력 72 번.

이제 $B = 500$이라 하면 $k = \lfloor 3{,}000/500 \rfloor - 1 = 5$이다.

- 런 4개가 한 번에 모두 합쳐진다: 입출력 $2 \times 24 = 48$ 번. 모두 입출력 72 번.

기억 자리가 더 크면($M = 6{,}000$) 런이 2개뿐이라 어느 쪽이든 합치기를 한 번만 하면 된다.

## k갈래 병합을 위한 최소 힙

최소 힙은 $(value, run\_index)$ 꼴의 항목을 담는다. 걸음마다 다음을 한다.

1. 최소 꺼내기가 아직 다루지 않은 원소 가운데 전체에서 가장 작은 것을 주고, 그것이 어느 런에서 왔는지 알려 준다.
2. 그 런에서 다음 원소를 넣는다(비용 $O(\log k)$).

모두 $N$ 번 꺼내고 나면 더미가 한 일감은 $O(N \log k)$이다.

## 구현

```python
"""
여러 갈래 병합 — 최소 힙을 써서 정렬된 런 k개를 병합한다.

바깥 정렬의 병합 훑기 횟수를 log_2(N/M)에서 log_k(N/M)으로 줄인다.
여기서 k = M/B - 1이다.
시간:  훑기마다 견줌 O(N log k)번
입출력: 블록 옮김 O((N/B) * log_{M/B}(N/M))번
"""

import heapq
from typing import Iterator


# === 최소 힙을 쓴 k갈래 병합 ================================================

def k_way_merge(sorted_runs: list[list[int]]) -> list[int]:
    """정렬된 목록 k개를 정렬된 목록 하나로 병합한다.

    매개변수
    ----------
    sorted_runs : list[list[int]]
        정렬된 목록(런) k개의 목록.

    반환값
    -------
    list[int]
        병합된 정렬 리스트.
    """
    heap: list[tuple[int, int, int]] = []

    # 런마다 첫 원소로 힙 첫걸음 잡기
    for run_idx, run in enumerate(sorted_runs):
        if run:
            heapq.heappush(heap, (run[0], run_idx, 0))

    result: list[int] = []
    while heap:
        val, run_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # 같은 런에서 다음 원소 밀어 넣기
        next_idx = elem_idx + 1
        if next_idx < len(sorted_runs[run_idx]):
            heapq.heappush(
                heap,
                (sorted_runs[run_idx][next_idx], run_idx, next_idx),
            )

    return result


# === 여러 갈래 병합을 쓴 바깥 정렬 ===========================================

def external_sort_multiway(
    data: list[int], memory_size: int, merge_factor: int
) -> list[int]:
    """여러 갈래 바깥 병합 정렬로 자료를 정렬한다.

    매개변수
    ----------
    data : list[int]
        입력 자료.
    memory_size : int
        한 번에 기억 공간에 들어가는 원소의 최대 개수.
    merge_factor : int
        한꺼번에 병합할 런의 개수(k).

    반환값
    -------
    list[int]
        정렬된 자료.
    """
    # 단계 1: 정렬된 런 만들기
    runs: list[list[int]] = []
    for start in range(0, len(data), memory_size):
        runs.append(sorted(data[start : start + memory_size]))

    # 단계 2: 여러 갈래 병합 훑기
    while len(runs) > 1:
        next_runs: list[list[int]] = []
        for i in range(0, len(runs), merge_factor):
            batch = runs[i : i + merge_factor]
            merged = k_way_merge(batch)
            next_runs.append(merged)
        runs = next_runs

    return runs[0] if runs else []


# === 시연 ===================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    data = random.sample(range(10000), 100)

    # 2갈래 병합
    sorted_2way = external_sort_multiway(data, memory_size=20, merge_factor=2)
    print(f"2-way merge correct: {sorted_2way == sorted(data)}")

    # 5갈래 병합
    sorted_5way = external_sort_multiway(data, memory_size=20, merge_factor=5)
    print(f"5-way merge correct: {sorted_5way == sorted(data)}")

    # 필요한 병합 훑기 보이기
    import math
    num_runs = math.ceil(len(data) / 20)
    for k in [2, 5, 10]:
        passes = math.ceil(math.log(num_runs) / math.log(k)) if num_runs > 1 else 0
        print(f"  k={k:2d}: {passes} merge pass(es) for {num_runs} runs")
```

**출력:**
```
2-way merge correct: True
5-way merge correct: True
  k= 2: 3 merge pass(es) for 5 runs
  k= 5: 1 merge pass(es) for 5 runs
  k=10: 1 merge pass(es) for 5 runs
```

## 실용적인 고려

| 요인 | 영향 |
|--------|--------|
| $k$을 키움 | 훑기는 줄지만 걸음마다 힙의 짐이 는다 |
| $k$이 아주 큼 | 얻는 것이 줄고 디스크 찾기 시간이 우세해진다 |
| SSD과 HDD | 아무 데나 빨리 읽는 SSD은 더 큰 $k$을 견딘다 |
| 이중 버퍼 두기 | 흐름 $k$개마다 입출력과 셈을 겹친다 |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 8장. MIT Press.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.), 5.4절. Addison-Wesley.


## 연습문제

**연습문제 1.**
여러 갈래 병합의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 여러 갈래 병합을 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 여러 갈래 병합이 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
여러 갈래 병합이 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.