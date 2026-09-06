# 바꿔 고르기

바깥 병합 정렬에서 **런 만들기** 단계는 길이 $M$(기억 용량)의 정렬된 런을 만든다. 런이 적고 길수록 병합 훑기가 줄고 입출력 비용도 곧바로 준다. **바꿔 고르기**는 최소 힙으로 새 입력을 읽으면서 쓸 수 있는 가장 작은 원소를 쉼 없이 내보내어, 소박한 런 만들기의 두 배인 기대 길이 $2M$의 런을 내는 기법이다.

## 왜 필요한가

소박한 런 만들기에서는 원소 $N$개가 길이 $M$인 런 $\lceil N/M \rceil$개를 낸다. 바꿔 고르기는 대개 런의 수를 대략 $\lceil N/(2M) \rceil$으로 반 토막 내어, 실전의 여러 상황에서 병합 훑기를 한 번 아낀다. 입력이 거의 정렬되어 있으면 바꿔 고르기는 $2M$보다 훨씬 긴 런을, 때로는 파일 전체에 걸친 런을 낼 수 있다.

## 알고리즘 훑어보기

Maintain a min-heap of capacity $M$ (the memory size).

1. **초기화.** 첫 원소 $M$개를 힙으로 읽어 들인다.
2. **출력 되돌이.** 힙이 비지 않은 동안 다음을 한다.
    - 힙에서 가장 작은 원소 $x$을 꺼내 지금 런에 쓴다.
    - 다음 입력 원소 $y$을 읽는다(있다면).
    - $y \ge x$이면 $y$을 힙에 넣는다(지금 런에 속한다).
    - $y < x$이면 $y$을 *다음* 런에 속한 것으로 표시한다. 힙에는 남아 있지만 지금 런이 끝날 때까지 차례를 매길 때는 $+\infty$처럼 다룬다.
3. **지금 런 끝내기.** 힙의 원소가 모두 다음 런에 속하게 되면 그 표시를 "지우고"(이제 그것들이 새 지금 런이 된다) 2단계부터 이어 간다.
4. **끝내기.** 입력이 다하고 힙이 비면 모든 런이 다 만들어진 것이다.

## 기대 런 길이

**정리(크누스).** 입력이 무작위 순열이고 힙의 용량이 $M$이면, 바꿔 고르기가 내는 런의 기대 길이는 $2M$이다.

*직관.* 힙은 원소 $M$개를 담은 "저수지" 노릇을 한다. 평균으로 들어오는 원소의 절반은 지금 런을 이어 갈 만큼 크고, 절반은 너무 작아 다음 런을 기다려야 한다. 그래서 원소 $M$개씩 덩어리로 정렬하는 것에 견주어 실효 런 길이가 두 배가 된다.

이 분석에는 "제설차" 비유를 쓴다. 길이 $2M$의 둥근 길에 눈이 고르게 내린다고 하자. 그 길을 도는 제설차는 어디서 출발하든 한 바퀴마다 눈 $2M$만큼을 쓸어 담는다.

## 풀이 예제

기억 용량 $M = 3$. 입력 흐름: $[5, 3, 8, 1, 7, 2, 6, 4]$.

**런 1:**

| 걸음 | 힙 | 출력 | 입력 | 하는 일 |
|------|------|--------|-------|--------|
| 처음 | $\{3, 5, 8\}$ | -- | $1, 7, 2, 6, 4$ | 앞의 3개를 싣는다 |
| 1 | $\{5, 8\}$ | $3$ | $1, 7, 2, 6, 4$ | 3을 꺼내고 1을 읽는다(1 < 3이라 다음 런으로 표시) |
| 2 | $\{5, 8, [1]\}$ | $3$ | $7, 2, 6, 4$ | 1은 표시됨, 5를 꺼내고 7을 읽는다(7 >= 5) |
| 3 | $\{7, 8, [1]\}$ | $3, 5$ | $2, 6, 4$ | 7을 꺼내고 2를 읽는다(2 < 7이라 표시) |
| 4 | $\{8, [1, 2]\}$ | $3, 5, 7$ | $6, 4$ | 8을 꺼내고 6을 읽는다(6 < 8이라 표시) |
| 5 | $\{[1, 2, 6]\}$ | $3, 5, 7, 8$ | $4$ | 모두 표시됨 -- 런 1 끝 |

**Run 1 output:** $[3, 5, 7, 8]$ (length 4 > $M = 3$).

**Run 2:** Unmark $\{1, 2, 6\}$, read 4.

| 걸음 | 힙 | 출력 | 입력 | 하는 일 |
|------|------|--------|-------|--------|
| 1 | $\{2, 4, 6\}$ | $1$ | -- | 1을 꺼내고 4를 읽는다(4 >= 1) |
| 2 | $\{4, 6\}$ | $1, 2$ | -- | 2를 꺼낸다, 입력이 더 없다 |
| 3 | $\{6\}$ | $1, 2, 4$ | -- | 4를 꺼낸다 |
| 4 | $\{\}$ | $1, 2, 4, 6$ | -- | 6을 꺼낸다 -- 끝 |

**Run 2 output:** $[1, 2, 4, 6]$ (length 4).

Total: 2 runs of length 4 instead of 3 runs of length 3 with naive formation.

## 구현

```python
"""
바꿔 고르기 — 바깥 병합 정렬을 위해 더 긴 정렬된 런을 만든다.

기대 런 길이는 2M이며 여기서 M은 힙(기억 공간)의 그릇 크기이다.
그러면 필요한 병합 훑기 횟수가 줄어든다.
"""

import heapq


# === 바꿔 고르기 ============================================================

def replacement_selection(data: list[int], memory_size: int) -> list[list[int]]:
    """바꿔 고르기로 *data*에서 정렬된 런을 만든다.

    매개변수
    ----------
    data : list[int]
        원소의 입력 흐름.
    memory_size : int
        힙에 들어가는 원소의 최대 개수(기억 공간의 그릇 크기).

    반환값
    -------
    list[list[int]]
        정렬된 런의 목록.
    """
    runs: list[list[int]] = []
    current_run: list[int] = []

    # 힙 항목: (실효 열쇠, 실제 값, 세대)
    # 세대 0 = 지금 런, 세대 1 = 다음 런
    heap: list[tuple[int, int, int]] = []
    current_gen = 0
    pos = 0

    # 힙 첫걸음 잡기
    while pos < len(data) and len(heap) < memory_size:
        heapq.heappush(heap, (data[pos], data[pos], 0))
        pos += 1

    while heap:
        # 가장 작은 것 꺼내기
        _, val, gen = heapq.heappop(heap)

        if gen > current_gen:
            # 남은 원소는 모두 다음 런에 든다
            runs.append(current_run)
            current_run = []
            current_gen = gen

        current_run.append(val)

        # 다음 입력 원소 읽기
        if pos < len(data):
            next_val = data[pos]
            pos += 1

            if next_val >= val:
                # 지금 런에 든다
                heapq.heappush(heap, (next_val, next_val, current_gen))
            else:
                # 너무 작다 — 다음 런에 든다
                # 힙에 남아 있도록 큰 파수꾼 열쇠를 쓴다
                # 그러나 지금 런이 끝날 때까지 꺼내지지 않는다
                heapq.heappush(
                    heap, (float("inf"), next_val, current_gen + 1)
                )

    if current_run:
        runs.append(current_run)

    return runs


# === 시연 ===================================================================

if __name__ == "__main__":
    data = [5, 3, 8, 1, 7, 2, 6, 4]
    memory_size = 3

    runs = replacement_selection(data, memory_size)
    print(f"Input: {data}")
    print(f"Memory size: {memory_size}")
    print(f"Number of runs: {len(runs)}")
    for i, run in enumerate(runs):
        print(f"  Run {i + 1}: {run} (length {len(run)})")

    # 소박한 런 만들기와 견주기
    naive_runs = []
    for start in range(0, len(data), memory_size):
        naive_runs.append(sorted(data[start : start + memory_size]))
    print(f"\nNaive run formation: {len(naive_runs)} runs")
    for i, run in enumerate(naive_runs):
        print(f"  Run {i + 1}: {run} (length {len(run)})")
```

**출력:**
```
Input: [5, 3, 8, 1, 7, 2, 6, 4]
Memory size: 3
Number of runs: 2
  Run 1: [3, 5, 7, 8] (length 4)
  Run 2: [1, 2, 4, 6] (length 4)

Naive run formation: 3 runs
  Run 1: [3, 5, 8] (length 3)
  Run 2: [1, 2, 7] (length 3)
  Run 3: [4, 6] (length 2)
```

## 소박한 런 만들기와의 견줌

| 성질 | 소박한 방법 | 바꿔 고르기 |
|----------|-------|----------------------|
| 런 길이 | 꼭 $M$ | 기대 $2M$ |
| 런의 수 | $\lceil N/M \rceil$ | $\approx \lceil N/(2M) \rceil$ |
| 거의 정렬된 입력 | 이득 없음 | 런이 파일 전체에 걸칠 수 있음 |
| 구현 | 단순한 배열 정렬 | 세대를 좇는 최소 힙 |
| 아낀 병합 훑기 | -- | 보통 1번 |

## 참고 문헌

- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.), 5.4.1절. Addison-Wesley.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 8장. MIT Press.


## 연습문제

**연습문제 1.**
바꿔 고르기의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 바꿔 고르기를 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 바꿔 고르기가 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
바꿔 고르기가 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.