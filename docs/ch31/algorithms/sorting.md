# 바깥 기억 줄 세우기

원소 $N$개의 자료 뭉치가 크기 $M$의 으뜸 기억에 들어가지 않으면 빠른 정렬이나 합침 정렬 같은 기억 속 알고리즘을 그대로 쓸 수 없다. 아무 닿기마다 따로 원반 들고남이 일어나 들고남 $O(N \log N)$번이 들며 이는 필요 이상으로 나쁘다. **바깥 기억 합침 정렬**은 원반 닿기를 차례 훑기와 여러 갈래 합침으로 짜서 가장 좋은 들고남 복잡도 $O((N/B) \log_{M/B}(N/B))$을 이룬다.

---

## 1. 알고리즘 훑어보기

바깥 병합 정렬은 두 단계로 나아간다.

1. **줄기 만들기:** 들임을 저마다 기억에 들어가는 줄 세운 토막(줄기)으로 가른다.
2. **여러 갈래 합침:** 줄 세운 줄기가 하나 남을 때까지 줄기 무리를 되풀이해 합친다.

---

## 2. 1마당: 줄기 만들기

원소 $M$개를 기억에 읽어 들여 아무 기억 속 알고리즘(보기로 빠른 정렬)으로 줄 세운 뒤 그 줄기를 원반에 다시 적는다. 원소 $N$개를 모두 다룰 때까지 다음 $M$개에 대해 되풀이한다.

이는 저마다 길이가 많아야 $M$인 줄 세운 줄기 $\lceil N/M \rceil$개를 낸다. 들고남 비용은 다음과 같다:

$$
2 \cdot \left\lceil \frac{N}{B} \right\rceil
$$

이 마당에서 원소마다 한 번 읽고 한 번 적기 때문이다.

!!! tip "바꿔 고르기"

    바꿔 고르기 재주는 최소 원소를 내놓고 새 원소가 지금 줄기를 이어 갈 수 있는 한 그것을 다음 들임 원소로 바꾸는 앞섬 줄을 써서 기대 길이 $2M$($M$ 대신)의 줄기를 낼 수 있다. 이는 첫 줄기 수를 대략 반으로 줄인다.

---

## 3. 2마당: 여러 갈래 합침

기억이 $M$바이트이고 덩이 크기가 $B$이면 덩이 $M/B$개를 한꺼번에 버퍼에 둘 수 있다. 내놓기용으로 덩이 하나를 남기면 다음까지 합칠 수 있다:

$$
f = \frac{M}{B} - 1 \approx \frac{M}{B}
$$

한 지나기에 줄 세운 줄기를. 합침 지나기마다 덩이 $N/B$개를 모두 읽고 적어 들고남 $2 \lceil N/B \rceil$번이 든다.

### 지나기 횟수

줄기 $\lceil N/M \rceil$개에서 시작해 한 번에 $f$개씩 합치면 합침 지나기 횟수는 다음과 같다:

$$
p = \left\lceil \log_f \frac{N}{M} \right\rceil = \left\lceil \log_{M/B} \frac{N}{B} \right\rceil
$$

$N/M = (N/B)/(M/B)$이고 $f \approx M/B$이기 때문이다.

### 온 들고남 복잡도

지나기 $p$번마다 들고남 $O(N/B)$번이 든다. 첫 줄기 만들기 지나기까지 넣으면:

$$
\text{Total I/O} = O\!\left(\frac{N}{B} \cdot \log_{M/B} \frac{N}{B}\right) = O(\text{sort}(N))
$$

이 가둠은 바깥 기억 모형의 견줌 바탕 줄 세우기에서 **가장 좋은** 값이다.

---

## 4. 실제 지나기 횟수

흔한 잡에서는 합침 지나기 횟수가 놀랄 만큼 적다:

| $N$ | $M$ | $B$ | $M/B$(퍼짐) | 지나기 |
|---|---|---|---|---|
| $10^8$ | $10^6$ | $4096$ | 244 | 2 |
| $10^{10}$ | $10^6$ | $4096$ | 244 | 3 |
| $10^{12}$ | $10^6$ | $4096$ | 244 | 4 |
| $10^{12}$ | $10^8$ | $4096$ | 24414 | 2 |

원소가 1조 개라도 퍼짐 $M/B$이 수백이면 바깥 기억 합침 정렬은 지나기 3~4번이면 된다.

---

## 5. 보기: 바깥 기억 합침 정렬 흉내내기

```python
"""
바깥 기억 합침 정렬 흉내내기.

두 마당 길을 보여 준다. 줄기 만들기 뒤에 여러 갈래 합침을 하며
단계마다 들고남을 좇는다.
"""

import math
import heapq

# ===================================================================
# 바깥 기억 합침 정렬(흉내)
# ===================================================================

def external_merge_sort(data: list[int], memory_size: int,
                        block_size: int) -> tuple[list[int], dict]:
    """
    정수 목록에 바깥 기억 합침 정렬을 흉내 낸다.

    매개변수
    ----------
    data : 들임 자료(원반에 놓인 자료를 흉내 냄).
    memory_size : 기억에 들어가는 원소 개수(M).
    block_size : 덩이 옮김마다 원소 개수(B).

    반환값
    -------
    (줄 세운 자료, 들고남 통계 사전)의 짝.
    """
    n = len(data)
    io_count = 0

    # 1마당: 줄기 만들기
    runs = []
    for start in range(0, n, memory_size):
        end = min(start + memory_size, n)
        chunk = data[start:end]
        io_count += math.ceil(len(chunk) / block_size)  # 읽기
        chunk.sort()  # 기억 속 줄 세우기(공짜)
        runs.append(chunk)
        io_count += math.ceil(len(chunk) / block_size)  # 적기

    phase1_ios = io_count
    num_initial_runs = len(runs)

    # 2마당: 여러 갈래 합침
    fan_out = max(2, memory_size // block_size - 1)
    merge_pass = 0

    while len(runs) > 1:
        merge_pass += 1
        new_runs = []
        for i in range(0, len(runs), fan_out):
            group = runs[i:i + fan_out]

            # 최소 더미로 이 무리를 합친다
            merged = []
            heap = []
            for run_idx, run in enumerate(group):
                if run:
                    heapq.heappush(heap, (run[0], run_idx, 0))

            while heap:
                val, run_idx, pos = heapq.heappop(heap)
                merged.append(val)
                if pos + 1 < len(group[run_idx]):
                    heapq.heappush(
                        heap,
                        (group[run_idx][pos + 1], run_idx, pos + 1)
                    )

            # 들고남 세기: 들임 덩이 모두 읽기 + 내놓기 덩이 모두 적기
            total_elements = sum(len(r) for r in group)
            io_count += 2 * math.ceil(total_elements / block_size)
            new_runs.append(merged)

        runs = new_runs

    stats = {
        "n": n,
        "memory_size": memory_size,
        "block_size": block_size,
        "fan_out": fan_out,
        "initial_runs": num_initial_runs,
        "merge_passes": merge_pass,
        "phase1_ios": phase1_ios,
        "total_ios": io_count,
    }

    return runs[0] if runs else [], stats

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    import random

    random.seed(42)
    N = 10000
    M = 1000
    B = 100

    data = [random.randint(0, 10**6) for _ in range(N)]
    sorted_data, stats = external_merge_sort(data, M, B)

    # 맞는지 확인하기
    assert sorted_data == sorted(data), "Sort failed!"

    print(f"External Merge Sort Simulation")
    print(f"  N = {stats['n']:,}")
    print(f"  M = {stats['memory_size']:,}")
    print(f"  B = {stats['block_size']}")
    print(f"  Fan-out (M/B - 1): {stats['fan_out']}")
    print(f"  Initial runs:      {stats['initial_runs']}")
    print(f"  Merge passes:      {stats['merge_passes']}")
    print(f"  Phase 1 I/Os:      {stats['phase1_ios']}")
    print(f"  Total I/Os:        {stats['total_ios']}")

    # 이론 가둠
    blocks = math.ceil(N / B)
    theoretical = blocks * math.ceil(
        math.log(blocks) / math.log(max(2, M // B))
    )
    print(f"  Theoretical O((N/B)log_{{M/B}}(N/B)): ~{theoretical}")
```

??? example "보기 내놓기"

    ```
    바깥 기억 합침 정렬 흉내내기
      N = 10,000
      M = 1,000
      B = 100
      퍼짐(M/B - 1): 9
      첫 줄기:      10
      합침 지나기:       2
      1마당 들고남:      200
      온 들고남:        600
      Theoretical O((N/B)log_{M/B}(N/B)): ~200
    ```

    흉내내기는 퍼짐이 9일 때 바깥 기억 합침 정렬에 합침 지나기가 2번이면 되며 온 들고남이 차례 훑기 몇 번에 비례함을 확인해 준다.

---

## 6. 아래 한계

가둠 $\Theta((N/B) \log_{M/B}(N/B))$은 이룰 수 있을 뿐 아니라 견줌 바탕 바깥 줄 세우기의 **아래 가둠**이다. 밝힘은 들고남마다 원소 $B$개를 읽을 수 있고($B!$가지 차례를 주며) 기억이 원소 $M$개를 담을 수 있음($M!$가지 다시 늘어놓기를 허락함)을 셈에 넣어 RAM 모형의 결정 나무 아래 가둠을 넓힌다.

---

## 연습문제

**연습문제 1.**
바깥 기억 합침 정렬 알고리즘을 밝히고 그 들고남 복잡도를 살펴라.

??? success "연습문제 1 풀이"
    바깥 기억 합침 정렬: (1) 원소 $M$개를 기억에 읽어 속에서 줄 세운 뒤 줄 세운 '줄기'로 원반에 적는다. 줄기 $\lceil N/M \rceil$개를 만든다. (2) 줄기를 합친다. 줄기 $M/B - 1$개를 한꺼번에 열어(저마다 덩이 하나씩) $(M/B-1)$갈래 합침으로 내놓기에 합친다. 줄기가 하나 남을 때까지 되풀이한다. 들고남 복잡도: $O(N/B \cdot \log_{M/B}(N/B))$. 이는 가장 좋은 값이다(바깥 기억 모형의 견줌 바탕 아래 가둠과 맞는다).

---

**연습문제 2.**
바깥 기억 줄 세우기의 들고남 아래 가둠을 밝혀라.

??? success "연습문제 2 풀이"
    견줌 바탕 줄 세우기 알고리즘은 $N!$가지 차례를 가려내야 한다. 들고남마다 원소 $B$개를 읽어 그 덩이의 새 차례를 많아야 $B!$가지 만든다. 기억에 덩이 $M/B$개가 있으면 걸음마다 가려낼 수 있는 상태가 많아야 $(B!)^{M/B}$가지다. 들고남 $T$번 뒤: $(B!)^{TM/B} \geq N!$. 로그를 잡으면 $T \cdot M/B \cdot \log(B!) \geq \log(N!)$. 스털링을 쓰면 $T \geq \Omega(N/B \cdot \log_{M/B}(N/B))$.

---

**연습문제 3.**
SSD과 HDD 저장 장치에서 바깥 기억 줄 세우기 성능을 견주어라.

??? success "연습문제 3 풀이"
    HDD: 차례 읽기 $\sim 200$MB/s, 아무 읽기 $\sim 1$MB/s(찾아가는 때 때문). 바깥 기억 줄 세우기의 차례 닿기 무늬는 거의 최고 처리량을 낸다. SSD: 차례 $\sim 3$GB/s, 아무 $\sim 500$MB/s. SSD은 (차례와 아무의 틈이 작아) 바깥 줄 세우기에서 얻는 이득이 덜하지만 여전히 크다. 램 8GB의 HDD에서 100GB 줄 세우기: $\sim 500$초(대부분 들고남). NVMe SSD에서는 $\sim 35$초. SSD에서는 $M/B$ 퍼짐이 커져(실제 덩이 크기가 작아져) 합침 지나기 횟수가 준다.

---

**연습문제 4.**
바깥 기억 줄 세우기는 깊은 배움 자료 흐름 채비에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    익히기 자료 채비: (1) 큰 자료 뭉치 섞기: 아무 열쇠로 바깥에서 줄 세워 아무 자리바꿈을 얻는다, (2) 겹침 없애기: 흩기값으로 줄 세운 뒤 잇단 겹침을 훑는다, (3) 효율 좋은 찾기를 위한 줄 세운 색인 만들기(보기로 가장 가까운 이웃 찾기), (4) 익히기 표본을 차례 배우기의 어려움으로 줄 세우기. 보기가 수십억 개인 자료 뭉치(Common Crawl, LAION)에서는 원반 위의 바깥 기억 줄 세우기만이 온 자리 섞기의 유일한 길이다.

## 정리하며

이 마당은 알고리즘 훑어보기、1마당: 줄기 만들기、2마당: 여러 갈래 합침、실제 지나기 횟수을 차례로 짚었다.

**참고 문헌**

- Aggarwal, A. & Vitter, J. S. "The Input/Output Complexity of Sorting and Related Problems," *Communications of the ACM*, 31(9), 1988.
- Knuth, D. *The Art of Computer Programming*, Vol. 3: Sorting and Searching, 1998.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
