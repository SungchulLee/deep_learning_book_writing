# 중앙값 유지하기

흘려보내며 하는 분석, 누적 통계, 온라인 알고리즘 같은 여러 응용에서 원소가 하나씩 들어올 때 커 가는 데이터의 **중앙값**을 좇아야 한다. 삽입마다 정렬하면 원소마다 $O(n \log n)$이 들어 큰 흐름에는 너무 느리다. **두 힙 기법**은 데이터를 아래 절반은 최대 힙에, 위 절반은 최소 힙에 나누어 담아 삽입마다 $O(\log n)$, 꺼내기는 $O(1)$으로 중앙값을 지킨다.

## 두 힙 불변식

이 알고리즘은 힙 둘을 지킨다.

- **max_heap**: 원소의 작은 절반을 담는다. 뿌리가 아래 절반에서 가장 큰 원소이다.
- **min_heap**: 원소의 큰 절반을 담는다. 뿌리가 위 절반에서 가장 작은 원소이다.

삽입마다 그 뒤에 다음 불변식을 지킨다.

1. **순서**: max_heap의 모든 원소가 min_heap의 모든 원소 이하이다.
2. **균형**: 크기 차이가 많아야 1이다. 곧 $|\,|\text{max\_heap}| - |\text{min\_heap}|\,| \le 1$이다.

이 불변식 아래에서 중앙값은 언제나 한쪽 또는 양쪽 뿌리에서 얻을 수 있다.

- 두 힙의 크기가 같으면 중앙값은 두 뿌리의 평균이다.
- max_heap에 원소가 하나 더 있으면 중앙값은 max_heap의 뿌리이다.

## 알고리즘

새 원소 $x$마다 다음과 같이 한다.

1. **넣기**: $x$이 max_heap의 뿌리 이하이면(또는 max_heap이 비어 있으면) $x$을 max_heap에 넣는다. 그렇지 않으면 min_heap에 넣는다.
2. **균형 되잡기**: 크기 차이가 1보다 크면 큰 힙에서 꺼내 작은 힙에 넣는다.
3. **묻기**: 중앙값을 뿌리에서 $O(1)$에 얻는다.

## 한 걸음씩 보는 예

흐름 `[5, 2, 8, 1, 7, 3]`을 넣어 보자.

| 단계 | 원소 | max_heap (아래) | min_heap (위) | 중앙값 |
|:----:|:-------:|:-----------------:|:----------------:|:------:|
| 1 | 5 | [5] | [] | 5 |
| 2 | 2 | [2] | [5] | 3.5 |
| 3 | 8 | [2] | [5, 8] | 5 |
| 4 | 1 | [2, 1] | [5, 8] | 3.5 |
| 5 | 7 | [2, 1] | [5, 7, 8] | 5 |
| 6 | 3 | [3, 2, 1] | [5, 7, 8] | 4.0 |

3단계 뒤 min_heap에는 원소가 2개, max_heap에는 1개 있다. 균형 검사는 차이 1을 허락하므로 균형을 되잡을 필요가 없다. min_heap이 하나 더 크게 두고, 중앙값은 그 뿌리(5)이다.

## 복잡도

| 연산 | 시간 |
|-----------|------|
| 원소 하나 넣기 | $O(\log n)$ |
| 중앙값 묻기 | $O(1)$ |
| 원소 $n$개 처리하기 | 모두 $O(n \log n)$ |
| 공간 | $O(n)$ |

삽입마다 힙에 넣기가 많아야 한 번, 균형 되잡기(한 힙에서 꺼내 다른 힙에 넣기)가 많아야 한 번이며 저마다 $O(\log n)$이 든다.

## 구현

```python
"""
힙 둘로 중앙값 유지하기.

아래 절반은 최대 힙으로, 위 절반은 최소 힙으로 지켜
삽입은 O(log n), 중앙값 묻기는 O(1)이 되게 한다.
"""

import heapq


# === 중앙값 찾개 ===

class MedianFinder:
    """두 힙 기법으로 하는 실시간 중앙값 유지.

    max_heap: 작은 절반을 담는다 (heapq 최소 힙을 위해 부호를 뒤집는다).
    min_heap: 큰 절반을 담는다.
    불변식: 크기 차이가 많아야 1이고 max_heap이 더 클 수 있다.
    """

    def __init__(self):
        self.max_heap = []  # 부호를 뒤집은 값 (heapq는 최소 힙만 받쳐 준다)
        self.min_heap = []

    def add(self, x):
        """원소 x를 넣고 불변식을 지킨다. O(log n)."""
        # 어느 힙에 넣을지 정한다
        if not self.max_heap or x <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -x)
        else:
            heapq.heappush(self.min_heap, x)

        # 균형 되잡기: max_heap은 min_heap보다 많아야 1개 더 가질 수 있다
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)

    def median(self):
        """지금의 중앙값을 돌려준다. O(1)."""
        if not self.max_heap:
            raise IndexError("median of empty collection")

        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2

    def __len__(self):
        return len(self.max_heap) + len(self.min_heap)


# === 시연 ===

if __name__ == "__main__":
    mf = MedianFinder()
    stream = [5, 2, 8, 1, 7, 3]

    print("Streaming median maintenance:")
    print(f"{'Element':>8} {'max_heap (lower)':>20} {'min_heap (upper)':>20} {'Median':>8}")
    print("-" * 60)

    for x in stream:
        mf.add(x)
        lower = sorted([-v for v in mf.max_heap])
        upper = sorted(mf.min_heap)
        print(f"{x:>8} {str(lower):>20} {str(upper):>20} {mf.median():>8.1f}")

    # 정렬해서 구한 중앙값과 맞춰 본다
    print("\n--- Verification ---")
    mf2 = MedianFinder()
    data = []
    for x in [41, 35, 62, 5, 97, 108, 3, 25, 22, 78]:
        data.append(x)
        mf2.add(x)
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            expected = sorted_data[n // 2]
        else:
            expected = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        actual = mf2.median()
        status = "OK" if abs(actual - expected) < 1e-9 else "FAIL"
        print(f"  After {x:>3}: median = {actual:>6.1f} (expected {expected:>6.1f}) [{status}]")
```

**출력:**
```
Streaming median maintenance:
 Element   max_heap (lower)    min_heap (upper)   Median
------------------------------------------------------------
       5                [5]                   []      5.0
       2             [2, 5]                   []      3.5
       8                [5]               [2, 8]      5.0
       1             [1, 2]             [5, 8]        3.5
       7             [1, 2]          [5, 7, 8]        5.0
       3          [1, 2, 3]          [5, 7, 8]        4.0

--- Verification ---
  After  41: median =   41.0 (expected   41.0) [OK]
  After  35: median =   38.0 (expected   38.0) [OK]
  After  62: median =   41.0 (expected   41.0) [OK]
  After   5: median =   38.0 (expected   38.0) [OK]
  After  97: median =   41.0 (expected   41.0) [OK]
  After 108: median =   51.5 (expected   51.5) [OK]
  After   3: median =   41.0 (expected   41.0) [OK]
  After  25: median =   38.0 (expected   38.0) [OK]
  After  22: median =   35.0 (expected   35.0) [OK]
  After  78: median =   38.0 (expected   38.0) [OK]
```

## 올바름의 논증

불변식은 max_heap이 가장 작은 $\lceil n/2 \rceil$개를, min_heap이 가장 큰 $\lfloor n/2 \rfloor$개를 담게 한다. max_heap의 모든 원소가 min_heap의 모든 원소 이하이므로 두 뿌리가 데이터를 중앙값에서 가른다.

- $n$이 홀수이면 중앙값은 max_heap의 뿌리(아래 절반에서 가장 큰 것)이다.
- $n$이 짝수이면 중앙값은 두 뿌리의 평균이다.

균형 되잡기 단계는 차이가 1을 넘을 때 힙 사이로 원소를 옮겨 크기 불변식을 지킨다. 옮기는 원소가 제 힙의 끝값이므로 옮길 때마다 순서 불변식이 지켜진다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Problem 9-1: Sorting and order statistics. MIT Press.


## 연습문제

**연습문제 1.**
중앙값 유지하기의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 중앙값 유지하기를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
중앙값 유지하기의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.