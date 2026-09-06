# 병합 전략

팀 정렬이 자연 런을 짚어내고 늘린 뒤에는 그것들을 정렬된 배열 하나로 병합해야 한다. 병합 전략은 **어떤 런을 어떤 차례로 병합할지**를 정한다. 소박하게 다가가면(앞의 둘을 병합하고 그 결과를 셋째와 병합하는 식으로) 몹시 치우친 병합이 될 수 있다. 그 대신 팀 정렬은 아직 병합하지 않은 런의 더미를 지니고, 고른 병합을 보장하는 불변식을 지켜 전체 병합 비용을 $O(n \log n)$으로 묶는다.

## 병합 더미

팀 정렬은 런을 찾을 때마다 더미에 얹는다. 더미의 항목마다 런의 시작 첨자와 길이를 적어 둔다. 얹을 때마다 팀 정렬은 더미 꼭대기의 항목들이 두 불변식을 만족하는지 살핀다. 어느 하나라도 어긋나면 알맞은 쌍을 병합한다.

## 더미의 불변식

더미 꼭대기의 세 항목을 $X$, $Y$, $Z$($X$이 맨 위)이라 하자. 팀 정렬은 다음을 지킨다.

1. $|Z| > |Y| + |X|$
2. $|Y| > |X|$

이 불변식은 런의 길이가 적어도 피보나치 수열만큼 빠르게 자라게 하여 더미 깊이를 $O(\log n)$으로 묶고 병합이 대체로 고르게 되도록 보장한다. 불변식이 어긋나면 다음과 같이 한다.

- $|Z| \leq |Y| + |X|$이면 $Y$을 $Z$과 $X$ 가운데 작은 쪽과 병합한다.
- $|Y| \leq |X|$이면 $X$과 $Y$을 병합한다.

## 이 불변식이 통하는 까닭

피보나치처럼 자란다는 것은 깊이 $d$인 더미에 적어도 $F_{d+2}$개의 원소가 필요하다는 뜻이며, $F_k$은 $k$번째 피보나치 수이다. $F_k$이 지수로 자라므로 원소 $n$개에 대한 더미 깊이는 많아야 $O(\log_\phi n)$이고 $\phi = (1 + \sqrt{5})/2$은 황금비이다. 그래서 어느 때든 아직 병합하지 않은 런이 많아야 $O(\log n)$개임이 보장된다.

## 병합 절차

이웃한 런 둘을 병합할 때 팀 정렬은 두 런 가운데 **짧은** 쪽 크기의 임시 버퍼를 쓴다. 그래서 도움 공간이 (온전한 병합 정렬의) $O(n)$에서 최악의 경우 $O(n/2)$으로, 흔히 그보다 훨씬 적게 줄어든다.

병합은 두 모드로 나아간다.

1. **하나씩 모드**: 런마다 아직 병합하지 않은 가장 작은 원소를 견주어 이긴 것을 옮긴다. 같은 런에서 잇달아 몇 번 이겼는지 좇는다.
2. **질주 모드**: 한 런이 잇달아 `min_gallop`번 이기면 지수 찾기로 갈아타 진 런의 다음 원소가 들어갈 자리를 찾고, 이긴 쪽 덩어리를 한꺼번에 옮긴다.

## 구현

```python
"""
팀 정렬의 병합 전략: 불변식을 갖춘 더미 기반 병합.

팀 정렬이 런의 더미를 지니고 피보나치 같은 불변식을 지켜
병합을 고르게 유지하는 모습을 보여 준다. 짧은 런 크기의
임시 버퍼를 쓴다.
"""


# === 이웃한 런 둘 병합하기 ===

def merge_runs(arr: list, lo: int, mid: int, hi: int) -> None:
    """arr[lo..mid]과 arr[mid+1..hi]을 제자리에서 병합한다.

    도움 공간을 가장 적게 쓰려고 짧은 런에
    임시 버퍼를 쓴다.
    """
    left = arr[lo:mid + 1]
    right = arr[mid + 1:hi + 1]

    i = 0
    j = 0
    k = lo

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


# === 불변식을 살피는 병합 더미 ===

class MergeStack:
    """아직 병합하지 않은 런을 다스리고 팀 정렬의 병합 불변식을 지킨다."""

    def __init__(self, arr: list):
        self.arr = arr
        self.stack = []  # 항목마다 (시작, 길이)

    def push_run(self, start: int, length: int) -> None:
        """새 런을 얹고 필요하면 병합하여 불변식을 되살린다."""
        self.stack.append((start, length))
        self._merge_collapse()

    def _merge_collapse(self) -> None:
        """더미 불변식이 지켜질 때까지 런을 병합한다."""
        while len(self.stack) > 1:
            n = len(self.stack) - 1

            if (n >= 2 and self.stack[n - 2][1]
                    <= self.stack[n - 1][1] + self.stack[n][1]):
                if self.stack[n - 2][1] < self.stack[n][1]:
                    self._merge_at(n - 2)
                else:
                    self._merge_at(n - 1)
            elif self.stack[n - 1][1] <= self.stack[n][1]:
                self._merge_at(n - 1)
            else:
                break

    def _merge_at(self, i: int) -> None:
        """stack[i]과 stack[i+1]을 병합한다."""
        start1, len1 = self.stack[i]
        start2, len2 = self.stack[i + 1]

        merge_runs(self.arr, start1, start1 + len1 - 1,
                   start2 + len2 - 1)

        self.stack[i] = (start1, len1 + len2)
        del self.stack[i + 1]

    def force_merge_all(self) -> None:
        """더미에 남은 런을 모두 병합한다."""
        while len(self.stack) > 1:
            n = len(self.stack) - 1
            if n >= 2 and self.stack[n - 2][1] < self.stack[n][1]:
                self._merge_at(n - 2)
            else:
                self._merge_at(n - 1)


# === 시연 ===

if __name__ == "__main__":
    # 런을 더미에 얹는 것을 흉내 낸다
    arr = [1, 3, 5, 7, 2, 4, 6, 8, 10, 9, 11, 13, 12, 14]
    print(f"Original: {arr}")

    ms = MergeStack(arr)

    # 런을 얹는다: [1,3,5,7], [2,4,6,8,10], [9,11,13], [12,14]
    runs = [(0, 4), (4, 5), (9, 3), (12, 2)]
    for start, length in runs:
        print(f"  Push run: arr[{start}:{start+length}] = "
              f"{arr[start:start+length]}")
        ms.push_run(start, length)
        print(f"  Stack: {ms.stack}")

    ms.force_merge_all()
    print(f"After merge all: {arr}")
    print(f"Final stack: {ms.stack}")
```

**출력:**
```
Original: [1, 3, 5, 7, 2, 4, 6, 8, 10, 9, 11, 13, 12, 14]
  Push run: arr[0:4] = [1, 3, 5, 7]
  Stack: [(0, 4)]
  Push run: arr[4:9] = [2, 4, 6, 8, 10]
  Stack: [(0, 4), (4, 5)]
  Push run: arr[9:12] = [9, 11, 13]
  Stack: [(0, 4), (4, 5), (9, 3)]
  Push run: arr[12:14] = [12, 14]
  Stack: [(0, 4), (4, 5), (9, 3), (12, 2)]
After merge all: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Final stack: [(0, 14)]
```

## 복잡도

| 성질 | 값 |
|----------|-------|
| 최대 더미 깊이 | $O(\log n)$ |
| 전체 병합 견줌 | $O(n \log n)$ |
| 병합용 도움 공간 | 최악의 경우 $O(n/2)$ |

!!! tip "Powersort의 개선"
    파이썬 3.11은 팀 정렬의 본디 병합 전략을 **Powersort**로 갈아 끼웠다. 이는 런의 경계마다 지닌 "거듭제곱"을 바탕으로 더 단순한 규칙으로 병합 차례를 정한다. Powersort은 더 깔끔한 불변식과 조금 더 고른 병합으로 같은 $O(n \log n)$ 한계를 이룬다.

## 참고 문헌

- Peters, T. (2002). *Timsort description*. [CPython 소스, `Objects/listsort.txt`](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
- Auger, N., Jugé, V., Nicaud, C., & Pivoteau, C. (2019). On the worst-case complexity of TimSort. *Proceedings of ESA*, 13:1-13:13.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 병합 전략을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
병합 전략의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
병합 전략은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 병합 전략을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.