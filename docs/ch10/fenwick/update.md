# 점 갱신

바탕 배열의 원소가 바뀌면 이진 색인 트리는 그 원소를 담은 범위의 모든 트리 노드를 고쳐야 한다. **점 갱신** 연산은 자리 $i$에 값 $\delta$을 더하며 그 변화를 BIT 위로 $O(\log n)$ 시간에 퍼뜨린다. 이 쪽은 갱신 순회가 어떻게 도는지와 왜 올바른지를 자세히 다룬다.

---

## 1. 갱신 알고리즘

자리 $i$에 $\delta$을 더하려고 알고리즘은 가장 낮은 켜진 비트를 되풀이해 더하며 숨은 BIT를 거슬러 올라간다.

1. `tree[i]`에 $\delta$을 더한다.
2. 다음 조상으로 옮긴다: $i \leftarrow i + \text{lowbit}(i)$. 여기서 $\text{lowbit}(i) = i \;\&\; (-i)$이다.
3. $i > n$이 될 때까지 되풀이한다.

핵심 통찰은 가장 낮은 켜진 비트를 **더하면** 지금 범위를 엄밀히 품는 가장 가까운 트리 노드로 옮겨 간다는 것이다. 가장 낮은 켜진 비트를 **빼서** 겹치지 않는 이웃 덩어리로 옮기는 접두사 질의의 거울상이다.

---

## 2. 가장 낮은 켜진 비트를 더하면 부모로 가는 까닭

색인 $i$을 이진으로 생각하자. $i$의 BIT 노드는 원소 $\text{lowbit}(i)$개의 범위를 덮는다. $i + \text{lowbit}(i)$을 셈하면 가장 낮은 켜진 비트를 한 자리 위로 "올림"한다. 그렇게 나온 색인은 $i$의 범위를 담는 엄밀히 더 큰 범위를 덮는다.

!!! example "i = 3의 갱신 순회"
    | 단계 | $i$ (이진) | $\text{lowbit}(i)$ | 다음 $i$ |
    |:----:|:---:|:---:|:---:|
    | 1 | 3 = `011` | 1 | 3 + 1 = 4 |
    | 2 | 4 = `100` | 4 | 4 + 4 = 8 |
    | 3 | 8 = `1000` | 8 | 8 + 8 = 16 > $n$ |

    그러므로 크기가 8인 BIT에서 자리 3을 고치면 노드 3, 4, 8을 건드린다. 노드 3은 $[3,3]$을, 노드 4는 $[1,4]$을, 노드 8은 $[1,8]$을 덮는데 모두 자리 3을 담는 범위이다.

---

## 3. 한 걸음씩 따라가기

(모두 0인) 배열 $a = [0, 0, 0, 0, 0, 0, 0, 0]$에서 시작해 자리 3에 5를 더해 보자.

**갱신 전:** 모든 `tree[i] = 0`.

**Update(3, 5):**

| 단계 | $i$ | 앞의 `tree[i]` | 뒤의 `tree[i]` |
|:----:|:---:|:---:|:---:|
| 1 | 3 | 0 | 5 |
| 2 | 4 | 0 | 5 |
| 3 | 8 | 0 | 5 |

갱신한 뒤 $i \geq 3$인 접두사 질의는 값 5를 담고 $i < 3$인 질의는 담지 않는다. 자리 3에 5를 더한 것이 꼭 그대로 나타난다.

---

## 4. 올바름의 논증

자리 $i$을 범위에 담는 노드가 모두 고쳐질 때만 갱신이 올바르다. 그런 노드의 집합이 바로 숨은 BIT에서 $i$의 **조상 사슬**이다. 되풀이 $i \leftarrow i + \text{lowbit}(i)$이 모든 조상을 들르는 까닭은 다음과 같다.

1. 조상마다 자리 $i$을 담는 엄밀히 더 큰 범위를 덮는다.
2. 단계마다 가장 낮은 켜진 비트가 커진다(비트 자리가 적어도 하나 올라간다).
3. 끝내 $i$이 $n$을 넘어 고리가 많아야 $\lfloor \log_2 n \rfloor$단계 뒤에 끝난다.

---

## 5. 구현

```python
"""
이진 색인 트리의 점 갱신.

잎 자리의 변화를 모든 조상 노드로 퍼뜨리는 갱신 순회를
어느 노드를 들르는지 자세히 따라가며
보인다.
"""

# === 갱신을 따라가는 펜윅 트리 ===

class FenwickTree:
    """가르치려고 갱신을 따라가는 방식을 선택으로 갖춘 BIT."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """자리 i에 delta를 더하며 조상으로 퍼뜨린다."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def update_traced(self, i: int, delta: int) -> None:
        """조상 순회를 찍어 보이며 갱신한다."""
        step = 1
        print(f"  Update position {i} by {delta}:")
        while i <= self.n:
            self.tree[i] += delta
            print(f"    Step {step}: tree[{i}] updated "
                  f"(lowbit={i & (-i)}, next i={i + (i & (-i))})")
            i += i & (-i)
            step += 1

    def query(self, i: int) -> int:
        """1부터 i까지의 접두사 합을 돌려준다."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

# === 시연 ===

if __name__ == "__main__":
    n = 8
    ft = FenwickTree(n)

    # 따라가는 갱신
    ft.update_traced(3, 5)
    print()
    ft.update_traced(5, 7)
    print()

    # 접두사 질의로 확인한다
    print("Prefix sums after updates:")
    for i in range(1, n + 1):
        print(f"  prefix({i}) = {ft.query(i)}")

    # 고치고 변화를 살핀다
    print()
    ft.update_traced(3, 10)
    print()
    print("Prefix sums after adding 10 more to position 3:")
    for i in range(1, n + 1):
        print(f"  prefix({i}) = {ft.query(i)}")
```

**출력:**
```
  Update position 3 by 5:
    Step 1: tree[3] updated (lowbit=1, next i=4)
    Step 2: tree[4] updated (lowbit=4, next i=8)
    Step 3: tree[8] updated (lowbit=8, next i=16)

  Update position 5 by 7:
    Step 1: tree[5] updated (lowbit=1, next i=6)
    Step 2: tree[6] updated (lowbit=2, next i=8)
    Step 3: tree[8] updated (lowbit=8, next i=16)

Prefix sums after updates:
  prefix(1) = 0
  prefix(2) = 0
  prefix(3) = 5
  prefix(4) = 5
  prefix(5) = 12
  prefix(6) = 12
  prefix(7) = 12
  prefix(8) = 12

  Update position 3 by 10:
    Step 1: tree[3] updated (lowbit=1, next i=4)
    Step 2: tree[4] updated (lowbit=4, next i=8)
    Step 3: tree[8] updated (lowbit=8, next i=16)

Prefix sums after adding 10 more to position 3:
  prefix(1) = 0
  prefix(2) = 0
  prefix(3) = 15
  prefix(4) = 15
  prefix(5) = 22
  prefix(6) = 22
  prefix(7) = 22
  prefix(8) = 22
```

---

## 6. 복잡도

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 점 갱신 | $O(\log n)$ | $O(1)$ |

갱신은 노드를 많아야 $\lfloor \log_2 n \rfloor$개 들르고 노드마다 (덧셈 한 번과 비트 연산 한 번인) $O(1)$의 일이 든다.

!!! tip "더하기와 덮어쓰기"
    BIT의 갱신 연산은 자리에 **차이값**을 더한다. 자리 $i$을 새 값 $v$으로 **덮어쓰려면** 먼저 지금 값을 $a[i] = \text{prefix}(i) - \text{prefix}(i-1)$으로 셈한 뒤 `update(i, v - a[i])`을 부른다. 아니면 $O(1)$에 찾도록 지금 값을 담는 배열을 따로 지킨다.

---

## 연습문제

**연습문제 1.**
점 갱신의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 점 갱신을 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
점 갱신가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.

## 정리하며

이 마당은 갱신 알고리즘、가장 낮은 켜진 비트를 더하면 부모로 가는 까닭、한 걸음씩 따라가기、올바름의 논증을 차례로 짚었다.

**참고 문헌**

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
