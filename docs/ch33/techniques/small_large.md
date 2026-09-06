# 작은 것을 큰 것에 합치기

뿌리 있는 나무를 아래에서 위로 다룰 때 꼭짓점마다 그 밑나무에서 모은 값의 모임을 지니는 일이 흔하다. 아이의 모임을 모두 어버이로 옮기는 어설픈 길은 모두 $O(n^2)$이 들 수 있다. **작은 것을 큰 것에 합치기**(*나무 위 DSU*나 *무거움-가벼움 합치기*라고도 한다)는 늘 작은 모임을 큰 모임에 합쳐 모두 $O(n \log n)$을 이룬다. 이 단순한 규칙이 원소마다 많아야 $O(\log n)$번 옮겨지도록 보장한다.

## 핵심 통찰

크기 $a$와 $b$($a \le b$)인 두 모임을 합친다고 하자. 작은 모임의 원소를 모두 큰 모임으로 옮기면 $O(a)$가 든다. 합친 뒤 결과의 크기는 $a + b \ge 2a$이므로 옮겨진 원소가 든 모임은 적어도 두 배가 되었다. 모임 크기의 최댓값이 $n$이므로 원소마다 그 모임이 크기 $n$에 이르기 전에 많아야 $\log_2 n$번 옮겨질 수 있다.

**정리.** 마디 $n$개의 나무에서 작은 것을 큰 것에 합치기는 원소를 모두 많아야 $O(n \log n)$번 옮긴다.

??? note "증명"
    원소마다 0으로 시작하는 셈틀을 매긴다. 원소가 옮겨질 때마다(작은 모임에서 큰 모임으로) 그 셈틀을 올린다. 옮긴 뒤 그 원소의 모임은 적어도 두 배가 되었다. 모임 크기의 최댓값이 $n$이므로 원소마다 셈틀은 많아야 $\lfloor \log_2 n \rfloor$이다. 원소 $n$개에 걸쳐 더하면 옮김은 모두 많아야 $n \lfloor \log_2 n \rfloor = O(n \log n)$번이다. $\square$

## 알고리즘

마디 $v$마다 어떤 값으로 시작하는 모임 $S(v)$을 지닌 뿌리 있는 나무에서:

1. 마디를 뒤 차례(아이를 어버이보다 먼저)로 다룬다.
2. 마디 $v$마다 모든 아이의 모임을 $v$의 모임에 합친다.
3. 두 모임을 합칠 때는 늘 작은 모임을 훑어 그 원소를 큰 모임에 넣는다.
4. 합친 뒤 $v$에 딸린 물음에 답한다.

## 구현

```python
"""
작은 것을 큰 것에 합치기(나무 위 서로소 집합 합치기).

늘 작은 것을 큰 것에 합쳐 아이 모임을 어버이 모임에 합치며
moving elements from the smaller set to the larger one,
모두 O(n log n)번의 셈을 이룸을 보인다.
"""

from collections import defaultdict

# ===================================================================
# 나무에서 작은 것을 큰 것에 합치기
# ===================================================================

def small_to_large(adj, colors, root=0):
    """작은 것을 큰 것에 합치기로 밑나무마다 서로 다른 빛깔 수를 센다.

    인수:
        adj: adjacency list (list of lists)
        colors: color[v] is the color of vertex v
        root: 나무의 뿌리

    반환값:
        distinct: distinct[v] = number of distinct colors in subtree of v
        total_moves: 이뤄진 전체 원소 옮김 수
    """
    n = len(adj)
    distinct = [0] * n
    sets = [set() for _ in range(n)]
    parent = [-1] * n
    order = []
    total_moves = 0

    # 뒤 차례 훑기 셈
    stack = [(root, False)]
    visited = [False] * n
    visited[root] = True
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        stack.append((node, True))
        for child in adj[node]:
            if not visited[child]:
                visited[child] = True
                parent[child] = node
                stack.append((child, False))

    # 뒤 차례로 다룸
    for v in order:
        sets[v].add(colors[v])
        # 아이의 모임을 v의 모임에 합침
        for child in adj[v]:
            if child == parent[v]:
                continue
            # 늘 작은 것을 큰 것에 합침
            if len(sets[child]) > len(sets[v]):
                sets[v], sets[child] = sets[child], sets[v]
            total_moves += len(sets[child])
            sets[v].update(sets[child])
            sets[child] = None  # 빈 기억

        distinct[v] = len(sets[v])

    return distinct, total_moves

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    #       0 (빨강)
    #      / \
    #     1   2 (파랑)
    #    / \   \
    #   3   4   5 (빨강)
    #  (초록) (파랑)
    #  /
    # 6 (빨강)
    n = 7
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    colors = ["red", "red", "blue", "green", "blue", "red", "red"]

    distinct, total_moves = small_to_large(adj, colors, root=0)

    print("Distinct colors per subtree:")
    for v in range(n):
        print(f"  node {v} (color={colors[v]}): "
              f"{distinct[v]} distinct")
    print(f"\nTotal element moves: {total_moves}")
    print(f"Upper bound O(n log n): {n} * {n.bit_length()-1} = "
          f"{n * (n.bit_length()-1)}")
```

**출력:**
```
Distinct colors per subtree:
  node 0 (color=red): 3 distinct
  node 1 (color=red): 3 distinct
  node 2 (color=blue): 2 distinct
  node 3 (color=green): 2 distinct
  node 4 (color=blue): 1 distinct
  node 5 (color=red): 1 distinct
  node 6 (color=red): 1 distinct

전체 원소 옮김: 4
Upper bound O(n log n): 7 * 2 = 14
```

## 복잡도

| 마디 | 시간 | 공간 |
|---|---|---|
| 나무 훑기 | $O(n)$ | $O(n)$ |
| 전체 합치기 값 | $O(n \log n)$ | -- |
| 마디마다의 물음 | $O(1)$ | -- |
| **모두** | $O(n \log n)$ | $O(n)$ |

## 응용

- **밑나무마다 서로 다른 값**: 위의 보기처럼 밑나무마다 서로 다른 값(빛깔, 이름표)의 수를 센다.
- **밑나무 잦기 물음**: 밑나무마다 가장 잦은 원소를 찾는다.
- **오일러 돌기를 거친 길 물음**: 오일러 돌기 줄이기와 엮으면 작은 것을 큰 것에 합치기가 어떤 길 물음 문제를 잘 다룬다.
- **나무 위 합치기-찾기**: 나무 위 DSU 변형은 온 자리 자료 얼개를 지니고 작은 것을 큰 것에 합치는 원리로 밑나무를 드나들며 밑나무 물음을 다룬다.

!!! tip "나무 위 DSU 변형"
    이따금 "나무 위 DSU"라 하는 다른 표현은 온 자리 자료 얼개 하나만 지닌다. 마디마다 가벼운 아이(작은 밑나무)를 먼저 다루고 그 이바지를 되돌린 뒤, 무거운 아이(가장 큰 밑나무)를 마지막에 되돌리지 않고 다룬다. 이는 드러난 모임 합치기를 피하면서 같은 $O(n \log n)$ 가둠을 이룬다.

## 참고 문헌

- Competitive Programmer's Handbook (Laaksonen), "Small to large" 마디.
- Sack, J.-R. and Strothmann, T. (1989). "A characterization of heaps and its applications." *Information and Computation*.

## 연습문제

**연습문제 1.**
마디가 7개인 나무가 있고 마디마다 빛깔이 있다. 밑나무마다 서로 다른 빛깔의 수를 알려야 한다. 뚜렷한 보기 나무에서 작은 것을 큰 것에 합치기 알고리즘을 짚어 가며 전체 원소 옮김 수를 세어라.

??? success "연습문제 1 풀이"
    이런 나무를 보자. 뿌리 1의 아이가 2, 3이다. 마디 2의 아이는 4, 5다. 마디 3의 아이는 6, 7이다. 빛깔: 1=빨강, 2=파랑, 3=빨강, 4=초록, 5=파랑, 6=빨강, 7=노랑. 아래에서 위로 다룬다. 잎은 홑 모임으로 시작한다. $S_4 = \{\text{초록}\}$, $S_5 = \{\text{파랑}\}$, $S_6 = \{\text{빨강}\}$, $S_7 = \{\text{노랑}\}$. 마디 2에서 $|S_4| = 1, |S_5| = 1$이니 작은 것을 큰 것에 합친다($S_5$을 $S_4$에 넣는다고 하자). 파랑을 $S_4$로 옮겨 $\{\text{초록, 파랑}\}$이 된다. 마디 2의 빛깔(파랑, 이미 있음)을 더한다. 밑나무 2의 서로 다른 수: 2. 옮김: 1. 마디 3에서 $S_7$을 $S_6$에 합친다. 노랑을 옮긴다. 빨강(이미 있음)을 더한다. 모임: $\{\text{빨강, 노랑}\}$. 수: 2. 옮김: 1. 마디 1에서 $|S_2| = 2, |S_3| = 2$이니 $S_3$을 $S_2$에 합친다. 빨강(이미 있음)을 옮기고 노랑을 옮긴다. 빨강(이미 있음)을 더한다. 모임: $\{\text{초록, 파랑, 빨강, 노랑}\}$. 수: 4. 옮김: 2. 전체 옮김: $1 + 1 + 2 = 4 \le 7 \cdot \lfloor \log_2 7 \rfloor = 14$. $\square$

---

**연습문제 2.**
작은 것을 큰 것에 합치기에서 모든 합치기에 걸친 원소 옮김 수가 모두 $O(n \log n)$임을 증명하여라.

??? success "연습문제 2 풀이"
    원소마다 처음에 0인 셈틀을 매긴다. 원소가 작은 모임에서 큰 모임으로 옮겨질 때 "옮겨졌다"고 한다. 옮긴 뒤 그 원소는 크기가 적어도 $2 \times (\text{그 모임의 앞선 크기})$인 모임에 든다. 합쳐 들어가는 모임이 적어도 온 모임만큼 크기 때문이다. 따라서 옮길 때마다 그 원소가 든 모임의 크기가 적어도 두 배가 된다. 모임 크기의 최댓값이 $n$이므로 원소마다 셈틀은 그 모임이 크기 $n$에 이르기 전에 많아야 $\lfloor \log_2 n \rfloor$번 늘 수 있다. 원소 $n$개가 저마다 많아야 $\lfloor \log_2 n \rfloor$번 옮겨지므로 전체 옮김 수는 많아야 $n \lfloor \log_2 n \rfloor = O(n \log n)$이다. $\square$

---

**연습문제 3.**
마디 $n$개의 나무가 있고 마디마다 정수 값이 있다. 마디 $v$마다 $v$의 밑나무에서 중앙값이 무엇인지 답하여라. 작은 것을 큰 것에 합치기로 이를 잘 풀 수 있는가? 복잡도를 살펴라.

??? success "연습문제 3 풀이"
    마디마다의 밑나무 모임에 대해 차례 통계 나무(C++의 정책 바탕 `__gnu_pbds::tree`나 좌표 옥죄기를 쓴 펜윅 나무처럼 등수 물음을 받치는 균형 두 갈래 찾기 나무)를 지닌다. 작은 것을 큰 것에 합칠 때 작은 모임의 원소를 큰 모임의 차례 통계 나무에 넣는다. 마디 $v$에서 아이를 모두 합친 뒤 $v$ 자신의 값을 넣고 밑나무 크기가 $k$일 때 $\lfloor k/2 \rfloor$번째 원소를 묻는다. 차례 통계 나무에 한 번 넣는 데 $O(\log n)$이 든다. 넣기는 작은 것을 큰 것에 합치기 가둠으로 모두 $O(n \log n)$번이다. 넣기마다 나무에서 $O(\log n)$이 드므로 모든 때는 $O(n \log^2 n)$이다. 중앙값 물음마다 $O(\log n)$이 들어 모든 마디에 대해 $O(n \log n)$이다. 모두: $O(n \log^2 n)$. $\square$

---

**연습문제 4.**
작은 것을 큰 것에 합치기를 "나무 위 DSU"(오일러 돌기와 깊이 먼저 차례) 길과 견주어라. 어떤 조건에서 저마다 더 나은가?

??? success "연습문제 4 풀이"
    둘 다 밑나무 모으기에 모두 $O(n \log n)$을 이룬다. 작은 것을 큰 것에 합치기는 마디마다의 모임을 드러나게 지니고 원소를 옮긴다. 모임이 아무 물음(들어 있음, 등수, 범위)을 받치는 진짜 자료 얼개이므로 더 너그럽다. 나무 위 DSU는 온 자리 자료 얼개 하나를 쓰고 무거운 아이를 마지막에 다루며(그 이바지를 남긴다) 가벼운 아이의 이바지는 되돌린다. 나무 위 DSU는 모임을 다시 잡지 않고 배열 하나만 쓰므로 상수 인수가 작다. 작은 것을 큰 것에 합치기가 나은 때는 (1) 마디마다의 결과를 남겨야 하거나(보기로 다룬 뒤에도 마디마다 모임이 있어야 함), (2) 모으기에 "되돌리기"가 어려운 복잡한 자료 얼개가 필요할 때다. 나무 위 DSU가 나은 때는 (1) 마디마다 답 하나만 필요하고 그것을 온 자리 상태에서 셈할 수 있을 때, (2) 넣기와 빼기 셈이 서로 대칭이고 값싼 때다. $\square$

---

**연습 5.**
변이 즉시 더해지는 숲(나무 여럿)을 다루도록 작은 것을 큰 것에 합치기를 늘려라. 작은 것을 큰 것에 합치는 DSU가 합치기와 모임 물음을 어떻게 받치는지 밝혀라.

??? success "연습 5의 풀이"
    덩이마다 그 원소 자료의 모임(보기로 균형 두 갈래 찾기 나무나 흩임 모임)을 갈무리하는 DSU를 지닌다. 변 $(u, v)$이 더해지면 $u$와 $v$가 든 덩이의 뿌리를 찾는다. 같으면 아무것도 하지 않는다. 아니면 작은 덩이의 모임을 큰 덩이의 모임에 합치고(작은 것을 큰 것에) DSU 대표를 합친다. 옮길 때마다 덩이 크기가 적어도 두 배가 되므로 모든 합치기에 걸쳐 원소마다 많아야 $O(\log n)$번 옮겨진다. 원소 $n$개와 아무 합치기 차례에 대해 모든 합치기의 일감은 $O(n \log n)$이다. 모임 물음(보기로 "$u$의 덩이에 값 $x$이 있는가?")은 그 덩이 뿌리의 균형 두 갈래 찾기 나무로 $O(\log n)$이 든다. 이는 덩이마다의 모음 자료를 지닌 즉시 처리 이음 물음을 셈마다 고르게 나눈 $O(\log n)$에 받친다. $\square$
