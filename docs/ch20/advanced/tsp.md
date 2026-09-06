# 떠돌이 장수
```python
from itertools import permutations
```

```python
def cycle_cost(cycle, graph):
    cost = 0
    start = cycle[0]
    for end in cycle[1:]:
        cost += graph[start][end]
        start = end
    return cost
```

```python
def tsp(graph):
    start = 0
    vertex = list(range(len(graph)))[1:]
    for i, path in enumerate(permutations(vertex)):
        cycle = [start] + list(path) + [start] 
        current_cost = cycle_cost(cycle, graph)
        if i == 0:
            tsp_cost = current_cost
            tsp_cycle = cycle
        elif current_cost < tsp_cost:
            tsp_cost = current_cost
            tsp_cycle = cycle     
    return tsp_cost, tsp_cycle
```

```python
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
```

```python
tsp_cost, tsp_cycle = tsp(graph)
print(f'tsp cost  : {tsp_cost}')
print(f'tsp cycle : {tsp_cycle}')
```

**출력:**
```
tsp cost  : 80
tsp cycle : [0, 1, 3, 2, 0]
```

# 참고 문헌

[7.3 Traveling Salesman Problem - Branch and Bound](https://www.youtube.com/watch?v=1FEP_sNb62k&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=71)

[Traveling Salesman Problem (TSP) Implementation](https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/)

## 연습문제

**연습문제 1.**
행렬 사슬 곱하기 문제를 정의하라. 곱하는 차례가 왜 중요한가?

??? success "연습문제 1 풀이"
    $A_i$의 꼴이 $p_{i-1} \times p_i$인 행렬 $A_1, A_2, \ldots, A_n$이 주어질 때 스칼라 곱셈의 수를 가장 적게 하는 괄호 매김을 찾아라. 행렬 곱은 결합은 되지만 교환은 되지 않고 괄호 매김에 따라 값이 크게 달라지므로 차례가 중요하다. 예컨대 $(10 \times 30) \cdot (30 \times 5) \cdot (5 \times 60)$을 곱할 때 괄호 매김 $((A_1 A_2) A_3)$은 $10 \cdot 30 \cdot 5 + 10 \cdot 5 \cdot 60 = 4500$이 들지만 $(A_1 (A_2 A_3))$은 $30 \cdot 5 \cdot 60 + 10 \cdot 30 \cdot 60 = 27000$이 든다.

---

**연습문제 2.**
행렬 사슬 곱하기 문제의 되돌이 관계식을 쓰고 그 짜임을 설명하라.

??? success "연습문제 2 풀이"
    $m[i][j]$을 $A_i \cdots A_j$을 곱하는 가장 적은 값이라 하자. 바탕 경우: $m[i][i] = 0$. 되돌이 관계식:

    $$m[i][j] = \min_{i \leq k < j} \{m[i][k] + m[k+1][j] + p_{i-1} \cdot p_k \cdot p_j\}$$

    가르는 점 $k$이 사슬을 $(A_i \cdots A_k)$과 $(A_{k+1} \cdots A_j)$으로 나누며, 그렇게 나온 두 행렬을 합치는 값은 $p_{i-1} \cdot p_k \cdot p_j$이다. 여기에 가장 좋은 아래 짜임과 겹치는 아래 문제가 모두 드러난다.

---

**연습문제 3.**
행렬 사슬 곱하기의 동적 짜기 풀이는 시간 복잡도와 공간 복잡도가 무엇인가?

??? success "연습문제 3 풀이"
    아래 문제가 $O(n^2)$개 있다($1 \leq i \leq j \leq n$인 짝 $(i, j)$). 아래 문제마다 가르는 점을 $O(n)$개 살핀다. 전체 시간: $O(n^3)$. 공간: 동적 짜기 표에 $O(n^2)$. 이는 가능한 괄호 매김을 모두 살피는 막무가내 방식, 곧 카탈랑 수 $C_{n-1} = \Omega(4^n / n^{3/2})$에 견주면 엄청난 나아짐이다.

---

**연습문제 4.**
방향 있는 돌기 없는 그래프 위의 동적 짜기가 여느 동적 짜기 얼거리를 어떻게 넓히는지 설명하라.

??? success "연습문제 4 풀이"
    방향 있는 돌기 없는 그래프 위의 동적 짜기에서는 상태 공간이 방향 있는 돌기 없는 그래프를 이루고 변이 상태 사이의 옮아감을 나타낸다. 그 그래프의 위상 정렬 차례가 동적 짜기 표를 채우는 옳은 차례를 정한다. 곧 어떤 상태는 그것이 기대는 상태를 모두 셈한 뒤에야 셈한다. 이는 상태에 흔히 정수나 짝으로 번호를 매기는 여느 동적 짜기를 아무 기댐 짜임에나 쓸 수 있게 넓힌 것이다. 보기: 돌기 없는 그래프의 최단 길(꼭짓점을 위상 정렬 차례로 처리), 앞뒤 제약이 있는 차례 짜기, 나무 짜임 위의 동적 짜기(어버이보다 자식을 먼저 처리).
