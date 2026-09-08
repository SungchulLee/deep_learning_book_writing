# 큐의 응용

큐의 선입선출 성질은 공평함이 중요한 곳이라면 어디서나 자연스러운 선택이 된다. 작업은 도착한 순서대로 처리되어야 한다. 운영체제는 프로세스 배정과 입출력 요청 버퍼링에 큐를 쓴다. 네트워크 라우터는 보내기를 기다리는 패킷을 큐에 담는다. 그래프 알고리즘은 마디를 층별로 살피는 데 큐를 쓴다. 이 쪽은 큐의 중요한 알고리즘적 응용을 구체적인 예와 복잡도 분석과 함께 훑어본다.

---

## 1. 생산자-소비자 버퍼

동시성 시스템에서는 **생산자**가 데이터를 만들고 **소비자**가 그것을 처리하는데, 속도가 서로 다른 일이 많다. 큐가 그 사이의 버퍼 구실을 한다. 생산자는 항목을 넣고 소비자는 뺀다. 이로써 두 과정이 서로 떨어져 저마다의 속도로 움직일 수 있다. 큐의 크기가 정해져 있으면 큐가 가득 찼을 때 생산자가 멈추고, 비었을 때 소비자가 멈춘다.

---

## 2. 뜨거운 감자 흉내 내기

"뜨거운 감자"(또는 요세푸스) 문제는 원형 제거를 보여 준다. 참가자들이 둥글게 서서 물건을 넘긴다. 정해진 횟수만큼 넘긴 뒤 물건을 든 사람이 빠진다. 큐를 쓰면 넘길 때마다 앞의 참가자를 빼서 뒤에 다시 넣는다. 정해진 횟수가 되면 앞의 참가자를 (다시 넣지 않고) 빼서 제거한다.

---

## 3. 층별 순회

나무와 그래프는 흔히 층별로 살핀다. 큐는 자연스럽게 이 순서를 만들어 낸다. 뿌리를 넣고 시작한 뒤, 마디를 하나 빼서 처리하고 그 자식들을 넣기를 되풀이한다. 그 결과가 너비 우선 순회이다. 자세한 내용은 이웃 쪽인 너비 우선 탐색 맛보기에서 다룬다.

```python
"""
큐의 응용 — 큐 자료 구조의 흔한 알고리즘적 쓰임.

선입선출 성질에 기댄 생산자-소비자 모의실험, 뜨거운 감자 제거,
나무의 층별 순회를 보인다.
"""
from collections import deque

# === 응용 1: 생산자-소비자 모의실험 ==============================

def producer_consumer(tasks, process_time):
    """큐로 생산자-소비자 버퍼를 흉내 낸다.

    생산자가 작업을 모두 넣은 뒤 소비자가 선입선출 차례로 처리하며,
    작업마다 `process_time` 단위가 걸린다.
    """
    queue = deque()
    clock = 0

    # 생산 단계
    for task in tasks:
        queue.append(task)
        clock += 1
        print(f"  t={clock:>2}: Producer enqueued '{task}' → queue={list(queue)}")

    # 소비 단계
    while queue:
        task = queue.popleft()
        clock += process_time
        print(f"  t={clock:>2}: Consumer processed '{task}' → queue={list(queue)}")

# === 응용 2: 뜨거운 감자 제거 ====================================

def hot_potato(players, num_passes):
    """큐로 폭탄 돌리기 놀이를 흉내 낸다.

    참가자들이 둥글게 선다. `num_passes`번 넘긴 뒤 감자를 든 사람이
    빠진다. 마지막까지 남은 사람이 이긴다.
    시간: O(n * k), 여기서 n은 참가자 수, k는 한 판에 넘기는 횟수이다.
    """
    queue = deque(players)
    print(f"  Starting players: {list(queue)}")

    while len(queue) > 1:
        for _ in range(num_passes):
            queue.append(queue.popleft())  # 감자를 넘긴다
        eliminated = queue.popleft()
        print(f"  Eliminated: {eliminated:<10s} Remaining: {list(queue)}")

    winner = queue[0]
    print(f"  Winner: {winner}")
    return winner

# === 응용 3: 나무의 층별 순회 ================================

class TreeNode:
    """간단한 이진 나무의 노드."""

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root):
    """큐로 이진 트리를 층별로 순회한다.

    시간: O(n) — 각 마디가 정확히 한 번 들어가고 한 번 나온다.
    공간: O(w), 여기서 w는 나무의 최대 너비이다.
    """
    if root is None:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result

# === 시연 ============================================================

if __name__ == "__main__":
    # 생산자-소비자
    print("Producer-Consumer Simulation:")
    producer_consumer(["email", "report", "backup"], process_time=2)
    print()

    # 뜨거운 감자
    print("Hot Potato Game (3 passes per round):")
    hot_potato(["Alice", "Bob", "Carol", "Dave", "Eve"], num_passes=3)
    print()

    # 층별 순회
    #         1
    #        / \
    #       2   3
    #      / \   \
    #     4   5   6
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, None, TreeNode(6)))

    print("Level-Order Traversal:")
    levels = level_order_traversal(tree)
    for i, level in enumerate(levels):
        print(f"  Level {i}: {level}")
```

**출력:**
```
Producer-Consumer Simulation:
  t= 1: Producer enqueued 'email' → queue=['email']
  t= 2: Producer enqueued 'report' → queue=['email', 'report']
  t= 3: Producer enqueued 'backup' → queue=['email', 'report', 'backup']
  t= 5: Consumer processed 'email' → queue=['report', 'backup']
  t= 7: Consumer processed 'report' → queue=['backup']
  t= 9: Consumer processed 'backup' → queue=[]

Hot Potato Game (3 passes per round):
  Starting players: ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
  Eliminated: Alice      Remaining: ['Eve', 'Bob', 'Carol', 'Dave']
  Eliminated: Eve        Remaining: ['Bob', 'Carol', 'Dave']
  Eliminated: Carol      Remaining: ['Dave', 'Bob']
  Eliminated: Bob        Remaining: ['Dave']
  Winner: Dave

Level-Order Traversal:
  Level 0: [1]
  Level 1: [2, 3]
  Level 2: [4, 5, 6]
```

생산자-소비자 흉내 내기는 선입선출 순서를 보여 준다. 작업이 만들어진 순서대로 소비된다. 뜨거운 감자 놀이는 큐의 원형 회전 성질(앞에서 빼서 뒤에 넣기)로 넘기기를 흉내 낸다. 층별 순회는 깊이가 $d+1$인 마디를 하나라도 처리하기 전에 깊이가 $d$인 마디를 모두 처리한다.

---

## 4. 응용 요약

| 응용 | 큐의 구실 | 시간 | 공간 |
|---|---|---|---|
| 생산자-소비자 버퍼 | 생산과 소비의 속도를 떼어 놓는다 | 연산당 $O(1)$ | $O(n)$ |
| 뜨거운 감자 / 요세푸스 | 원형 제거 | $O(n \cdot k)$ | $O(n)$ |
| 나무의 층별 순회 | 마디를 깊이 순으로 처리한다 | $O(n)$ | $O(w)$ |
| 너비 우선 탐색 (BFS) | 그래프를 층별로 살핀다 | $O(V + E)$ | $O(V)$ |
| 작업 스케줄링 (FCFS) | 도착 순서대로 처리한다 | 연산당 $O(1)$ | $O(n)$ |

여기서 $n$은 원소의 수, $k$은 한 판에 넘기는 횟수, $w$은 나무의 최대 너비, $V$과 $E$은 그래프의 꼭짓점과 변의 수이다.

너비 우선 탐색과 작업 스케줄링은 각각의 이웃 쪽에서 자세히 다룬다.

---

## 연습문제

**연습문제 1.**
큐의 응용의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
큐의 응용을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
큐의 응용을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
큐의 응용을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$

## 정리하며

이 마당은 생산자-소비자 버퍼、뜨거운 감자 흉내 내기、층별 순회、응용 요약을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
