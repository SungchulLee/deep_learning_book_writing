# 무작위로 만든 이진 탐색 트리

이진 탐색 트리의 최악 높이는 $n - 1$(치우친 사슬)이고 그러면 모든 연산이 $O(n)$이 된다. 그런데 서로 다른 열쇠 $n$개를 고르게 무작위인 순서로 넣으면 만들어지는 트리는 평균적으로 훨씬 균형 잡혀 있다. 무작위로 만든 이진 탐색 트리의 기대 높이를 이해하면 이진 탐색 트리가 왜 실제로 잘 통하는지 알 수 있고, 무작위 퀵 정렬과의 깊은 관계도 드러난다.

---

## 1. 정의

서로 다른 열쇠 $n$개에 대한 **무작위로 만든 이진 탐색 트리**란, 고르게 무작위인 순열 순서로 열쇠를 빈 트리에 표준 [삽입](insertion.md) 알고리즘으로 넣어 만든 트리이다.

!!! note "퀵 정렬과의 관계"
    무작위로 만든 이진 탐색 트리의 구조는 무작위 퀵 정렬의 재귀 트리와 같다. 이진 탐색 트리의 뿌리는 첫 축에 대응하고, 왼쪽 부분 트리는 축보다 작은 원소에, 오른쪽 부분 트리는 축보다 큰 원소에 대응한다. 두 과정은 원소를 똑같은 방식으로 나눈다.

---

## 2. 기대 높이

무작위로 만든 이진 탐색 트리의 핵심 결과는 다음과 같다.

!!! note "정리 (CLRS 12.4)"
    서로 다른 열쇠 $n$개로 무작위로 만든 이진 탐색 트리의 기대 높이는 $O(\log n)$이다.

더 정확히, $h_n$을 열쇠 $n$개로 무작위로 만든 이진 탐색 트리의 높이라 하면 다음과 같다.

$$
E[h_n] \leq 3 \ln n = 3 \log_e n \approx 4.33 \log_2 n
$$

---

## 3. 증명의 얼개

증명은 분석을 다루기 쉽게 하려고 **지수 높이**를 쓴다. 지수 높이를 $Y_n = 2^{h_n}$으로 정의한다. 핵심은 $Y_n$이 지시 확률 변수로 분석하기 좋은 점화식을 만족한다는 것이다.

처음 넣은 열쇠(뿌리)가 열쇠 $n$개 가운데 순위 $i$일 때(각 $i$에 대해 확률 $1/n$으로 일어난다) 왼쪽 부분 트리에는 열쇠가 $i - 1$개, 오른쪽 부분 트리에는 $n - i$개 있으며 둘 다 무작위로 만들어진다. 트리의 높이는 다음과 같다.

$$
h_n = 1 + \max(h_{i-1}, h_{n-i})
$$

지수 높이에 대해서는 다음과 같다.

$$
Y_n = 2 \cdot \max(Y_{i-1}, Y_{n-i}) \leq 2(Y_{i-1} + Y_{n-i})
$$

기댓값을 취하고 무작위 순위의 대칭성을 쓰면 다음을 얻는다.

$$
E[Y_n] \leq \frac{4}{n} \sum_{k=0}^{n-1} E[Y_k]
$$

이 점화식을 풀면 $E[Y_n] = O(n^3)$이 나오고, 따라서 다음이 성립한다.

$$
E[h_n] = E[\log_2 Y_n] \leq \log_2 E[Y_n] = O(\log n)
$$

여기서 부등식은 ($\log$이 오목하므로) 옌센 부등식을 쓴 것이다.

---

## 4. 무작위 노드의 기대 깊이

이와 관련된 결과로 무작위로 만든 이진 탐색 트리에서 노드의 기대 깊이가 있다. 순위가 $i$인 노드의 기대 깊이는 다음과 같다.

$$
E[\text{depth of rank-}i\text{ node}] = H_i + H_{n-i+1} - 1
$$

여기서 $H_k = \sum_{j=1}^{k} 1/j$은 $k$번째 조화수이다. 모든 노드에 대한 평균은 다음과 같다.

$$
\frac{1}{n}\sum_{i=1}^{n} E[\text{depth of rank-}i\text{ node}] = \Theta(\log n)
$$

이는 평균 탐색 시간이 $\Theta(\log n)$임을 확인해 준다.

---

## 5. 실험으로 보이기

```python
"""
무작위로 만든 이진 탐색 트리: 기대 높이 분석.

무작위로 만든 이진 탐색 트리의 기대 높이가 O(log n)이고 정렬된 순서로
넣으면 O(n)이 됨을 실험으로 보인다.
"""

import random
import math

# === 노드 정의 ===

class Node:
    """이진 탐색 트리의 노드."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

# === 이진 탐색 트리 연산 ===

def insert(root, key):
    """열쇠를 넣는다 (n이 클 때를 위해 반복으로 한다)."""
    new_node = Node(key)
    if root is None:
        return new_node
    current = root
    while True:
        if key <= current.key:
            if current.left is None:
                current.left = new_node
                return root
            current = current.left
        else:
            if current.right is None:
                current.right = new_node
                return root
            current = current.right

def height(node):
    """너비 우선 탐색으로 높이를 반복적으로 계산한다."""
    if node is None:
        return -1
    from collections import deque
    queue = deque([(node, 0)])
    max_depth = 0
    while queue:
        current, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        if current.left:
            queue.append((current.left, depth + 1))
        if current.right:
            queue.append((current.right, depth + 1))
    return max_depth

# === 실험 ===

def experiment(n, trials=50):
    """무작위 삽입으로 여러 번 시험하고 평균 높이를 보고한다."""
    heights = []
    for _ in range(trials):
        keys = list(range(n))
        random.shuffle(keys)
        root = None
        for k in keys:
            root = insert(root, k)
        heights.append(height(root))
    return sum(heights) / len(heights)

# === 메인 ===

if __name__ == "__main__":
    random.seed(42)

    print(f"{'n':>8}  {'Sorted h':>10}  {'Random h (avg)':>15}  {'log2(n)':>8}  {'3*ln(n)':>8}")
    print("-" * 58)

    for n in [100, 500, 1000, 5000, 10000]:
        # 정렬된 순서로 삽입
        sorted_root = None
        for k in range(n):
            sorted_root = insert(sorted_root, k)
        sorted_h = height(sorted_root)

        # 무작위 삽입 (여러 번의 평균)
        avg_h = experiment(n, trials=20)

        log2_n = math.log2(n)
        three_ln_n = 3 * math.log(n)

        print(f"{n:>8}  {sorted_h:>10}  {avg_h:>15.1f}  {log2_n:>8.1f}  {three_ln_n:>8.1f}")
```

**출력:**
```
       n    Sorted h  Random h (avg)   log2(n)   3*ln(n)
----------------------------------------------------------
     100         99             12.2       6.6      13.8
     500        499             19.2       9.0      18.6
    1000        999             22.0       10.0      20.7
    5000       4999             27.8       12.3      25.5
   10000       9999             30.8       13.3      27.6
```

실험 결과는 무작위로 만든 이진 탐색 트리의 높이가 $3 \ln n$에 가깝고, 정렬된 순서로 넣었을 때의 최악값 $n - 1$보다 훨씬 낮음을 확인해 준다.

---

## 연습문제

**연습문제 1.**
무작위로 만든 이진 탐색 트리에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 무작위로 만든 이진 탐색 트리을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 무작위로 만든 이진 탐색 트리이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 정의、기대 높이、증명의 얼개、무작위 노드의 기대 깊이을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 12.4절](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
