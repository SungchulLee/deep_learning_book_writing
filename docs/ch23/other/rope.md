# 밧줄 자료 짜임

배열과 이은 목록 모두 글줄을 잇거나 가운데에 끼울 때 최악의 경우 $O(n)$이 든다. **밧줄**은 잎마다 짧은 글줄 조각을 담고 안쪽 마디마다 왼쪽 아래 나무의 전체 길이를 담는 두 갈래 나무이다. 이 짜임은 잇기, 쪼개기, 끼우기를 $O(\log n)$ 시간에 받쳐 글월 편집기나 큰 글줄을 자주 고치는 다른 쓰임새에 안성맞춤이다.

---

## 1. 구조

밧줄은 다음 성질을 가진 두 갈래 나무이다:

- **잎 마디**는 짧은 글줄 조각을 담는다(흔히 어떤 문턱 길이까지).
- **안쪽 마디**는 왼쪽 아래 나무의 모든 잎의 전체 길이와 같은 `weight` 칸을 담는다.
- 온전한 글줄은 잎을 가운데 차례로 돌아보아 얻는다.

글줄 "Hello_World"에 대해:

```
        [5]
       /    \
    "Hello"  [1]
            /   \
          "_"  "World"
```

뿌리의 무게는 5("Hello"의 길이)이고 오른쪽 자식의 무게는 1("_"의 길이)이다.

---

## 2. 연산

### 번호로 찾기 — O(log n)

자리 $i$의 글자를 찾으려면:

1. $i < \text{weight}$이면 왼쪽 아래 나무로 되돌이한다.
2. 아니면 $i \leftarrow i - \text{weight}$으로 오른쪽 아래 나무로 되돌이한다.
3. 잎에서 그 조각 안의 자리 $i$ 글자를 돌려준다.

### 잇기 — O(1) 또는 O(log n)

두 밧줄을 왼쪽과 오른쪽 자식으로 하는 새 뿌리를 만든다. 무게는 왼쪽 밧줄의 전체 길이이다. 저울질이 필요하면 $O(\log n)$이 든다.

### 쪼개기 — O(log n)

밧줄을 자리 $i$에서 두 밧줄(글자 $0 \ldots i-1$과 $i \ldots n-1$)로 쪼갠다:

1. 자리 $i$으로 찾아가며 길 위의 마디를 쪼갠다.
2. 왼쪽과 오른쪽 몫을 다시 맞춘다.

### 끼우기 — O(log n)

자리 $i$에 글줄을 끼운다. 곧 $i$에서 쪼개고 왼쪽 몫과 새 글줄을 이은 뒤 오른쪽 몫과 잇는다.

### 지우기 — O(log n)

자리 $i$부터 $j$까지의 글자를 지운다. 곧 $i$과 $j$에서 쪼갠 뒤 바깥 몫을 잇는다.

---

## 3. 복잡도 비교

| 연산 | 배열 | 밧줄 |
|---|---|---|
| 번호로 찾기 | $O(1)$ | $O(\log n)$ |
| 잇기 | $O(n)$ | $O(\log n)$ |
| 쪼개기 | $O(n)$ | $O(\log n)$ |
| 끼우기 | $O(n)$ | $O(\log n)$ |
| 삭제 | $O(n)$ | $O(\log n)$ |

---

## 4. 파이썬 구현

```python
"""
밧줄 자료 짜임 — 글줄 조각의 두 갈래 나무.

큰 고칠 수 있는 글줄에 효율 좋은 잇기, 쪼개기, 번호로 찾기,
끼우기를 받친다.
"""

# === 밧줄 마디 ===

class RopeNode:
    """밧줄 두 갈래 나무의 마디."""

    def __init__(
        self, text: str = "",
        left: "RopeNode | None" = None,
        right: "RopeNode | None" = None,
    ):
        if left is None and right is None:
            # 잎 마디
            self.text = text
            self.weight = len(text)
            self.left = None
            self.right = None
        else:
            # 안쪽 마디
            self.text = ""
            self.left = left
            self.right = right
            self.weight = left.total_length() if left else 0

    def total_length(self) -> int:
        """이 아래 나무가 뜻하는 글줄의 전체 길이를 돌려준다."""
        if self.left is None and self.right is None:
            return len(self.text)
        length = self.weight
        if self.right:
            length += self.right.total_length()
        return length

    def index(self, i: int) -> str:
        """자리 i의 글자를 돌려준다."""
        if self.left is None and self.right is None:
            return self.text[i]
        if i < self.weight:
            return self.left.index(i) if self.left else ""
        return self.right.index(i - self.weight) if self.right else ""

    def to_string(self) -> str:
        """가운데 차례로 돌아보아 온전한 글줄을 모은다."""
        if self.left is None and self.right is None:
            return self.text
        result = ""
        if self.left:
            result += self.left.to_string()
        if self.right:
            result += self.right.to_string()
        return result

# === 밧줄 연산 ===

def concatenate(left: RopeNode | None, right: RopeNode | None) -> RopeNode:
    """밧줄 둘을 이어 새 밧줄을 만든다."""
    if left is None:
        return right
    if right is None:
        return left
    return RopeNode(left=left, right=right)

def split(node: RopeNode, i: int) -> tuple[RopeNode | None, RopeNode | None]:
    """밧줄을 자리 i에서 (왼쪽, 오른쪽)으로 쪼갠다."""
    if node.left is None and node.right is None:
        # 잎 마디
        if i <= 0:
            return None, node
        if i >= len(node.text):
            return node, None
        return RopeNode(node.text[:i]), RopeNode(node.text[i:])

    if i < node.weight:
        left_split, right_split = split(node.left, i) if node.left else (None, None)
        return left_split, concatenate(right_split, node.right)
    elif i > node.weight:
        left_split, right_split = (
            split(node.right, i - node.weight) if node.right else (None, None)
        )
        return concatenate(node.left, left_split), right_split
    else:
        return node.left, node.right

def insert(node: RopeNode, i: int, text: str) -> RopeNode:
    """자리 i에 글월을 끼운다."""
    left, right = split(node, i)
    new_leaf = RopeNode(text)
    return concatenate(concatenate(left, new_leaf), right)

def delete(node: RopeNode, i: int, j: int) -> RopeNode:
    """자리 i부터 j 앞까지의 글자를 지운다."""
    left, temp = split(node, i)
    _, right = split(temp, j - i)
    return concatenate(left, right)

# === 메인 ===

if __name__ == "__main__":
    # 조각으로 밧줄을 세운다
    rope = concatenate(RopeNode("Hello"), RopeNode("_World"))
    print(f"Rope: '{rope.to_string()}'")
    print(f"Length: {rope.total_length()}")
    print(f"Index 6: '{rope.index(6)}'")

    # 끼우기
    rope = insert(rope, 5, " Beautiful")
    print(f"After insert: '{rope.to_string()}'")

    # 지우기
    rope = delete(rope, 5, 15)
    print(f"After delete: '{rope.to_string()}'")
    # 내임:
    # 밧줄: 'Hello_World'
    # 길이: 11
    # 번호 6: 'W'
    # 끼운 뒤: 'Hello Beautiful_World'
    # 지운 뒤: 'Hello_World'
```

**출력:**

```
Rope: 'Hello_World'
Length: 11
Index 6: 'W'
After insert: 'Hello Beautiful_World'
After delete: 'Hello_World'
```

---

## 연습문제

**연습문제 1.**
밧줄 자료 짜임의 핵심 자료 짜임이나 개념과 그 으뜸 쓰임새를 설명하라.

??? success "연습문제 1 풀이"
    밧줄 자료 짜임은 글줄이나 차례 자료를 미리 다듬고 묻는 효율 좋은 길을 준다. 으뜸 쓰임새는 부분 글줄, 본, 들임의 짜임 성질에 대한 되풀이되는 물음에 답하는 것이다. 미리 다듬기가 다룰 만한 시간에 자료 짜임을 세우고 나면 맨바닥에서 다시 다듬는 것보다 훨씬 빠르게 물음에 답할 수 있다. $\square$

---

**연습문제 2.**
밧줄 자료 짜임을 세우는 시간 복잡도는 무엇인가? 으뜸 연산의 묻기 시간은 무엇인가?

??? success "연습문제 2 풀이"
    세우는 시간은 쓰는 알고리즘에 달렸다. 흔한 한계는 $n$이 들임 크기일 때 $O(n)$에서 $O(n \log n)$ 사이이다. 묻기는 흔히 본 찾기에 $O(m)$($m$은 물음 길이), 미리 셈한 성질에 $O(1)$이 든다. 공간 복잡도는 흔히 $O(n)$이거나 $\sigma$이 글자 모임의 크기일 때 $O(n\sigma)$이다. $\square$

---

**연습문제 3.**
밧줄 자료 짜임을 더 단순한 다른 방식과 견주어라. 더 정교한 짜임은 언제 값어치가 있는가?

??? success "연습문제 3 풀이"
    더 단순한 방식(예컨대 막무가내 훑기나 정렬)은 묻기 시간이 더 길지만 세우는 군더더기가 적다. 정교한 짜임은 다음일 때 값어치가 있다. (1) 같은 자료에 물음을 많이 던져 세우는 값이 고르게 나뉠 때, (2) 묻기 시간이 결정적일 때(실시간 쓰임새), (3) 자료가 커서 점근 나아짐이 실전에서 중요할 때이다. 작은 자료에 물음을 한 번 던지는 경우에는 상수 인수가 작은 단순한 방식이 더 빠를 수 있다. $\square$

---

**연습문제 4.**
들임 글줄 "banana"에 대해 밧줄 자료 짜임을 세우는 것을 좇아라. 중간 걸음을 보여라.

??? success "연습문제 4 풀이"
    "banana"($n = 6$)에 대해: 글줄을 글자마다(또는 뒷가지마다) 처리하며 자료 짜임을 조금씩 세운다. 마지막 짜임은 뒷가지 "banana", "anana", "nana", "ana", "na", "a"을 모두 담는다. 결과의 핵심 성질을 확인할 수 있다. 곧 공통 앞가지를 나눠 쓰고, 뒷가지 차례가 지켜지며, 부분 글줄에 대한 모든 물음을 그 짜임에서 답할 수 있다. $\square$

## 정리하며

이 마당은 구조、연산、복잡도 비교、파이썬 구현을 차례로 짚었다.

**참고 문헌**

- Boehm, H., Atkinson, R., & Plass, M. (1995). Ropes: An alternative to strings. *Software: Practice and Experience*, 25(12), 1315-1330.
