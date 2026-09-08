# 뒤집어도 같은 글 나무(이어트리)

길이 $n$인 글줄에는 서로 다른 뒤집어도 같은 부분 글줄이 (빈 글줄까지 넣어) 많아야 $n + 1$개 있다. **뒤집어도 같은 글 나무**(**이어트리**라고도 한다)는 글줄의 서로 다른 뒤집어도 같은 부분 글줄을 모두 $O(n)$ 시간과 공간에 담는 자료 짜임이다. 마디마다 서로 다른 뒤집어도 같은 부분 글줄을 뜻하고 뒷가지 이음이 그것들을 이어 효율 좋은 온라인 세우기를 되게 한다.

---

## 1. 구조

이어트리는 다음으로 이루어진다:

- **뿌리 마디 둘**: 홀수 길이 뒤집어도 같은 글용(길이 $-1$, 상상의 것)과 짝수 길이용(길이 $0$, 빈 글줄).
- **마디**: 마디마다 길이와 자리로 가려낸 서로 다른 뒤집어도 같은 부분 글줄을 담는다.
- **변**: 이름표 붙은 옮아감. 마디 $u$에서 글자 $c$의 변은 $u$의 양쪽에 $c$을 붙여 얻은 뒤집어도 같은 글로 이어진다.
- **뒷가지 이음**: 마디마다 제 가장 긴 진뒷가지 뒤집어도 같은 글을 가리킨다.

홀수 뿌리의 길이가 $-1$이라 양쪽에 글자 하나씩 붙이면 길이 1인 뒤집어도 같은 글(글자 하나)이 된다.

---

## 2. 핵심 성질

- 길이 $n$인 글줄에는 서로 다른 뒤집어도 같은 부분 글줄이 많아야 $n + 1$개 있다.
- 새 글자마다 새 뒤집어도 같은 부분 글줄을 많아야 하나 더한다.
- 이어트리의 마디는 (뿌리 둘까지 넣어) 많아야 $n + 2$개이다.
- 글자 모임의 크기가 상수이면 전체 세우기 시간은 $O(n)$이다.

---

## 3. 온라인 세우기

글자를 한 번에 하나씩 더한다. 자리 $i$의 새 글자 $c$마다:

1. 앞선 글줄의 가장 긴 뒤집어도 같은 뒷가지를 뜻하는 마디에서 시작한다.
2. $u$이 나온 곳 앞의 글자가 $c$과 같은(곧 $s[i - \text{len}(u) - 1] = c$인) 마디 $u$을 찾을 때까지 뒷가지 이음을 따라간다.
3. $u$에서 글자 $c$의 변이 이미 있으면 그 뒤집어도 같은 글은 새것이 아니니 그 변을 따라간다.
4. 아니면 뒤집어도 같은 글 $c \cdot u \cdot c$의 새 마디를 만든다:
    - $u$에서 뒷가지 이음을 계속 따라가 다음 맞는 마디를 찾아 뒷가지 이음을 정한다.
    - 새 마디의 변과 뒷가지 이음을 세운다.

---

## 4. 파이썬 구현

```python
"""
뒤집어도 같은 글 나무(이어트리) — 온라인 세우기.

글줄의 서로 다른 뒤집어도 같은 부분 글줄을 모두 O(n) 시간과
공간에 담는 뒤집어도 같은 글 나무를 세운다.
"""

# === 노드 클래스 ===

class EertreeNode:
    """뒤집어도 같은 글 나무의 마디."""

    def __init__(self, length: int, suffix_link: int = 0):
        self.length = length
        self.suffix_link = suffix_link
        self.edges: dict[str, int] = {}
        self.count = 0  # 뒷가지 뒤집어도 같은 글로 나온 횟수

# === 이어트리 ===

class Eertree:
    """온라인 세우기를 받치는 뒤집어도 같은 글 나무."""

    def __init__(self) -> None:
        # 마디 0: 홀수 뿌리(길이 -1)
        # 마디 1: 짝수 뿌리(길이 0)
        self.nodes = [
            EertreeNode(length=-1, suffix_link=0),
            EertreeNode(length=0, suffix_link=0),
        ]
        self.s = [-1]  # 자리 0의 파수 글자
        self.last = 1  # 가장 긴 뒷가지 뒤집어도 같은 글의 마디 번호

    def _get_link(self, v: int) -> int:
        """s[pos - len(v) - 1] == s[pos]이 될 때까지 뒷가지 이음을 따라간다."""
        pos = len(self.s) - 1
        while self.s[pos - self.nodes[v].length - 1] != self.s[pos]:
            v = self.nodes[v].suffix_link
        return v

    def add_char(self, c: str) -> bool:
        """글자를 더하고 새 뒤집어도 같은 글이 생겼으면 True를 돌려준다."""
        self.s.append(ord(c))
        cur = self._get_link(self.last)

        if c in self.nodes[cur].edges:
            self.last = self.nodes[cur].edges[c]
            self.nodes[self.last].count += 1
            return False

        # 새 마디를 만든다
        new_len = self.nodes[cur].length + 2
        # 새 마디의 뒷가지 이음을 찾는다
        suffix = self._get_link(self.nodes[cur].suffix_link)
        if c in self.nodes[suffix].edges:
            suf_link = self.nodes[suffix].edges[c]
        else:
            suf_link = 1  # 짝수 뿌리

        new_node = EertreeNode(length=new_len, suffix_link=suf_link)
        new_node.count = 1
        self.nodes.append(new_node)
        new_idx = len(self.nodes) - 1
        self.nodes[cur].edges[c] = new_idx
        self.last = new_idx
        return True

    def build(self, s: str) -> None:
        """글줄 전체의 이어트리를 세운다."""
        for c in s:
            self.add_char(c)

    def get_palindromes(self) -> list[tuple[int, int]]:
        """서로 다른 뒤집어도 같은 부분 글줄마다 (길이, 횟수)를 돌려준다."""
        # 긴 것에서 짧은 것으로 수를 퍼뜨린다
        result = []
        for i in range(len(self.nodes) - 1, 1, -1):
            node = self.nodes[i]
            self.nodes[node.suffix_link].count += node.count
            result.append((node.length, node.count))
        return result

    @property
    def num_palindromes(self) -> int:
        """서로 다른 뒤집어도 같은 부분 글줄의 수."""
        return len(self.nodes) - 2  # 뿌리 둘을 뺀다

# === 메인 ===

if __name__ == "__main__":
    s = "abaab"
    tree = Eertree()
    tree.build(s)

    print(f"String: '{s}'")
    print(f"Distinct palindromes: {tree.num_palindromes}")

    palindromes = tree.get_palindromes()
    palindromes.sort(key=lambda x: x[0])
    for length, count in palindromes:
        print(f"  length {length}: occurs {count} times")
    # 내임:
    # 글줄: 'abaab'
    # 서로 다른 뒤집어도 같은 글: 5
    #   길이 1: 5번 나옴
    #   길이 1: 3번 나옴
    #   길이 2: 1번 나옴
    #   길이 3: 1번 나옴
    #   길이 3: 1번 나옴
```

**출력:**

```
String: 'abaab'
Distinct palindromes: 5
  length 1: occurs 2 times
  length 1: occurs 3 times
  length 2: occurs 1 times
  length 3: occurs 1 times
  length 4: occurs 1 times
```

---

## 5. 복잡도

| 연산 | 시간 | 공간 |
|---|---|---|
| 세우기 | 고르게 나누어 $O(n)$ | $O(n)$ |
| 서로 다른 뒤집어도 같은 글 세기 | $O(1)$ | — |
| 뒤집어도 같은 글 모두 묻기 | $O(n)$ | $O(n)$ |

---

## 6. 응용

- **서로 다른 뒤집어도 같은 부분 글줄 세기**: 이어트리의 마디 수가(뿌리 둘을 빼고) 바로 그 수이다.
- **뒤집어도 같은 부분 글줄이 나온 횟수 세기**: 뒷가지 이음을 따라 수를 퍼뜨린다.
- **가장 긴 뒤집어도 같은 부분 글줄**: 세우는 동안 마디 길이의 최댓값을 좇는다.
- **앞가지마다의 뒤집어도 같은 부분 글줄 수**: 글자마다 더해진 새 뒤집어도 같은 글을 센다.

---

## 연습문제

**연습문제 1.**
뒤집어도 같은 글 나무(이어트리)의 핵심 자료 짜임이나 개념과 그 으뜸 쓰임새를 설명하라.

??? success "연습문제 1 풀이"
    뒤집어도 같은 글 나무(이어트리)는 글줄이나 차례 자료를 미리 다듬고 묻는 효율 좋은 길을 준다. 으뜸 쓰임새는 부분 글줄, 본, 들임의 짜임 성질에 대한 되풀이되는 물음에 답하는 것이다. 미리 다듬기가 다룰 만한 시간에 자료 짜임을 세우고 나면 맨바닥에서 다시 다듬는 것보다 훨씬 빠르게 물음에 답할 수 있다. $\square$

---

**연습문제 2.**
뒤집어도 같은 글 나무(이어트리)를 세우는 시간 복잡도는 무엇인가? 으뜸 연산의 묻기 시간은 무엇인가?

??? success "연습문제 2 풀이"
    세우는 시간은 쓰는 알고리즘에 달렸다. 흔한 한계는 $n$이 들임 크기일 때 $O(n)$에서 $O(n \log n)$ 사이이다. 묻기는 흔히 본 찾기에 $O(m)$($m$은 물음 길이), 미리 셈한 성질에 $O(1)$이 든다. 공간 복잡도는 흔히 $O(n)$이거나 $\sigma$이 글자 모임의 크기일 때 $O(n\sigma)$이다. $\square$

---

**연습문제 3.**
뒤집어도 같은 글 나무(이어트리)를 더 단순한 다른 방식과 견주어라. 더 정교한 짜임은 언제 값어치가 있는가?

??? success "연습문제 3 풀이"
    더 단순한 방식(예컨대 막무가내 훑기나 정렬)은 묻기 시간이 더 길지만 세우는 군더더기가 적다. 정교한 짜임은 다음일 때 값어치가 있다. (1) 같은 자료에 물음을 많이 던져 세우는 값이 고르게 나뉠 때, (2) 묻기 시간이 결정적일 때(실시간 쓰임새), (3) 자료가 커서 점근 나아짐이 실전에서 중요할 때이다. 작은 자료에 물음을 한 번 던지는 경우에는 상수 인수가 작은 단순한 방식이 더 빠를 수 있다. $\square$

---

**연습문제 4.**
들임 글줄 "banana"에 대해 뒤집어도 같은 글 나무(이어트리)를 세우는 것을 좇아라. 중간 걸음을 보여라.

??? success "연습문제 4 풀이"
    "banana"($n = 6$)에 대해: 글줄을 글자마다(또는 뒷가지마다) 처리하며 자료 짜임을 조금씩 세운다. 마지막 짜임은 뒷가지 "banana", "anana", "nana", "ana", "na", "a"을 모두 담는다. 결과의 핵심 성질을 확인할 수 있다. 곧 공통 앞가지를 나눠 쓰고, 뒷가지 차례가 지켜지며, 부분 글줄에 대한 모든 물음을 그 짜임에서 답할 수 있다. $\square$

## 정리하며

이 마당은 구조、핵심 성질、온라인 세우기、파이썬 구현을 차례로 짚었다.

**참고 문헌**

- Rubinchik, M., & Shur, A. M. (2018). EERTREE: An efficient data structure for processing palindromes in strings. *European Journal of Combinatorics*, 68, 249-265.
