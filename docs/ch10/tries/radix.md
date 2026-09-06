# 기수 트리

표준 트라이에서는 열쇠의 글자마다 노드를 하나씩 차지한다. 긴 열쇠가 공통 접두사를 거의 나누어 갖지 않으면 자식이 하나뿐인 노드의 사슬이 생겨 기억과 훑는 시간을 모두 버린다. **기수 트리**(**눌러 담은 트라이**나 **패트리샤 트라이**라고도 한다)는 자식이 하나뿐인 노드의 사슬마다 부분 문자열 전체를 이름표로 단 변 하나로 합쳐 이 군더더기를 없앤다. 그 결과 내부 노드의 수가 열쇠의 길이와 무관하게 담긴 열쇠의 수를 결코 넘지 않는 트리가 된다.

## 트라이에서 기수 트리로

열쇠 `"romane"`, `"romanus"`, `"romulus"`, `"rubens"`, `"ruber"`을 표준 트라이에 넣는다고 하자. 뿌리를 지나 `r → o → m → a → n`으로 가는 경로는 저마다 자식이 꼭 하나인 노드로 이루어진다. 변 이름표 `"roman"` 하나로 담을 수 있는 것을 노드 다섯 개로 나타내는 셈이다. 기수 트리는 바로 이런 외자식 사슬을 눌러 담는다.

!!! note "눌러 담는 규칙"
    기수 트리의 내부 노드는 (뿌리만 빼면) 자식이 적어도 둘이다. 노드의 자식이 꼭 하나가 될 때마다 그 노드를 자식과 합치고 변의 이름표를 이어 붙인다.

## 엄밀한 정의

낱자 집합 $\Sigma$ 위의 기수 트리는 문자열 집합 $S$을 담는다. 변마다 (비어 있지 않은 문자열인) 이름표 $\ell \in \Sigma^+$을 지닌다. (열쇠가 하나뿐인 뿌리를 빼면) 어떤 내부 노드 $v$이든 자식이 적어도 둘이다. 노드에 딸린 열쇠는 뿌리에서 그 노드까지의 경로에 있는 변 이름표를 이어 붙인 것이다. 해당 열쇠가 $S$에 속하면 그 노드를 **끝점**으로 표시한다.

**공간 복잡도.** 열쇠 $n$개를 담은 기수 트리는 끝점 노드가 많아야 $n$개, (뿌리를 빼고) 내부 노드가 많아야 $n - 1$개여서 모두 $O(n)$개이며 열쇠의 길이와 무관하다.

## 노드 구조

기수 트리의 노드마다 다음을 담는다.

- 나가는 변 이름표마다 그 **첫 글자**를 `(이름표, 자식)` 쌍으로 잇대는 사전.
- 그 노드가 온전한 열쇠를 나타내는지 알려 주는 참거짓 깃발.

```python
"""
기수 트리(눌러 담은 트라이) 구현.

공간을 아끼는 눌러 담은 트라이 짜임에서 삽입과 찾기와
접두사 모으기와 삭제를 보인다.
"""


# === 노드 정의 ===

class RadixNode:
    """기수 트리의 노드 하나."""

    def __init__(self):
        self.children = {}   # 첫_글자 -> (이름표, 자식_노드)
        self.is_terminal = False


# === 기수 트리 ===

class RadixTree:
    """외자식 사슬을 변의 이름표로 합치는, 눌러 담은 트라이."""

    def __init__(self):
        self.root = RadixNode()

    # --- 삽입 ---

    def insert(self, key: str) -> None:
        """기수 트리에 열쇠를 넣는다.

        맞는 변의 이름표를 따라 트리를 내려간다. 변의 이름표 도중에
        어긋남이 생기면 그 어긋난 자리에서 변을 쪼개고
        새 가지를 붙인다.
        """
        node = self.root
        i = 0  # 열쇠에서의 자리

        while i < len(key):
            ch = key[i]
            if ch not in node.children:
                # 맞는 변이 없다 — 남은 부분으로 새 변을 만든다
                node.children[ch] = (key[i:], RadixNode())
                node.children[ch][1].is_terminal = True
                return

            label, child = node.children[ch]
            # 변의 이름표가 열쇠와 얼마나 맞는지 찾는다
            j = 0
            while j < len(label) and i + j < len(key) and label[j] == key[i + j]:
                j += 1

            if j == len(label):
                # 변이 온전히 맞는다 — 자식에서 이어 간다
                node = child
                i += j
            else:
                # 일부만 맞는다 — 자리 j에서 변을 쪼갠다
                split_node = RadixNode()
                # 지금 노드에서 쪼갠 노드로 가는 변: label[:j]
                # 쪼갠 노드에서 본디 자식으로 가는 변: label[j:]
                split_node.children[label[j]] = (label[j:], child)
                node.children[ch] = (label[:j], split_node)

                if i + j < len(key):
                    # 남은 열쇠는 쪼갠 노드에서 나가는 새 변이 된다
                    remainder = key[i + j:]
                    new_node = RadixNode()
                    new_node.is_terminal = True
                    split_node.children[remainder[0]] = (remainder, new_node)
                else:
                    # 열쇠가 쪼갠 자리에서 꼭 끝난다
                    split_node.is_terminal = True
                return

        # 열쇠가 이미 있는 노드에서 꼭 다 떨어진다
        node.is_terminal = True

    # --- 찾기 ---

    def search(self, key: str) -> bool:
        """열쇠가 기수 트리에 담겨 있으면 True를 돌려준다."""
        node = self.root
        i = 0

        while i < len(key):
            ch = key[i]
            if ch not in node.children:
                return False

            label, child = node.children[ch]
            if not key[i:i + len(label)] == label:
                return False
            i += len(label)
            node = child

        return node.is_terminal

    # --- 접두사 찾기 ---

    def starts_with(self, prefix: str) -> bool:
        """담긴 열쇠 가운데 주어진 접두사로 시작하는 것이 있으면 True를 돌려준다."""
        node = self.root
        i = 0

        while i < len(prefix):
            ch = prefix[i]
            if ch not in node.children:
                return False

            label, child = node.children[ch]
            remaining = len(prefix) - i
            if remaining <= len(label):
                return prefix[i:] == label[:remaining]
            if not prefix[i:i + len(label)] == label:
                return False
            i += len(label)
            node = child

        return True

    # --- 모든 열쇠 모으기 ---

    def _collect(self, node: RadixNode, prefix: str, results: list):
        """노드 아래의 모든 열쇠를 재귀적으로 모은다."""
        if node.is_terminal:
            results.append(prefix)
        for ch in sorted(node.children):
            label, child = node.children[ch]
            self._collect(child, prefix + label, results)

    def all_keys(self) -> list:
        """모든 열쇠를 정렬된 순서로 돌려준다."""
        results = []
        self._collect(self.root, "", results)
        return results

    # --- 삭제 ---

    def delete(self, key: str) -> bool:
        """기수 트리에서 열쇠를 지운다. 찾았으면 True를 돌려준다."""
        return self._delete(self.root, key, 0)

    def _delete(self, node: RadixNode, key: str, depth: int) -> bool:
        if depth == len(key):
            if not node.is_terminal:
                return False
            node.is_terminal = False
            return True

        ch = key[depth]
        if ch not in node.children:
            return False

        label, child = node.children[ch]
        if not key[depth:depth + len(label)] == label:
            return False

        found = self._delete(child, key, depth + len(label))
        if not found:
            return False

        # 뒷정리: 자식 없는 비끝점 노드를 없앤다
        if not child.is_terminal and not child.children:
            del node.children[ch]
        # 합치기: 자식의 자식이 꼭 하나이면 변을 눌러 담는다
        elif not child.is_terminal and len(child.children) == 1:
            only_ch = next(iter(child.children))
            only_label, only_grandchild = child.children[only_ch]
            node.children[ch] = (label + only_label, only_grandchild)

        return True


# === 시연 ===

if __name__ == "__main__":
    tree = RadixTree()

    words = ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon", "ruler"]
    for w in words:
        tree.insert(w)

    print("All keys:", tree.all_keys())
    print("Search 'romane':", tree.search("romane"))
    print("Search 'roman':", tree.search("roman"))
    print("Starts with 'rom':", tree.starts_with("rom"))

    tree.delete("romane")
    print("After deleting 'romane':", tree.all_keys())
```

**출력:**
```
All keys: ['romane', 'romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'ruler']
Search 'romane': True
Search 'roman': False
Starts with 'rom': True
After deleting 'romane': ['romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'ruler']
```

## 삽입이 도는 방식

삽입은 열쇠를 글자 하나씩 따라가며 변의 이름표와 맞추어 본다.

1. **맞는 변이 없을 때.** 남은 열쇠를 이름표로 하여 지금 노드에서 새 변을 만든다.
2. **변이 온전히 맞을 때.** 변의 이름표를 다 쓰고 남은 열쇠로 자식 노드에서 이어 간다.
3. **자리 $j$에서 변이 일부만 맞을 때.** 자리 $j$에서 변을 둘로 쪼갠다. 맞은 접두사가 새 중간 노드의 이름표가 되고 맞지 않은 접미사가 자식 변이 된다. 남은 열쇠는 그 중간 노드의 또 다른 자식 변이 된다.

삽입마다 열쇠의 길이를 $m$이라 할 때 노드를 많아야 $O(m)$개 건드린다.

## 찾기가 도는 방식

찾기는 삽입을 그대로 따르되 트리를 결코 고치지 않는다. 노드마다 나가는 변의 이름표가 질의 열쇠의 해당 부분과 맞는지 살핀다.

- 변의 이름표가 온전히 맞으면 자식으로 나아가 이어 간다.
- 변의 이름표가 맞지 않으면 그 열쇠는 없다.
- 어느 노드에서 열쇠가 다 떨어지면 끝점 깃발을 살핀다.

찾기는 질의 열쇠의 길이를 $m$이라 할 때 $O(m)$ 시간에 돈다.

## 복잡도 분석

$n$을 담긴 열쇠의 수, $m$을 다루는 열쇠의 길이라 하자.

| 연산 | 시간 복잡도 | 공간 복잡도 |
|-----------|----------------|-----------------|
| 삽입    | $O(m)$         | 새 변 $O(m)$개 |
| 찾기    | $O(m)$         | $O(1)$          |
| 삭제    | $O(m)$         | $O(1)$          |
| 공간     | —              | 모두 $O(n \cdot \bar{m})$ |

여기서 $\bar{m}$은 열쇠의 평균 길이이다. $O(n)$이라는 노드 한계 덕분에 열쇠가 길 때 기수 트리가 표준 트라이보다 공간을 훨씬 아낀다.

## 기수 트리와 표준 트라이

| 성질 | 표준 트라이 | 기수 트리 |
|----------|--------------|------------|
| 노드 | $O(n \cdot m_{\max})$ | $O(n)$ |
| 외자식 사슬 | 있음 | 없앰 |
| 변의 이름표 | 글자 하나 | 부분 문자열 |
| 구현의 까다로움 | 간단 | 보통 (변 쪼개기) |
| 캐시 움직임 | 나쁨 (포인터를 많이 건넌다) | 나음 (노드가 적다) |

열쇠의 수에 견주어 열쇠가 길수록 기수 트리의 이점이 커진다. 낱자 집합이 작고 열쇠가 짧으면 더 간단한 표준 트라이가 나을 수 있다.

## 응용

기수 트리는 실제 시스템 곳곳에 나온다.

- **IP 경로 표.** IP 주소의 이진 표현에 대한 최장 접두사 맞추기.
- **메모리 안 데이터베이스.** 적응 기수 트리(ART)가 HyPer 같은 요즘 데이터베이스의 색인 짜임을 떠받친다.
- **리눅스 커널.** 커널이 쪽 캐시 찾기와 다른 안쪽 잇댐에 기수 트리를 쓴다.
- **자동 완성과 맞춤법 검사.** 접두사 기반 찾기가 눌러 담은 짜임의 덕을 본다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 12.
- Morrison, D. R. (1968). PATRICIA — Practical Algorithm to Retrieve Information Coded in Alphanumeric. *Journal of the ACM*, 15(4), 514-534.


## 연습문제

**연습문제 1.**
기수 트리의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 기수 트리를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
기수 트리가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.