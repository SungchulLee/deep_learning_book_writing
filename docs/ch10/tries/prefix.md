# 접두사 찾기

트라이의 가장 강력한 특징 가운데 하나는 공통 접두사를 나누어 갖는 모든 문자열을 효율적으로 찾는 능력이다. 해시 표는 소속을 $O(1)$에 살필 수 있지만 항목을 모두 훑지 않고는 접두사로 문자열을 늘어놓을 수 없다. 트라이는 접두사 노드를 (접두사 길이를 $p$이라 할 때) $O(p)$ 시간에 짚고 자손을 모두 모아 접두사 기반 질의를 자연스럽고 효율적으로 만든다.

---

## 1. 알고리즘

접두사 찾기는 두 국면으로 나아간다.

1. **접두사 노드까지 내려간다**: 뿌리에서 시작해 접두사의 글자에 해당하는 경로를 따라간다. 경로가 없으면 트라이에 그 접두사를 가진 문자열이 없다.
2. **자손을 모두 모은다**: 접두사 노드에서 (깊이 우선 찾기 따위로) 잎 노드까지의 모든 경로를 훑으며 찾은 온전한 낱말을 모두 모은다.

---

## 2. 구현

```python
"""접두사 찾기를 갖춘 트라이 (자동 완성 방식으로 늘어놓기).

주어진 접두사를 나누어 갖는 담긴 낱말을 모두 찾는 법을 보인다.
"""

# === 트라이 노드와 트라이 ===
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True

    def _find_node(self, prefix):
        """접두사 경로의 끝에 있는 노드까지 내려간다."""
        node = self.root
        for c in prefix:
            if c not in node.children:
                return None
            node = node.children[c]
        return node

    def starts_with(self, prefix):
        """트라이에서 주어진 접두사로 시작하는 낱말을 모두 돌려준다."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results = []
        self._collect(node, prefix, results)
        return results

    def _collect(self, node, path, results):
        if node.end:
            results.append(path)
        for c, child in sorted(node.children.items()):
            self._collect(child, path + c, results)

# === 메인 ===
if __name__ == "__main__":
    t = Trie()
    for w in ["apple", "app", "application", "bat", "ball", "ban"]:
        t.insert(w)
    print("Prefix 'app':", t.starts_with("app"))
    print("Prefix 'ba':", t.starts_with("ba"))
    print("Prefix 'xyz':", t.starts_with("xyz"))
```

**출력:**
```
Prefix 'app': ['app', 'apple', 'application']
Prefix 'ba': ['ball', 'ban', 'bat']
Prefix 'xyz': []
```

---

## 3. 복잡도

| 국면 | 시간 |
|:---|:---:|
| 접두사 노드까지 내려가기 | $O(p)$ |
| 자손 모두 모으기 | $O(k)$ |
| **모두** | $O(p + k)$ |

여기서 $p$은 접두사의 길이이고 $k$은 맞는 낱말 전체의 글자 수이다. 맞는 낱말마다 적어도 한 번은 들러야 하므로 이것이 최적이다.

---

## 연습문제

**연습문제 1.**
접두사 찾기의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 접두사 찾기를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
접두사 찾기가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.

## 정리하며

이 마당은 알고리즘、구현、복잡도을 차례로 짚었다.

**참고 문헌**

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
