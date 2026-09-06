# 아호-코라식
아호-코라식 알고리즘은 길이 $n$인 글월 $T$에서 본 모임 $\{P_1, P_2, \ldots, P_k\}$이 나오는 곳을 한꺼번에 모두 찾는 여러 본 글줄 찾기 알고리즘이다. $m = \sum |P_i|$을 본 전체의 길이, $z$을 맞은 수라 하자. 이 알고리즘은 $O(n + m + z)$ 시간에 돈다.

## 개요

이 알고리즘은 두 마디로 돈다:

1. **미리 다듬기:** 본 전체로 트라이(열쇠말 나무)를 세운 뒤 어긋남 이음과 내놓기 이음(사전 이음)을 더해 유한 자동 기계를 만든다.
2. **찾기:** 글월을 글자마다 자동 기계에 흘려 넣는다. 상태마다 그 자리에서 끝나는 본을 모두 알린다.

$$
\text{전체 시간} = \underbrace{O(m)}_{\text{트라이와 이음 세우기}} + \underbrace{O(n)}_{\text{글월 훑기}} + \underbrace{O(z)}_{\text{맞음 알리기}} = O(n + m + z)
$$

## 구현

```python
from collections import deque

class AhoCorasick:
    def __init__(self):
        self.goto = [{}]       # goto 함수(트라이 변)
        self.fail = [0]        # 어긋남 이음
        self.output = [[]]     # 내놓기 함수(본 번호)

    def add_pattern(self, pattern: str, index: int):
        """본을 트라이에 끼운다."""
        state = 0
        for ch in pattern:
            if ch not in self.goto[state]:
                self.goto[state][ch] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            state = self.goto[state][ch]
        self.output[state].append(index)

    def build(self):
        """너비 먼저 돌아보기로 어긋남 이음을 셈하고 내놓기 이음을 퍼뜨린다."""
        queue = deque()
        for ch, s in self.goto[0].items():
            queue.append(s)
            self.fail[s] = 0
        while queue:
            r = queue.popleft()
            for ch, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while state != 0 and ch not in self.goto[state]:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(ch, 0)
                if self.fail[s] == s:
                    self.fail[s] = 0
                self.output[s] = self.output[s] + self.output[self.fail[s]]

    def search(self, text: str, patterns: list[str]) -> list[tuple[int, str]]:
        """글월에서 본이 나오는 곳을 모두 찾는다."""
        results = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for pid in self.output[state]:
                results.append((i - len(patterns[pid]) + 1, patterns[pid]))
        return results

# 예
patterns = ["he", "she", "his", "hers"]
ac = AhoCorasick()
for i, p in enumerate(patterns):
    ac.add_pattern(p, i)
ac.build()

text = "ahishers"
matches = ac.search(text, patterns)
for pos, pat in sorted(matches):
    print(f"  Pattern '{pat}' found at position {pos}")
# 내임:
#   본 'his'을 자리 1에서 찾음
#   본 'he'을 자리 4에서 찾음
#   본 'she'을 자리 3에서 찾음
#   본 'hers'을 자리 4에서 찾음
```

## 복잡도 분석

| 마디 | 시간 | 공간 |
|-------|------|-------|
| 트라이 세우기 | $O(m)$ | $O(m \cdot |\Sigma|)$ |
| 어긋남 이음 세우기 | $O(m)$ | 위에 포함 |
| 찾기 | $O(n + z)$ | 더 드는 공간 $O(1)$ |
| **전체** | $O(n + m + z)$ | $O(m \cdot |\Sigma|)$ |

# 참고 문헌

[Aho, Corasick - Efficient String Matching: An Aid to Bibliographic Search (1975)](https://doi.org/10.1145/360825.360855)

[Aho-Corasick Algorithm - CP-Algorithms](https://cp-algorithms.com/string/aho_corasick.html)

## 연습문제

**연습문제 1.**
아호-코라식이 본 여럿을 한꺼번에 다루려 KMP 어긋남 함수를 어떻게 넓히는지 설명하라.

??? success "연습문제 1 풀이"
    아호-코라식은 본 전체로 트라이(열쇠말 나무)를 세운 뒤 KMP의 어긋남 함수와 닮은 어긋남 이음을 더한다. 어긋남 이음마다 지금 앞가지의 진뒷가지이면서 트라이 속 어떤 본의 앞가지이기도 한 가장 긴 것을 가리킨다. 맞지 않으면 알고리즘이 어긋남 이음을 따라가 글월에서 되짚지 않고 찾기를 이어 간다. 게다가 **사전 이음**(내놓기 이음)이 본이 온전히 맞는 마디를 사슬로 이어 맞는 본이 모두 알려지게 한다. 그래서 글월을 한 번만 지나며 $z$이 맞은 수일 때 $O(n + m + z)$ 시간을 이룬다.

---

**연습문제 2.**
본 {"he", "she", "his", "hers"}에 대한 아호-코라식 자동 기계를 세우고 글월 "ushers"에서 도는 것을 좇아라.

??? success "연습문제 2 풀이"
    트라이의 길은 h-e, s-h-e, h-i-s, h-e-r-s이다. 어긋남 이음: "she"의 뒷가지 "he"이 "he" 길로 이어진다. "ushers"을 처리하면 u(뿌리), s(뿌리->s), h(s->h), e(트라이로 sh->he이며 "she"과 "he"도 맞는다), r(he->her), s(her->hers, "hers"이 맞는다)이다. 찾은 맞음은 모두 자리 1의 "she", 자리 2의 "he", 자리 2의 "hers"이다.

---

**연습문제 3.**
아호-코라식 자동 기계를 세우고 나오는 곳을 모두 찾는 시간 복잡도는 무엇인가?

??? success "연습문제 3 풀이"
    **세우기**: $M = \sum |p_i|$이 본 전체의 길이일 때 트라이 세우기에 $O(M)$이 든다. 어긋남 이음은 트라이를 너비 먼저 돌아보며 $O(M)$ 시간에 셈한다. **찾기**: $n$이 글월 길이일 때 글월 처리에 $O(n)$, 사전 이음으로 맞음 $z$개를 알리는 데 $O(z)$이 든다. **전체**: $O(M + n + z)$. 어떤 알고리즘이든 본을 모두 읽고($O(M)$) 글월을 모두 읽고($O(n)$) 맞음을 모두 내놓아야($O(z)$) 하므로 이것이 가장 좋다.

---

**연습문제 4.**
아호-코라식을 본마다 따로 KMP를 돌리는 것과 견주어라. 아호-코라식이 언제 가장 이로운가?

??? success "연습문제 4 풀이"
    본 $k$개에 KMP를 돌리면 $O(n \cdot k + M)$ 시간이 든다($k$번 지나며 글월을 훑는다). 아호-코라식은 $O(n + M + z)$이라 $k$이라는 인수를 없앤다. 다음일 때 가장 이롭다. (1) 본이 많을 때($k$이 클 때), (2) 글월이 길 때($k$번 지나는 것을 아끼는 것이 중요할 때), (3) 본이 공통 앞가지를 나눠 가질 때(트라이가 따로 둔 어긋남 함수 $k$개보다 촘촘하다). 쓰임새: 그물 오감에서 악성 코드 표지 수천 개를 훑는 침입 알아내기 체계, 본 자료 곳간이 큰 DNA 차례 찾기, 사전이 큰 글월 거르기.
