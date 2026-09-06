# 사전 이음
사전 이음(내놓기 이음이나 사전 뒷가지 이음이라고도 한다)은 글월 자리마다 맞는 본을 모두 효율 좋게 셀 수 있게 하는 아호-코라식 자동 기계의 다듬기이다. 어긋남 이음과 나란히 둘째 사슬을 이룬다.

## 왜 필요한가

글월 자리를 처리할 때 지금 상태 $s$이 뒷가지가 본 여럿과 맞는 글줄을 뜻할 수 있다. 상태 $s$의 내놓기에는 $s$ 자체에서 끝나는 본(있다면)뿐 아니라 어긋남 이음 사슬로 닿는 본도 모두 들어야 한다. 글월 자리마다 어긋남 이음 사슬 전체를 막무가내로 따라가면 너무 느리다.

## 정의

사전 이음 $\text{dict}(s)$은 어긋남 이음 사슬에서 내놓기 상태(곧 어떤 본이 끝나는 상태)인 $s$의 가장 가까운 조상을 가리킨다:

$$
\text{dict}(s) = \begin{cases}
f(s) & \text{$f(s)$ 이 내놓기 상태이면}\\
\text{dict}(f(s)) & \text{아니면}
\end{cases}
$$

## 세우기

사전 이음은 어긋남 이음을 세우는 같은 너비 먼저 돌아보기에서 셈한다:

```python
from collections import deque

def build_with_dict_links(goto, fail, output):
    """
    어긋남 이음과 사전 이음을 세운다.

    output[s] = 상태 s에서 끝나는 본 번호의 목록.
    dict_link[s] = 어긋남 사슬로 닿는 가장 가까운 내놓기 상태.
    """
    num_states = len(goto)
    dict_link = [0] * num_states
    queue = deque()

    for ch, s in goto[0].items():
        fail[s] = 0
        dict_link[s] = 0
        queue.append(s)

    while queue:
        r = queue.popleft()
        for ch, s in goto[r].items():
            queue.append(s)
            state = fail[r]
            while state != 0 and ch not in goto[state]:
                state = fail[state]
            fail[s] = goto[state].get(ch, 0)
            if fail[s] == s:
                fail[s] = 0
            if output[fail[s]]:
                dict_link[s] = fail[s]
            else:
                dict_link[s] = dict_link[fail[s]]

    return dict_link

def collect_outputs(state, output, dict_link):
    """주어진 상태에서 맞는 본을 모두 모은다."""
    results = list(output[state])
    s = dict_link[state]
    while s != 0:
        results.extend(output[s])
        s = dict_link[s]
    return results
```

## 복잡도 분석

사전 이음이 없으면 글월 자리마다 어긋남 이음 사슬 전체를 돌아볼 수 있어 최악의 경우 맞음 $z$개를 알리는 데 $O(n \cdot m)$이 들 수 있다. 사전 이음을 쓰면 내놓기 상태의 사슬이 눌린다:

- **세우기:** 너비 먼저 돌아보기의 일부로 $O(m)$.
- **맞음 알리기:** 사전 이음을 한 번 건널 때마다 적어도 하나를 내놓으므로 알리는 맞음마다 $O(1)$.
- **전체 찾기 시간:** $O(n + z)$이며 내놓기에 견주어 가장 좋은 복잡도를 이룬다.

사전 이음은 아호-코라식의 이론상 $O(n + m + z)$ 보장에 꼭 필요하다.

# 참고 문헌

[Aho, Corasick - Efficient String Matching (1975)](https://doi.org/10.1145/360825.360855)

[Aho-Corasick Algorithm - Stanford CS166](https://web.stanford.edu/class/cs166/)

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
