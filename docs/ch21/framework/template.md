# 되짚기 본

모든 되짚기 알고리즘은 같은 되돌이 뼈대를 나눠 갖는다. 곧 한 번에 결정 하나씩 어중간한 풀이를 세우고, 결정마다 제약을 살피고, 어중간한 풀이가 더는 옳은 답으로 이어질 수 없으면 그 결정을 되돌린다(되짚는다). 이 공통 짜임을 알면 두루 쓰는 본을 하나 쓰고 N-여왕, 스도쿠, 그래프 색칠하기처럼 서로 다른 문제에 맞춰 쓸 수 있다.

## 두루 쓰는 가짜 부호

아래 본이 되짚기의 요긴한 흐름을 담는다. 맞춰 쓰는 세 곳, 곧 `is_solution`, `candidates`, `make_move`/`undo_move`만 문제마다 달라진다.

```
BACKTRACK(state, decisions):
    if is_solution(state):
        process(state)          # 풀이를 적거나 찍거나 센다
        return

    for choice in candidates(state, decisions):
        if is_valid(state, choice):
            make_move(state, choice)
            BACKTRACK(state, decisions + 1)
            undo_move(state, choice)   # 다음 고름을 시험하기 앞서 상태를 되돌린다
```

본의 조각마다 제 몫이 있다:

| 부품 | 하는 일 |
|-----------|------|
| `state` | 지금 어중간한 풀이(예컨대 판의 짜임, 배정 벡터) |
| `decisions` | 여태 몇 번 결정했는지(상태 공간 나무에서의 지금 깊이) |
| `is_solution` | $n$번 결정이 모두 끝나면 `True`을 돌려준다 |
| `candidates` | 다음 결정에 쓸 수 있는 값을 만든다 |
| `is_valid` | `choice`이 지금 어중간한 풀이와 어긋나지 않는지 살핀다 |
| `make_move` | `choice`을 넣어 `state`을 넓힌다 |
| `undo_move` | `make_move`을 되돌려 `state`을 앞선 꼴로 되돌린다 |

## 파이썬 구현

아래 두루 쓰는 짜기가 파이썬으로 본을 보인다. 구체적인 문제는 도움 방법을 덮어쓴다.

```python
"""
두루 쓰는 되짚기 본.

되짚기 흐름을 감싼 바탕 갈래를 준다.
아래 갈래가 방법 다섯을 덮어써서 특정 조합 문제를 푼다.
"""


# === 두루 쓰는 되짚기 얼거리 ==========================================

class BacktrackingSolver:
    """되짚기 알고리즘의 바탕 갈래.

    아래 갈래가 짜야 할 것:
        is_solution, process, candidates, make_move, undo_move
    """

    def __init__(self):
        self.solutions = []

    def solve(self, state, depth=0):
        """깊이 *depth*의 *state*에서 시작해 되짚기 찾기를 돌린다."""
        if self.is_solution(state, depth):
            self.process(state)
            return
        for choice in self.candidates(state, depth):
            if self.is_valid(state, choice):
                self.make_move(state, choice)
                self.solve(state, depth + 1)
                self.undo_move(state, choice)

    # --- 맞춰 쓰는 곳(아래 갈래에서 덮어쓴다) ---

    def is_solution(self, state, depth):
        raise NotImplementedError

    def process(self, state):
        raise NotImplementedError

    def candidates(self, state, depth):
        raise NotImplementedError

    def is_valid(self, state, choice):
        raise NotImplementedError

    def make_move(self, state, choice):
        raise NotImplementedError

    def undo_move(self, state, choice):
        raise NotImplementedError
```

## 본의 뜯어보기

### 결정 지점

되돌이 부름마다 알고리즘은 **결정 지점**, 곧 상태 공간 나무의 한 마디에 선다. `candidates` 함수가 그 마디에서 나가는 갈래를 센다. 자리 바꿈 문제에서 깊이 $k$의 후보는 아직 쓰지 않은 $n - k$개 원소이고, $m$-색칠 문제에서는 쓸 수 있는 색 $m$개이다.

### 제약 살피기

`is_valid` 함수가 **될 수 있는지 살피기**를 한다. 지금 어중간한 풀이에 `choice`을 더하면 어떤 제약을 어기는지 정한다. 잘 짠 `is_valid`은 $O(1)$이나 $O(k)$($k$은 지금 깊이) 시간에 돌아 마디마다의 군더더기가 나무 돌아보기에 견주어 작게 남는다.

### 상태 고치기와 되돌리기

`make_move`과 `undo_move` 짝은 아래 나무를 살핀 뒤 알고리즘이 상태를 처음 그대로 두게 한다. 이 **되돌림 뜻**이 들른 마디를 그냥 버리는 수수한 깊이 먼저와 되짚기를 가른다:

1. `make_move(state, choice)` — `state`을 제자리에서 고친다.
2. 자식 아래 나무로 되돌이한다.
3. `undo_move(state, choice)` — 형제가 본디 상태를 보도록 고침을 되돌린다.

!!! warning "되돌리기를 잊음"

    가장 흔한 되짚기 벌레는 `undo_move`을 온전히 하지 않는 것이다. 딸림 자료 짜임(예컨대 쓴 원소의 모임, 부딪침 세개)을 `make_move`에서 새로 고치고 `undo_move`에서 되돌리지 않으면 뒤의 갈래가 망가진 상태를 보고 틀린 결과를 낸다.

### 풀이 알아내기

`is_solution` 함수는 지금 깊이가 $n$인지(결정을 모두 했는지) 살핀다. 가장 좋게 하기 변형에서는 지금 풀이가 여태 가장 좋은 목표를 넘는지도 살필 수 있다.

## 풀이 하나 찾기와 모두 찾기

위 본은 풀이를 **모두** 찾는다. 흔한 변형 둘이 흐름을 고친다:

**첫 풀이 찾기** — `process` 뒤에 곧바로 돌아간다:

```
BACKTRACK(state, decisions):
    if is_solution(state):
        process(state)
        return True             # 신호: 찾기를 멈춘다

    for choice in candidates(state, decisions):
        if is_valid(state, choice):
            make_move(state, choice)
            if BACKTRACK(state, decisions + 1):
                return True     # 멈춤 신호를 퍼뜨린다
            undo_move(state, choice)
    return False
```

**가장 좋은 풀이 찾기** — 전역 최고를 두고 가지를 친다:

```
BACKTRACK(state, decisions, best):
    if is_solution(state):
        if objective(state) > best.value:
            best.value = objective(state)
            best.solution = copy(state)
        return

    for choice in candidates(state, decisions):
        if is_valid(state, choice):
            if bound(state, choice) <= best.value:
                continue        # 가지치기: 나아질 수 없다
            make_move(state, choice)
            BACKTRACK(state, decisions + 1, best)
            undo_move(state, choice)
```

가장 좋게 하기 변형은 그 아래 나무가 현재 최고 풀이를 넘을 수 있는지 미리 보는 **묶기 걸음**을 더한다. 이 묶기 걸음이 되짚기와 가지 뻗어 묶기를 잇는 다리이다.

## 시간 복잡도

$b$을 평균 갈래 수, $n$을 결정 수라 하자. 가지치기가 없으면 본이 상태 공간 나무의 모든 마디를 들른다:

$$
T(n) = O\!\left(\sum_{k=0}^{n} b^k\right) = O(b^n)
$$

가지치기가 실효 갈래 수를 줄인다. 될 수 있는지 살피기가 층마다 갈래의 $p$ 몫을 없애면 실효 갈래 수가 $b(1 - p)$으로 떨어지고 도는 시간은 $O\!\bigl((b(1-p))^n\bigr)$이 된다. 가지치기가 셀수록 실제로 도는 시간이 다항에 가까워지지만, NP 어려움 문제에서는 최악의 경우가 여전히 지수이다.

## 참고 문헌

- Skiena, *The Algorithm Design Manual*, 9장: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)

## 연습문제

**연습문제 1.**
되짚기 본의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    되짚기 본은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
되짚기 본의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
되짚기 본의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 되짚기 본을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$
