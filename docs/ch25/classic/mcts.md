# 몬테카를로 나무 찾기

놀이와 차례 있는 결정 문제에서 찾기 나무는 천문학처럼 커질 수 있다. 장기는 놀이 상태가 대략 $10^{120}$가지이다. **몬테카를로 나무 찾기(MCTS)**는 아무 흉내 내기로 마디마다 값을 어림하여 나무에서 가장 그럴듯한 곳에 셈 힘을 모아 이를 다룬다. MCTS은 알파고가 바둑 세계 챔피언을 이긴 역사적 승리를 뒷받침했고 요즘 놀이 인공 지능에서도 여전히 핵심이다.

## 핵심 생각

MCTS은 찾기 나무를 조금씩 늘려 세운다. 되풀이마다 뿌리에서 잎으로 내려간 뒤 앎을 도로 위로 퍼뜨리는 네 단계로 이루어진다.

1. **고르기** — 뿌리에서 시작해 나무 방침(예컨대 UCB1)에 따라 자식을 골라 내려가며, 아직 살피지 않은 자식이 있는 마디에 이를 때까지 간다.
2. **넓히기** — 아직 살피지 않은 움직임에 자식 마디를 하나(또는 여럿) 더한다.
3. **흉내 내기**(굴려 보기) — 새 마디에서 기본 방침(흔히 고르게 아무 수)으로 놀이를 끝까지 둔다.
4. **뒤먹임 퍼뜨리기** — 새 마디에서 뿌리까지의 길에 있는 모든 마디의 들른 수와 이긴 통계를 새로 고친다.

## UCB1: 살펴보기와 써먹기의 균형

**나무를 위한 웃 믿음 한계(UCT)**는 다음을 가장 크게 하는 자식을 고른다.

$$
\text{UCB1}(v) = \frac{w_v}{n_v} + c \sqrt{\frac{\ln N}{n_v}}
$$

여기서 각 기호는 다음과 같다.

- $w_v$ = 마디 $v$에서 이긴 횟수(또는 온 보상)
- $n_v$ = 마디 $v$에 들른 횟수
- $N$ = $v$의 어버이에 들른 횟수
- $c$ = 살펴보기 상수(이론상 흔히 $\sqrt{2}$)

첫 항은 **써먹기** 몫(평균 이긴 비율)이다. 둘째 항은 **살펴보기** 몫으로 덜 들른 마디를 좋아한다.

!!! note "이론상의 보장"
    나무에 쓴 UCB1은 되풀이 수가 늘수록 가장 좋은 움직임으로 모인다. UCB1의 뉘우침은 $O(\ln n / \Delta)$으로만 커지며 $\Delta$은 가장 좋은 팔과 둘째로 좋은 팔의 값 차이이다.

## 알고리즘 자세히

### 고르기

뿌리에서 UCB1 값이 가장 큰 자식을 거듭 골라, 아직 들르지 않은 자식이 있는 마디에 이를 때까지 간다.

### 넓히기

들르지 않은 자식 하나를 골라 나무에 더한다.

### 흉내 내기

넓힌 마디에서 아무 수(또는 어림잡기 방침)로 끝 상태에 이를 때까지 놀이를 둔다. 결과를 적어 둔다.

### 뒤먹임 퍼뜨리기

넓힌 마디에서 뿌리까지 거슬러 걷는다. 길 위의 마디마다:

- $n_v$을 1 늘린다.
- $w_v$을 흉내 내기 결과만큼 늘린다(이기면 1, 지면 0, 비기면 0.5).

넉넉히 되풀이한 뒤 들른 수가 가장 많은 뿌리의 자식을 가장 좋은 수로 고른다.

## 풀이 예제

X이 둘 수가 A와 B 둘인 오목 판을 보자.

MCTS을 100번 되풀이한 뒤:

- 수 A: $n = 60$, $w = 42$(이긴 비율 $= 0.70$)
- 수 B: $n = 40$, $w = 32$(이긴 비율 $= 0.80$)

$c = \sqrt{2}$이고 $N = 100$이면:

$$
\text{UCB1}(A) = 0.70 + \sqrt{2} \sqrt{\frac{\ln 100}{60}} = 0.70 + 0.393 = 1.093
$$

$$
\text{UCB1}(B) = 0.80 + \sqrt{2} \sqrt{\frac{\ln 100}{40}} = 0.80 + 0.481 = 1.281
$$

수 B의 UCB1이 더 크므로 다음 되풀이는 B의 아래 나무를 살핀다. A을 더 많이 들렀지만 B의 높은 이긴 비율과 살펴보기 덤이 MCTS을 그쪽으로 이끈다.

## 구현

```python
"""
단순한 두 사람 놀이를 위한 몬테카를로 나무 찾기.

Implements the four MCTS phases (selection, expansion, simulation,
backpropagation) with UCB1 as the tree policy.
"""

import math
import random
from collections import defaultdict


# === MCTS 마디 ===

class MCTSNode:
    """MCTS 나무의 마디."""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = None

    def ucb1(self, c=1.414):
        """이 마디의 UCB1 값을 셈한다."""
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


# === 단순한 놀이: 님 ===

class NimGame:
    """단순한 님 놀이: 두 사람이 돌무더기에서 돌을 1~3개 가져간다.

    마지막 돌을 가져가는 사람이 이긴다.
    """

    def __init__(self, stones=10):
        self.stones = stones
        self.current_player = 0  # 0 또는 1

    def clone(self):
        g = NimGame(self.stones)
        g.current_player = self.current_player
        return g

    def get_actions(self):
        return [i for i in range(1, min(4, self.stones + 1))]

    def apply(self, action):
        self.stones -= action
        self.current_player = 1 - self.current_player

    def is_terminal(self):
        return self.stones <= 0

    def winner(self):
        if self.stones <= 0:
            return 1 - self.current_player  # 마지막에 두는 사람이 이긴다
        return None


# === MCTS 알고리즘 ===

def mcts(game, iterations=1000, c=1.414):
    """MCTS을 돌려 가장 좋은 움직임을 돌려준다.

    인수:
        game: 지금 놀이 상태.
        iterations: MCTS 되풀이 횟수.
        c: UCB1의 살펴보기 상수.

    반환값:
        지금 상태에서 둘 가장 좋은 움직임.
    """
    root = MCTSNode(game.clone())
    root.untried_actions = game.get_actions()

    for _ in range(iterations):
        node = root
        state = game.clone()

        # 고르기
        while node.untried_actions is not None and \
              len(node.untried_actions) == 0 and \
              len(node.children) > 0:
            node = max(node.children, key=lambda n: n.ucb1(c))
            state.apply(node.action)

        # 넓히기
        if node.untried_actions and len(node.untried_actions) > 0:
            action = random.choice(node.untried_actions)
            state.apply(action)
            child = MCTSNode(state.clone(), parent=node, action=action)
            child.untried_actions = state.get_actions() if not state.is_terminal() else []
            node.untried_actions.remove(action)
            node.children.append(child)
            node = child

        # 흉내 내기(아무 굴려 보기)
        sim_state = state.clone()
        while not sim_state.is_terminal():
            actions = sim_state.get_actions()
            if not actions:
                break
            sim_state.apply(random.choice(actions))

        # 역전파
        winner = sim_state.winner()
        while node is not None:
            node.visits += 1
            if winner is not None:
                if winner == game.current_player:
                    node.wins += 1.0
                else:
                    node.wins += 0.0
            else:
                node.wins += 0.5
            node = node.parent

    # 가장 많이 들른 자식의 움직임을 돌려준다
    best_child = max(root.children, key=lambda n: n.visits)
    return best_child.action, best_child.visits, best_child.wins


# === 메인 ===

if __name__ == "__main__":
    random.seed(42)

    game = NimGame(stones=10)
    print(f"Nim with {game.stones} stones")

    action, visits, wins = mcts(game, iterations=5000)
    print(f"Best action: take {action} stone(s)")
    print(f"  Visits: {visits}, Win rate: {wins/visits:.3f}")

    # 뿌리의 모든 자식 통계를 보인다
    root = MCTSNode(game.clone())
    root.untried_actions = game.get_actions()
    for _ in range(5000):
        node = root
        state = game.clone()
        while node.untried_actions is not None and \
              len(node.untried_actions) == 0 and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            state.apply(node.action)
        if node.untried_actions and len(node.untried_actions) > 0:
            a = random.choice(node.untried_actions)
            state.apply(a)
            child = MCTSNode(state.clone(), parent=node, action=a)
            child.untried_actions = state.get_actions() if not state.is_terminal() else []
            node.untried_actions.remove(a)
            node.children.append(child)
            node = child
        sim = state.clone()
        while not sim.is_terminal():
            acts = sim.get_actions()
            if not acts:
                break
            sim.apply(random.choice(acts))
        w = sim.winner()
        while node is not None:
            node.visits += 1
            node.wins += 1.0 if w == game.current_player else 0.0
            node = node.parent

    print("\nAll moves from root:")
    for child in sorted(root.children, key=lambda n: -n.visits):
        wr = child.wins / child.visits if child.visits > 0 else 0
        print(f"  Take {child.action}: visits={child.visits}, "
              f"win_rate={wr:.3f}")
```

## 변형과 확장

| 변형 | 핵심 고침 |
|---|---|
| UCT | 나무에 쓴 UCB1(여느 MCTS) |
| RAVE | 빠른 움직임 값 어림 — 나무 전체의 수 통계를 쓴다 |
| PUCT | 미리 헤아리개 + UCT — 신경망 사전 분포를 쓴다(알파고/알파제로에서 쓴다) |
| 차츰 넓히기 | 움직임 공간이 클 때 가지를 제한한다 |

## 복잡도

- **되풀이마다 시간:** $O(d)$이며 $d$은 흉내 내기의 깊이이다.
- **온 시간:** 되풀이 $n$번에 $O(n \cdot d)$이다.
- **자리:** 만든 나무 마디에 $O(n)$이다.

MCTS은 **언제든 멈출 수 있는 알고리즘**이다. 곧 어느 때 멈추어도 그럴듯한 움직임을 돌려준다. 되풀이가 많을수록 결과가 좋아진다.

## 참고 문헌

- Kocsis, L. & Szepesvari, C. "Bandit Based Monte-Carlo Planning." *ECML*, 2006.
- Browne, C. et al. "A Survey of Monte Carlo Tree Search Methods." *IEEE Trans. CI and AI in Games*, 2012.

## 연습문제

**연습문제 1.**
몬테카를로 나무 찾기의 핵심 마구잡이 재주와 그것이 정해진 방식보다 나은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    몬테카를로 나무 찾기은 마구잡이를 써서 정해진 알고리즘이 마주칠 수 있는 가장 나쁜 들임을 피한다. 아무렇게나 고르므로 알고리즘의 솜씨가 들임의 짜임이 아니라 제 동전 던지기에 달린다. 그래서 모든 들임에 대해 참인 센 기댓값 시간이나 높은 확률의 보장을 흔히 얻으며, 짓궂거나 병리적인 경우를 걱정할 까닭이 없어진다. $\square$

---

**연습문제 2.**
몬테카를로 나무 찾기의 기댓값 시간 복잡도는 얼마인가? 가장 나쁜 경우의 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    기댓값 시간 복잡도는 흔히 $O(n)$이나 $O(n \log n)$이며 높은 확률로 이룬다. 가장 나쁜 경우는 다항식만큼 더 나쁠 수 있지만(예컨대 $O(n^2)$) 그럴 확률은 무시할 만큼 작다. 기댓값과 가장 나쁜 경우의 틈이 마구잡이의 값이며, 가장 나쁜 움직임이 일어날 확률은 들임 크기에 따라 지수로 줄어든다. $\square$

---

**연습문제 3.**
몬테카를로 나무 찾기은 라스베이거스 알고리즘인가 몬테카를로 알고리즘인가? 그 차이를 설명하라.

??? success "연습문제 3 풀이"
    **라스베이거스**: 늘 옳은 결과를 내며 도는 시간이 아무 변수이다(기댓값이 다항식). **몬테카를로**: 늘 다항식 시간에 돌지만 결과가 어떤 가둔 확률로 틀릴 수 있다. 몬테카를로 나무 찾기은 옳음을 보장하느냐 도는 시간을 보장하느냐에 따라 이 가운데 하나에 든다. 이 가름이 어긋날 확률을 어떻게 다룰지 정한다. $\square$

---

**연습문제 4.**
몬테카를로 나무 찾기에서 마구잡이를 없애거나 솜씨가 나쁠 확률을 줄이는 법을 설명하라.

??? success "연습문제 4 풀이"
    방책은 다음과 같다. (1) **거듭 해 보기**: 알고리즘을 여러 번 돌려 가장 좋거나 많은 쪽 결과를 택하면 어긋날 확률이 지수로 줄어든다. (2) **마구잡이 없애기**: 조건부 기댓값이나 흩는 함수 무리로 아무 고르기를 정해진 고르기로 바꾼다. (3) **키우기**: 몬테카를로 알고리즘에서는 $k$번 되풀이해 어긋남을 $2^{-k}$으로 줄인다. (4) **비슷 마구잡이 만들개**: 알고리즘이 보기에 "마구잡이처럼 보이는" 정해진 차례를 쓴다. $\square$
