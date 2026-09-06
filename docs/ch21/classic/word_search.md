# 낱말 찾기

글자 2차원 격자와 목표 낱말이 주어질 때 **낱말 찾기** 문제는 칸마다 많아야 한 번씩 쓰면서 (가로나 세로로) 이웃한 칸의 길을 따라 그 낱말을 만들 수 있는지 정한다. 격자 위 되짚기의 자연스러운 쓰임새이다. 곧 걸음마다 여러 이웃 칸으로 갈래가 뻗고, 글자가 맞지 않으면 곧바로 막다른 길을 쳐 낸다.

## 문제 서술

글자 $m \times n$ 격자 `board`와 글줄 `word`이 주어질 때 격자에 `word`이 있으면 `True`을 돌려준다. 옳은 길은 가로나 세로로 이웃한 칸을 잇고 한 길 안에서 칸을 다시 쓸 수 없다.

## 되짚기 전략

격자에서 `word`의 첫 글자와 맞는 칸 $(r, c)$마다:

1. $(r, c)$을 들른 것으로 표시한다.
2. 아직 들르지 않은 이웃 칸(위, 아래, 왼쪽, 오른쪽)으로 옮기며 남은 글자를 맞추어 본다.
3. 낱말 전체가 맞으면 `True`을 돌려준다.
4. 어떤 이웃 칸도 맞춤을 이어 가지 못하면 **되짚어** $(r, c)$의 표시를 지우고 `False`을 돌려준다.

걸음마다 지금 칸의 글자를 낱말의 다음 글자와 견주어 될 수 있는지 살피고, 맞지 않으면 곧바로 갈래를 쳐 낸다.

## 복잡도

| 갈래 | 값 |
|---|---|
| 시간(최악의 경우) | 낱말 길이가 $L$일 때 $O(m \cdot n \cdot 3^L)$ |
| 공간 | 되돌이 쌓임 $O(L)$ |
| 갈래 수 | 3(온 칸을 빼고) |

걸음마다 아직 들르지 않은 이웃이 많아야 3개(넷째는 온 칸)이므로 $3^L$이라는 인수가 나온다. 실전에서는 글자가 맞지 않아 갈래 대부분이 쳐진다.

## 파이썬 구현

```python
"""
낱말 찾기 — 2차원 격자 위의 되짚기.

칸을 다시 쓰지 않고 이웃한 칸의 길을 따라 글자 격자에서
목표 낱말을 찾을 수 있는지 정한다.
"""


# === 낱말 찾기 ===

def word_search(board: list[list[str]], word: str) -> bool:
    """격자에 낱말이 있으면 True을 돌려준다.

    인수:
        board: 글자 m x n 격자.
        word: 찾을 목표 낱말.
    """
    if not board or not word:
        return False

    m, n = len(board), len(board[0])

    def backtrack(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            return True

        if (r < 0 or r >= m or c < 0 or c >= n
                or board[r][c] != word[idx]):
            return False

        # 칸을 잠깐 고쳐 들른 것으로 표시한다
        original = board[r][c]
        board[r][c] = "#"

        # 네 방향을 모두 살핀다
        found = (
            backtrack(r + 1, c, idx + 1)
            or backtrack(r - 1, c, idx + 1)
            or backtrack(r, c + 1, idx + 1)
            or backtrack(r, c - 1, idx + 1)
        )

        # 되짚기: 칸을 되돌린다
        board[r][c] = original
        return found

    for r in range(m):
        for c in range(n):
            if board[r][c] == word[0] and backtrack(r, c, 0):
                return True

    return False


# === 나온 곳을 모두 찾기 ===

def find_word_paths(
    board: list[list[str]], word: str
) -> list[list[tuple[int, int]]]:
    """그 낱말을 이루는 서로 다른 길을 모두 찾는다."""
    if not board or not word:
        return []

    m, n = len(board), len(board[0])
    results: list[list[tuple[int, int]]] = []

    def backtrack(r: int, c: int, idx: int, path: list[tuple[int, int]]) -> None:
        if idx == len(word):
            results.append(path[:])
            return

        if (r < 0 or r >= m or c < 0 or c >= n
                or board[r][c] != word[idx]):
            return

        original = board[r][c]
        board[r][c] = "#"
        path.append((r, c))

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            backtrack(r + dr, c + dc, idx + 1, path)

        path.pop()
        board[r][c] = original

    for r in range(m):
        for c in range(n):
            if board[r][c] == word[0]:
                backtrack(r, c, 0, [])

    return results


# === 메인 ===

if __name__ == "__main__":
    board = [
        ["A", "B", "C", "E"],
        ["S", "F", "C", "S"],
        ["A", "D", "E", "E"],
    ]

    test_words = ["ABCCED", "SEE", "ABCB"]
    for w in test_words:
        # 시험마다 새 벌을 만든다
        b = [row[:] for row in board]
        print(f"'{w}': {word_search(b, w)}")
    # 내임:
    # 'ABCCED': True
    # 'SEE': True
    # 'ABCB': False
```

## 풀이 예제

다음 판에 대해:

```
A B C E
S F C S
A D E E
```

"ABCCED"을 찾으면:

1. $(0,0)$에서 시작한다. 'A'이 맞는다. 들른 것으로 표시한다.
2. 오른쪽 $(0,1)$으로 간다. 'B'이 맞는다.
3. 오른쪽 $(0,2)$으로 간다. 'C'이 맞는다.
4. 아래 $(1,2)$으로 간다. 'C'이 맞는다.
5. 아래 $(2,2)$으로 간다. 'E'이 맞는다.
6. 왼쪽 $(2,1)$으로 간다. 'D'이 맞는다. 낱말을 찾았다.

"ABCB"을 찾으면: 자리 $(0,0) \to (0,1) \to (0,2)$에서 "ABC"을 맞춘 뒤 다음 'B'은 이미 길 위에 있는 $(0,1)$을 다시 들러야 한다. 다른 길이 없으므로 찾기가 `False`을 돌려준다.

## 참고 문헌

- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.

## 연습문제

**연습문제 1.**
낱말 찾기의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    낱말 찾기은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
낱말 찾기의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
낱말 찾기의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 낱말 찾기을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$
