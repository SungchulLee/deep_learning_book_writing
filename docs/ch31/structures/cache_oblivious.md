# 두름을 모르는 B 나무

여느 B 나무는 가장 좋은 들고남 복잡도를 이루지만 갈래 수를 밑바탕 기계의 덩이 크기 $B$에 맞춰야 한다. $B$이 다른 기계에서 돌리려면 나무를 다시 세워야 한다. **두름을 모르는** 알고리즘은 이 매임을 없앤다. 두 잡을 모른 채 *모든* $B$과 $M$ 값에서 한꺼번에 가장 좋은 들고남 복잡도를 이룬다. 이 옮겨 쓸 수 있음이 두름을 모르는 길의 한가운데 매력이다.

---

## 1. 이상적인 두름 모형

두름을 모르는 살피기는 바깥 기억 모형에 여김 둘을 더한 **이상적인 두름 모형**을 쓴다:

1. **가장 좋은 바꿔 넣기:** 두름은 앞으로 가장 늦게 닿을 덩이를 내보낸다(오프라인 최적인 벨레이디 알고리즘과 같다).
2. **온전한 짝지음:** 어떤 원반 덩이든 어떤 두름 줄에도 담을 수 있다.

이 여김은 실제로는 현실과 다르지만, 핵심 정리가 이상적인 두름 모형에서 살핀 어떤 알고리즘도 $M \ge 2B$(**높은 두름 여김**)이면 LRU 바꿔 넣기를 쓰는 실제 두름에서 (상수배 안에서) 같은 점근 들고남 복잡도를 이룸을 보장한다.

---

## 2. 판 엠데 보아스 놓기

두름을 모르는 찾기 나무의 핵심 재주는 **판 엠데 보아스(vEB) 놓기**다. 붙박인 두 갈래 나무를 기억에 담되 어떤 크기의 부분 나무든 이어져 놓이게 한다.

높이 $h$의 완전 두 갈래 나무가 주어질 때 vEB 놓기는 되돌이로 돈다:

1. 나무를 높이 $h/2$에서 갈라 높이 $h/2$의 **위 부분 나무** 하나와 저마다 높이 $h/2$인 **아래 부분 나무** $\Theta(2^{h/2})$개로 만든다.
2. 위 부분 나무를 이어 담고 이어서 아래 부분 나무를 저마다 이어 담는다.
3. 부분 나무마다 같은 놓기를 되돌이로 쓴다.

이 되돌이 가르기 덕에 어떤 덩이 크기 $B$에서도 길이 $\log_2 N$의 찾기 길이 덩이 $O(\log_B N)$개만 지난다.

---

## 3. 찾기 복잡도

vEB으로 놓은 나무의 찾기는 마디 $\log_2 N$개의 뿌리-잎 길을 따른다. vEB 놓기는 그 길이 많아야 다음만큼 덩이 경계를 넘음을 보장한다:

$$
O(\log_B N)
$$

높이 $\frac{1}{2}\log_2 B$의 이어진 부분 나무마다 덩이 상수 개 안에 들어가기 때문이다. 이는 $B$을 모르고도 B 나무의 찾기 가둠 $O(\log_B N)$과 맞는다.

---

## 4. 붙박인 두름을 모르는 B 나무

붙박인 두름을 모르는 찾기 나무는 vEB 놓기에 찾기 셈속을 더한다:

1. 줄 세운 열쇠 $N$개를 완전 두 갈래 찾기 나무에 담는다.
2. vEB 놓기로 나무를 기억에 놓는다.
3. 뿌리에서 잎까지 여느 두 갈래 찾기 나무 길을 따라 찾는다.

| 연산 | 들고남 복잡도 |
|---|---|
| 찾기 | $O(\log_B N)$ |
| 세우기 | $O(N/B)$(줄 세운 자료를 훑기만 한다) |

찾기 가둠이 가장 좋은 값으로 B 나무와 맞으면서도 알고리즘은 $B$이나 $M$을 결코 가리키지 않는다.

---

## 5. 바뀌는 두름을 모르는 B 나무

넣기와 지우기를 두름을 모른 채 받쳐 주기는 더 어렵다. **채운 기억 배열** 재주는 다스려진 틈을 둔 줄 세운 배열에 원소를 지녀 vEB 놓기를 지키면서 넣기와 지우기를 할 수 있게 한다:

1. 빽빽함 불변량을 지킨다. 배열의 토막마다 25%에서 75% 사이로 찬다.
2. 토막이 너무 차거나 너무 성기면 빽빽함 가둠을 어기는 가장 작은 감싸는 토막을 다시 고르게 한다.
3. 배열 위의 나무 색인은 vEB 놓기를 쓴다.

이는 다음을 이룬다:

| 연산 | 고루 나눈 들고남 |
|---|---|
| 찾기 | $O(\log_B N)$ |
| 넣기 | $O\!\left(\frac{\log^2 N}{B}\right)$ |
| 지우기 | $O\!\left(\frac{\log^2 N}{B}\right)$ |

넣기와 지우기 가둠에 $\log^2 N / B$ 항이 있어 B 나무의 $O(\log_B N)$보다 조금 나쁘지만 $B$이 크면 여전히 효율 좋다.

---

## 6. 보기: 판 엠데 보아스 놓기

```python
"""
두름을 모르는 찾기를 위한 판 엠데 보아스 나무 놓기.

되돌이 vEB 놓기가 두 갈래 찾기 나무를 담아 어떤 덩이 크기 B에서도
찾기가 덩이 경계를 O(log_B N)번만 넘게 하는 모습을 보여 준다.
"""

import math

# ===================================================================
# 판 엠데 보아스 놓기
# ===================================================================

def veb_layout(keys: list[int]) -> list[int]:
    """
    줄 세운 열쇠를 판 엠데 보아스 기억 놓기로 늘어놓는다.

    완전 두 갈래 찾기 나무를 세우고 되돌이로 놓아 어느 잣수의
    부분 나무든 이어져 담기게 한다.

    매개변수
    ----------
    keys : 줄 세운 열쇠 목록(길이는 2^h - 1이어야 한다).

    반환값
    -------
    vEB 놓기 차례의 열쇠 목록.
    """
    n = len(keys)
    if n <= 1:
        return keys[:]

    # 먼저 두 갈래 찾기 나무를 켜 차례로 세운다
    bst = [0] * (n + 1)  # 1부터 센다
    _fill_bst(keys, bst, 1, 0, n - 1)

    # vEB 놓기를 쓴다
    result = []
    _veb_recurse(bst, 1, int(math.log2(n + 1)), result)
    return result

def _fill_bst(keys, bst, node, lo, hi):
    """줄 세운 열쇠로 1에서 시작하는 두 갈래 찾기 나무 배열을 채운다."""
    if lo > hi or node >= len(bst):
        return
    mid = (lo + hi) // 2
    bst[node] = keys[mid]
    _fill_bst(keys, bst, 2 * node, lo, mid - 1)
    _fill_bst(keys, bst, 2 * node + 1, mid + 1, hi)

def _veb_recurse(bst, root, height, result):
    """되돌이로 vEB 놓기 차례를 낸다."""
    if height <= 0 or root >= len(bst):
        return
    if height == 1:
        result.append(bst[root])
        return

    top_h = height // 2
    bottom_h = height - top_h

    # 위 부분 나무를 모은다
    _collect_top(bst, root, top_h, result)

    # 아래 부분 나무를 모은다
    bottom_roots = []
    _find_bottom_roots(root, top_h, bottom_roots)
    for br in bottom_roots:
        if br < len(bst):
            _veb_recurse(bst, br, bottom_h, result)

def _collect_top(bst, root, height, result):
    """위 부분 나무의 마디를 모은다(너비 우선처럼)."""
    if height <= 0 or root >= len(bst):
        return
    result.append(bst[root])
    if height > 1:
        _collect_top(bst, 2 * root, height - 1, result)
        _collect_top(bst, 2 * root + 1, height - 1, result)

def _find_bottom_roots(root, top_height, roots):
    """아래 부분 나무의 뿌리 번호를 찾는다."""
    if top_height <= 0:
        roots.append(root)
        return
    _find_bottom_roots(2 * root, top_height - 1, roots)
    _find_bottom_roots(2 * root + 1, top_height - 1, roots)

def count_block_crossings(layout: list[int], block_size: int,
                          search_path: list[int]) -> int:
    """찾기 길이 덩이 경계를 몇 번 넘는지 센다."""
    positions = {val: i for i, val in enumerate(layout)}
    blocks_visited = set()
    for node in search_path:
        if node in positions:
            blocks_visited.add(positions[node] // block_size)
    return len(blocks_visited)

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    # 마디 2^h - 1개의 완전 두 갈래 찾기 나무를 세운다
    h = 4
    n = 2**h - 1
    keys = list(range(1, n + 1))

    layout = veb_layout(keys)
    print(f"Sorted keys: {keys}")
    print(f"vEB layout:  {layout}")
    print()

    # 덩이 크기마다 넘는 경계 수를 보인다
    # 두 갈래 찾기 나무에서 열쇠 5의 찾기 길: 뿌리 -> 왼쪽/오른쪽 ...
    search_path = [8, 4, 2, 1]  # 두 갈래 찾기 나무의 보기 길
    for B in [2, 4, 8]:
        crossings = count_block_crossings(layout, B, search_path)
        theoretical = math.ceil(math.log(n) / math.log(max(2, B)))
        print(f"B={B}: blocks visited = {crossings}, "
              f"O(log_B N) = {theoretical}")
```

??? example "보기 내놓기"

    ```
    Sorted keys: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    vEB layout:  [8, 4, 12, 2, 6, 1, 3, 5, 7, 10, 14, 9, 11, 13, 15]

    B=2: blocks visited = 3, O(log_B N) = 4
    B=4: blocks visited = 2, O(log_B N) = 2
    B=8: blocks visited = 1, O(log_B N) = 2
    ```

    vEB 놓기는 덩이 크기가 클수록 옮김마다 찾기 길을 더 많이 담게 하여 어떤 $B$에서도 가둠 $O(\log_B N)$과 맞는다.

---

## 7. 두름을 모르는 것과 두름을 아는 것

| 성질 | B 나무(두름을 앎) | 두름을 모르는 B 나무 |
|---|---|---|
| $B$과 $M$을 아는가 | 예 | 아니오 |
| 찾기 들고남 | $O(\log_B N)$ | $O(\log_B N)$ |
| 넣기 들고남 | $O(\log_B N)$ | 고루 나누어 $O(\log^2 N / B)$ |
| 옮겨 쓰기 | 한 기계에 맞춰짐 | 어느 기계에서도 가장 좋음 |
| 짜기 | 곧바름 | 더 복잡함 |

두름을 모르는 길은 고침 비용에서 작은 상수배를 내주고, 다시 손보지 않아도 어떤 기계에서든 자료 얼개가 잘 돈다는 보장을 얻는다.

---

## 연습문제

**연습문제 1.**
두름을 모르는 알고리즘을 뜻매김하고 두름을 아는 알고리즘보다 나은 점을 밝혀라.

??? success "연습문제 1 풀이"
    두름을 모르는 알고리즘은 덩이 크기 $B$이나 기억 크기 $M$을 모른 채 가장 좋은 들고남 복잡도를 이룬다. 이상적인 두름 모형(가장 좋은 바꿔 넣기, $B$과 $M$을 모름)에서 설계하고 살피지만 기억 계층의 모든 켜에서 한꺼번에 잘 돈다. 이점: 잡을 손보지 않고도 알고리즘 하나가 L1-L2, L2-L3, 램-원반 계층 모두에서 효율 좋다. (바깥 기억 합침 정렬 같은) 두름을 아는 알고리즘은 $B$과 $M$을 들임으로 요구한다.

---

**연습문제 2.**
두름을 모르는 찾기를 위한 판 엠데 보아스 놓기를 밝혀라.

??? success "연습문제 2 풀이"
    판 엠데 보아스(vEB) 놓기는 완전 두 갈래 나무를 가운데 켜에서 되돌이로 갈라 담는다. 위 반과 아래 부분 나무마다 기억에 이어 담기를 되돌이로 한다. 이로써 어떤 덩이 크기 $B$에서도 뿌리에서 잎까지 밟을 때 덩이 $O(\log_B N)$개에만 닿는다. 놓기의 되돌이 켜마다 부분 나무가 덩이 $O(1)$개에 들어가기 때문이다. 찾기는 들고남 $O(\log_B N)$번을 이루어 $B$을 모르고도 B 나무와 맞선다.

---

**연습문제 3.**
두름을 모르는 행렬 곱하기(나누어 정복하기 길)를 밝혀라.

??? success "연습문제 3 풀이"
    행렬마다 사분면 넷으로 나누고 되돌이로 곱한다. $C_{11} = A_{11}B_{11} + A_{12}B_{21}$ 등이다($n/2 \times n/2$ 행렬에 대한 되돌이 부름 8번). 들고남 복잡도는 $n^2 > M$(행렬이 기억에 안 들어감)일 때 $T(n) = 8T(n/2) + O(n^2/B)$이고 $n^2 \leq M$(다 들어감)일 때 $T(n) = O(n^2/B)$이다. 풀이: $T(n) = O(n^3/(B\sqrt{M}))$으로 $B$이나 $M$을 모르고도 가장 좋은 두름을 아는 타일 알고리즘과 맞는다.

---

**연습문제 4.**
두름을 모르는 원칙이 서로 다른 기계에서 하는 깊은 배움에 왜 중요한가?

??? success "연습문제 4 풀이"
    깊은 배움은 여러 기계에서 돈다. 두름 크기가 제각각인 CPU, 공유 기억 크기가 다른 GPU, 기억 계층이 다른 TPU, 가장자리 장치가 그것이다. 두름을 모르는 알고리즘은 장치마다 손보지 않고도 그 기계의 기억 계층에 저절로 맞춰 간다. 쓰임새: (1) GPU 세대를 가로질러 도는 타일 행렬 곱하기, (2) 효율 좋은 텐서 닿기를 위한 되돌이 자료 놓기, (3) 기억이 몹시 빠듯한 가장자리 헤아림의 흐름 알고리즘. 기계가 다양해질수록 '알 수 없는 기계 잡을 여기고 설계하라'는 원칙이 더 중요해진다.

## 정리하며

이 마당은 이상적인 두름 모형、판 엠데 보아스 놓기、찾기 복잡도、붙박인 두름을 모르는 B 나무을 차례로 짚었다.

**참고 문헌**

- Frigo, M. et al. "Cache-Oblivious Algorithms," *FOCS*, 1999.
- Bender, M. et al. "Cache-Oblivious B-Trees," *SIAM Journal on Computing*, 35(2), 2005.
- Prokop, H. "Cache-Oblivious Algorithms," 석사 논문, MIT, 1999.
