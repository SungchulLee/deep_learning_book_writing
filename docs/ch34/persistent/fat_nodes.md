# 살진 마디

자료 얼개를 영속하게 만들 때 가장 곧은 길은 모든 고침을 마디 안에 갈무리하는 것이다. 고칠 때마다 새 마디를 만드는 대신 마디마다 "살지게" 하여, 그 밭이 여태 지녔던 온 값의 때 도장 붙은 기록을 들고 다니게 한다. 드리스콜, 사르낙, 슬리터, 타잔(1989)이 내놓은 이 재주는 고침마다 $O(1)$ 나눠 갚는 덧자리로 부분 영속을 이룬다.

## 직관

고침이 어느 마디의 손가락질이나 자료 밭을 바꾸는 이음 얼개(이진 찾기 나무, 이음 목록 따위)를 여겨 보자. 마디를 베끼는 대신 고침 적바림 $(t, \text{밭}, \text{값})$을 마디에 덧붙인다. 여기서 $t$은 판 번호다. 판 $v$에서 어떤 밭을 읽으려면 때 도장이 많아야 $v$인 마지막 항목을 고침 목록에서 찾는다.

## 살진 마디 얼개

마디마다 다음을 갈무리한다.

- **처음 밭**: 판 0의 열쇠, 자료, 손가락질.
- **고침 목록**: 모든 바뀜을 적어 둔, 때 차례로 매긴 세 짝 $(t, \text{밭}, \text{값})$의 열.

손가락질 밭이 $p$개인 마디에서(보기로 왼쪽/오른쪽을 지닌 이진 찾기 나무는 $p = 2$이다) 고침 목록은 $p + d$개 밭($d$은 자료 밭 개수) 가운데 어느 것의 바뀜이든 갈무리한다.

## 읽기 연산

판 $v$에서 마디 $x$의 밭 $f$을 읽으려면:

1. 밭 $f$에 미치는 항목을 $x$의 고침 목록에서 훑는다.
2. 때 도장이 $\le v$인 마지막 항목의 값을 돌려준다.
3. 그런 항목이 없으면 판 0의 처음 값을 돌려준다.

매긴 고침 목록에 두 갈래 찾기를 쓰면 다음과 같다.

$$
T_{\text{읽기}} = O(\log m_x)
$$

여기서 $m_x$은 마디 $x$에 벌인 고침의 개수다.

## 적기 연산

지금 판 $t$에서 마디 $x$의 밭 $f$에 적으려면:

1. $x$의 고침 목록에 $(t, f, \text{새\_값})$을 덧붙인다.

$$
T_{\text{적기}} = O(1) \text{ 나눠 갚음}, \quad S_{\text{적기}} = O(1) \text{ 나눠 갚음}
$$

## 매인 살진 마디

종요로운 다듬기 하나는 마디마다 고침 목록을 붙박인 용량 $c$으로 매어 두는 것이다(손가락질 밭이 $p$개일 때 흔히 $c = 2p$이다). 목록이 꽉 차면:

1. 지금 밭 값을 모두 구워 넣은 새 마디 벌을 만든다.
2. 새 마디의 고침 목록을 비운다.
3. 어버이의 손가락질을 새 마디로 고친다(이것이 다시 줄줄이 번지는 베끼기를 부를 수 있다).

이 매인 갈래는 나눠 갚는 따짐으로 고침마다 $O(1)$ 자리를 보장한다. 마디마다 쪼개지기 앞서 고침 $c$개를 빨아들일 수 있으므로 쪼개기 비용을 그 $c$번의 적기에 나누어 물린다.

$$
S_{\text{적기마다 나눠 갚음}} = O(1)
$$

## 복잡도 간추림

| 연산 | 때 | 덧자리 |
|---|---|---|
| 읽기(밭, 판 $v$) | $O(\log m)$ | -- |
| 적기(밭, 값) | $O(1)$ 나눠 갚음 | $O(1)$ 나눠 갚음 |
| 매인 쪼개기 | 가장 나쁠 때 $O(p)$ | $O(p)$ |

여기서 $m$은 물음을 던진 마디에 벌인 고침의 개수이고 $p$은 마디마다의 손가락질 밭 개수다.

!!! tip "살진 마디와 길 베끼기"
    살진 마디는 길 베끼기보다 자리를 적게 쓰지만(고른 나무에서 고침마다 $O(\log n)$이 아니라 $O(1)$) 읽기가 더 느리다($O(\log n)$이 아니라 $O(\log m)$). 고침이 잦고 읽기가 드물면 살진 마디를, 읽기가 판치면 길 베끼기를 고르라.

## 구현

```python
"""
살진 마디 영속 이진 찾기 나무 -- 부분 영속.

이진 찾기 나무 마디마다 고침 기록을 갈무리한다. 지난 어느
판에서든 밭을 읽으면 기록을 훑고, 적으면 새 항목을 덧붙인다.
"""

from __future__ import annotations
from bisect import bisect_right


# === 살진 마디 ================================================================

class FatNode:
    """부분 영속을 위해 고침 내력을 지닌 이진 찾기 나무 마디."""

    def __init__(self, key: int, version: int = 0):
        self.key = key
        self._creation = version
        # 고침 기록: (판, 밭 이름, 값)의 목록
        self._mods: list[tuple[int, str, object]] = []
        # 처음 밭 값(판 0)
        self._left: FatNode | None = None
        self._right: FatNode | None = None

    def get_field(self, field: str, version: int) -> object:
        """*version* 시점의 *field*을 읽는다."""
        original = getattr(self, f"_{field}")
        best_val = original
        best_ver = self._creation
        for ver, fname, val in self._mods:
            if fname == field and best_ver < ver <= version:
                best_val = val
                best_ver = ver
        return best_val

    def set_field(self, field: str, value: object, version: int) -> None:
        """*version*에서 *field*에 대한 고침을 적어 둔다."""
        self._mods.append((version, field, value))


# === 영속 이진 찾기 나무 ======================================================

class PersistentBST:
    """살진 마디를 쓰는 부분 영속 이진 찾기 나무."""

    def __init__(self):
        self.roots: list[FatNode | None] = [None]  # roots[v] = 판 v의 뿌리
        self.current_version = 0

    def insert(self, key: int) -> int:
        """*key*을 넣고 새 판 번호를 돌려준다."""
        self.current_version += 1
        v = self.current_version
        if self.roots[-1] is None:
            new_root = FatNode(key, v)
            self.roots.append(new_root)
        else:
            self.roots.append(self.roots[-1])
            self._insert_at(self.roots[-1], key, v)
        return v

    def _insert_at(self, node: FatNode, key: int, version: int) -> None:
        """살진 마디 나무에 되돌아 들어가며 넣는다."""
        if key < node.key:
            left = node.get_field("left", version)
            if left is None:
                node.set_field("left", FatNode(key, version), version)
            else:
                self._insert_at(left, key, version)
        elif key > node.key:
            right = node.get_field("right", version)
            if right is None:
                node.set_field("right", FatNode(key, version), version)
            else:
                self._insert_at(right, key, version)

    def inorder(self, version: int | None = None) -> list[int]:
        """주어진 판에서 가운데 먼저 훑기."""
        if version is None:
            version = self.current_version
        result: list[int] = []
        self._inorder(self.roots[version], version, result)
        return result

    def _inorder(self, node: FatNode | None, version: int,
                 result: list[int]) -> None:
        if node is None or node._creation > version:
            return
        left = node.get_field("left", version)
        self._inorder(left, version, result)
        result.append(node.key)
        right = node.get_field("right", version)
        self._inorder(right, version, result)


# === 메인 =====================================================================

if __name__ == "__main__":
    bst = PersistentBST()
    for key in [5, 3, 7, 1, 4]:
        bst.insert(key)

    print(f"v0 (empty): {bst.inorder(0)}")
    print(f"v1 (add 5): {bst.inorder(1)}")
    print(f"v3 (add 7): {bst.inorder(3)}")
    print(f"v5 (add 4): {bst.inorder(5)}")
```

**출력:**

```
v0 (empty): []
v1 (add 5): [5]
v3 (add 7): [3, 5, 7]
v5 (add 4): [1, 3, 4, 5, 7]
```

판마다 그에 맞는 넣기를 마친 뒤의 나무 상태를 비추며, 살진 마디마다 갈무리한 고침 기록을 거쳐 앞선 판에도 그대로 닿을 수 있다.

## 참고 문헌

- Driscoll, J.R., Sarnak, N., Sleator, D.D., and Tarjan, R.E. "Making Data Structures Persistent." *JCSS*, 1989
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)

## 연습문제

**연습문제 1.**
부분 영속을 위한 살진 마디 재주를 풀어라. 판 $t$의 물음은 어떤 마디에서 옳은 밭 값을 어떻게 찾는가?

??? success "연습문제 1 풀이"
    살진 마디 재주에서는 마디마다 바뀔 수 있는 밭마다 (때 도장, 값) 짝의 목록을 갈무리한다. 판 $t$에서 밭을 고치면 그 밭의 기록에 짝 $(t, \text{새\_값})$을 덧붙인다. 처음 값은 때 도장 0(또는 만든 때)과 함께 갈무리된다. 판 $t$에서 밭을 물으려면 때 도장이 $\le t$인 가장 큰 것을 기록에서 두 갈래로 찾아 그에 맞는 값을 돌려준다. 이것이 판 $t$ 시점의 밭 값이다. (부분 영속에서는 고침이 판 차례대로 매겨지므로) 기록이 때 도장으로 매겨져 있어 두 갈래 찾기가 $O(\log m)$이 든다. 여기서 $m$은 그 밭에 벌인 고침의 개수다. 판 $t$에서 이진 찾기 나무를 찾을 때 마디마다 닿는 비용이 $O(1)$이 아니라 $O(\log m)$이므로 온 찾기 때는 $O(h \log m)$이 된다. $\square$

---

**연습문제 2.**
마디마다 들어오는 차수가 매인(그것을 가리키는 손가락질이 많아야 $p$개인) 자료 얼개에서 살진 마디가 고침마다 $O(1)$ 나눠 갚는 자리를 이룸을 증명하여라.

??? success "연습문제 2 풀이"
    DSST 틀은 마디마다 고침 자리 $2p$개를 더 준다(여기서 $p$은 들어오는 차수의 최댓값이다). 마디의 밭을 고치면 그 바뀜을 빈 자리 가운데 하나에 적는다. 마디의 자리가 꽉 차면 마디를 "베껴 내보낸다". 곧 마지막 값을 지닌 새 마디를 만들고 옛 마디를 가리키던 모든 손가락질을 새 마디를 가리키게 고친다. 이 손가락질 고침 자체가 가리키던 마디에 대한 고침이므로 되돌아 들어가며 다룬다. 나눠 갚는 따짐은 알맞은 상수 $c$에 대해 퍼텐셜 함수 $\Phi = c \cdot (\text{찬 자리의 온 개수})$을 쓴다. 고침마다 자리 하나를 채우고(비용 1, 퍼텐셜이 $c$만큼 는다) 베껴 내보내기는 자리 $2p$개를 비우지만(퍼텐셜이 $2pc$만큼 준다) 손가락질 $\le p$개를 고쳐야 한다(비용 $\le p$이고 저마다 자리 하나를 채운다). $c \ge 1$으로 고르면 고침마다 나눠 갚는 비용이 $O(1)$이 된다. $\square$

---

**연습문제 3.**
마디가 $n$개이고 온 고침이 $m$번인 이진 찾기 나무의 판 $t$을 읽을 때, 살진 마디와 길 베끼기의 물음 때 덧듦을 견주어라.

??? success "연습문제 3 풀이"
    **길 베끼기**: 판마다 제 뿌리 손가락질을 지닌다. 판 $t$의 물음은 판 $t$의 뿌리에서 나무를 훑으며 함께 쓰거나 베낀 마디로 이어지는 손가락질을 따라간다. 마디마다 닿는 비용이 $O(1)$이다(그저 손가락질을 따라간다). 온 물음 때는 나무 높이를 $h$이라 할 때 $O(h)$이다. **살진 마디**: 나무 얼개가 하나뿐이고 마디마다 고침 기록이 있다. 판 $t$의 물음은 들르는 마디마다 기록을 두 갈래로 찾아 옳은 밭 값을 알아내야 한다. 마디마다 닿는 비용이 $O(\log m_v)$이며 여기서 $m_v$은 그 마디의 고침 개수다. 온 물음 때는 (고침이 몰려 있으면) 가장 나쁠 때 $O(h \cdot \log m)$이다. 실제로는 $m_v$이 마디마다 다르고 거의 모든 마디의 고침이 적으므로 어림 비용은 $O(h)$에 더 가깝다. 길 베끼기는 자리를 더 쓰는 값으로 엄격히 더 좋은 물음 성능을 얻는다. $\square$

---

**연습문제 4.**
살진 마디 영속 이음 목록이 판 1000개를 지니고 판마다 고침이 하나씩 있다. 온 자리 쓰임은 얼마인가? 목록을 따로 1000벌 갈무리하는 것과 견주면 어떠한가?

??? success "연습문제 4 풀이"
    목록의 마디가 $n$개라 하자. 1000번의 고침마다 어느 마디의 기록에 (때 도장, 값) 짝 하나를 더한다. 온 자리는 처음 목록에 $O(n)$, 고침 항목에 $O(1000)$이므로 $O(n + 1000)$이다. $n = 10{,}000$이면 온 자리가 대략 $11{,}000$ 낱이다. 따로 1000벌을 갈무리하면 $1000 \times n = 10{,}000{,}000$ 낱이 들어 1000배로 는다. 자료 얼개 크기에 견주어 고침이 성길 때 살진 마디는 놀랍도록 자리를 아낀다. 맞바꿈은 마디에 닿을 때마다 이제 고침 기록을 찾아야 하므로 닿기마다 $O(\log 1000) \approx 10$의 덧듦이 붙는다는 것이다. $\square$

---

**연습문제 5.**
살진 마디가 완전 영속(마지막 판뿐 아니라 아무 판이나 고치기)을 받쳐 줄 수 있는가? 어떤 근본 어려움이 생기며 드리스콜 등은 이를 어떻게 다루었는가?

??? success "연습문제 5 풀이"
    앞서 밝힌 살진 마디는 부분 영속만 받쳐 준다. 고침이 때 도장 차례로 마지막 판에 매겨져야 한다. 완전 영속에서는 (마지막이 아닐 수도 있는) 판 $t$에 대한 고침이 갈래를 만든다. 어려움은 마디마다의 고침 기록이 더는 때로 곧게 매겨지지 않는다는 것이다. 판 내력이 나무를 이루므로 곧은 기록에 대한 두 갈래 찾기가 듣지 않는다. 드리스콜 등은 살진 마디 재주에 "마디 쪼개기" 꾀와 판 나무를 오가는 닿기 함수를 더해 이를 다루었다. 마디마다의 고침 기록을 판 나무 얼개에 따라 짜고, 판 $t$에서 밭을 찾으려면 옳은 갈래를 찾아야 한다. 나눠 갚는 자리 매임은 고침마다 $O(1)$이 여전히 성립하지만 상수 인자가 커진다. 판 유향 비순환 그래프를 오가는 품 때문에 물음 덧듦도 는다. 실제로는 만들기가 더 쉬워 완전 영속에는 길 베끼기를 즐겨 쓴다. $\square$
