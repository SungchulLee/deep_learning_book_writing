# 허프먼 부호

글을 보내거나 담을 때 어떤 글자가 다른 것보다 훨씬 자주 나온다면 붙박이 길이 부호(글자마다 8비트인 아스키 같은)를 쓰는 것은 공간을 낭비한다. **허프먼 부호**는 자주 나오는 글자에 짧은 비트열을, 드문 글자에 긴 비트열을 매겨 가장 좋은 **앞가지 없는** 길이 바뀜 부호를 낸다. 이 알고리즘은 욕심쟁이 틀의 아름다운 쓰임새이다. 곧 잦기가 가장 낮은 기호 둘을 거듭 어울려 아래에서 위로 부호 나무를 세운다.

---

## 1. 앞가지 없는 부호

**앞가지 없는 부호**(때로 "앞가지 부호"라고도 한다)는 어떤 부호말도 다른 것의 앞가지가 아닌 두 값 부호이다. 이 성질 덕분에 이어 붙인 비트열을 나눔표 없이 왼쪽에서 오른쪽으로 아리송함 없이 풀어낼 수 있다.

보기로 $\{0, 10, 110, 111\}$은 머리말 없는 부호이지만 $\{0, 01, 10, 1\}$은 아니다($0$이 $01$의 머리말이고 $1$이 $10$의 머리말이기 때문이다).

앞가지 없는 부호는 이진 나무와 하나씩 맞대응된다. 곧 잎마다 글자 하나를 나타내고, 뿌리에서 잎까지의 길(왼쪽 = 0, 오른쪽 = 1)이 부호말을 준다. 부호말의 길이 $d_i$은 나무에서 글자 $i$의 깊이와 같다.

---

## 2. 가장 좋게 하기 목표

잦기가 $f_1, f_2, \ldots, f_n$인 글자 $n$개가 주어졌을 때 머리말 없는 부호의 **값**은 글자마다 드는 비트 수의 바라는 값이다.

$$
B(T) = \sum_{i=1}^{n} f_i \cdot d_i
$$

여기서 $d_i$은 부호 나무 $T$에서 글자 $i$의 깊이다. 허프먼 부호는 $B(T)$을 가장 작게 하는 나무 $T^*$을 짓는다.

---

## 3. 알고리즘

!!! note "허프먼 알고리즘"

    1. 글자마다 그 잦기를 담은 잎 마디를 만든다. 모든 마디를 잦기를 열쇠로 하는 최소 우선순위 줄서기 $Q$에 넣는다.
    2. $|Q| > 1$인 동안:
        - 잦기가 가장 작은 마디 $x$과 $y$을 꺼낸다.
        - $f_z = f_x + f_y$인 새 속 마디 $z$을 만들고 $x$을 왼쪽 자식, $y$을 오른쪽 자식으로 둔다.
        - $z$을 $Q$에 넣는다.
    3. $Q$에 남은 마디가 허프먼 나무의 뿌리이다.

이 알고리즘이 욕심쟁이인 까닭은 걸음마다 그 자리에서 가장 좋은 고름, 곧 잦기가 가장 낮은 마디 둘을 어울리기를 하기 때문이다. 그래서 잦기가 가장 낮은 글자가 나무의 가장 깊은 곳(가장 긴 부호말)에 놓이고, 가장 잦은 글자는 뿌리 가까이(가장 짧은 부호말) 남는다.

---

## 4. 풀이 예제

**글자와 잦기:**

| 글자 | a  | b  | c  | d  | e   | f  |
|-----------|----|----|----|----|-----|----|
| 잦기 | 5  | 9  | 12 | 13 | 16  | 45 |

**한 걸음씩 세우기:**

1. $a$(5)과 $b$(9)을 **합친다** $\to$ 안쪽 마디(14).
2. $c$(12)과 $d$(13)을 **합친다** $\to$ 안쪽 마디(25).
3. (14)과 $e$(16)을 **합친다** $\to$ 안쪽 마디(30).
4. (25)과 (30)을 **합친다** $\to$ 안쪽 마디(55).
5. $f$(45)과 (55)을 **합친다** $\to$ 뿌리(100).

**나온 부호:**

| 글자 | 잦기 | 부호 | 깊이 |
|-----------|-----------|------|-------|
| f         | 45        | 0    | 1     |
| c         | 12        | 100  | 3     |
| d         | 13        | 101  | 3     |
| a         | 5         | 1100 | 4     |
| b         | 9         | 1101 | 4     |
| e         | 16        | 111  | 3     |

**값:**

$$
B(T) = 45 \cdot 1 + 12 \cdot 3 + 13 \cdot 3 + 5 \cdot 4 + 9 \cdot 4 + 16 \cdot 3 = 224
$$

붙박인 3비트 부호라면 $100 \times 3 = 300$비트가 든다. 허프먼 부호는 25.3%을 아낀다.

---

## 5. 파이썬 구현

```python
"""
허프먼 부호: 가장 좋은 앞가지 없는 두 값 부호를 세운다.

잦기가 가장 낮은 기호 둘을 되풀이해 합치는 욕심쟁이 전략으로
기대 부호 길이를 가장 작게 하는 두 갈래 나무를 세운다.
"""

import heapq
from collections import Counter

# === 허프먼 나무 마디 ===

class HuffmanNode:
    """허프먼 나무의 마디."""

    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

# === 허프먼 나무 세우기 ===

def build_huffman_tree(frequencies):
    """글자 잦기로 허프먼 나무를 세운다.

    인수:
        frequencies: 글자를 잦기에 대응시킨 사전

    반환값:
        허프먼 나무의 뿌리 마디
    """
    heap = [HuffmanNode(char=c, freq=f) for c, f in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right,
        )
        heapq.heappush(heap, merged)

    return heap[0]

# === 부호 뽑기 ===

def extract_codes(node, prefix="", codes=None):
    """허프먼 나무에서 두 값 부호를 뽑는다.

    인수:
        node: 나무에서 지금 마디
        prefix: 여태 세운 두 값 글줄
        codes: 글자 -> 부호 대응을 쌓는 사전

    반환값:
        글자를 허프먼 부호에 대응시킨 사전
    """
    if codes is None:
        codes = {}

    if node.char is not None:
        codes[node.char] = prefix if prefix else "0"
    else:
        if node.left:
            extract_codes(node.left, prefix + "0", codes)
        if node.right:
            extract_codes(node.right, prefix + "1", codes)

    return codes

# === 값 셈하기 ===

def huffman_cost(codes, frequencies):
    """허프먼 부호의 무게 붙은 전체 길 길이를 셈한다.

    인수:
        codes: 글자를 두 값 부호 글줄에 대응시킨 사전
        frequencies: 글자를 잦기에 대응시킨 사전

    반환값:
        전체 값 sum(f_i * d_i)
    """
    return sum(frequencies[c] * len(code) for c, code in codes.items())

if __name__ == "__main__":
    frequencies = {"a": 5, "b": 9, "c": 12, "d": 13, "e": 16, "f": 45}

    tree = build_huffman_tree(frequencies)
    codes = extract_codes(tree)

    print("Huffman Codes:")
    print(f"{'Char':>5} {'Freq':>5} {'Code':>6} {'Depth':>6}")
    print("-" * 24)
    for char in sorted(codes, key=lambda c: len(codes[c])):
        print(f"{char:>5} {frequencies[char]:>5} {codes[char]:>6} {len(codes[char]):>6}")

    cost = huffman_cost(codes, frequencies)
    total_freq = sum(frequencies.values())
    fixed_cost = total_freq * 3  # 글자 6개에는 적어도 3비트가 필요하다

    print(f"\nTotal cost: {cost}")
    print(f"Fixed 3-bit cost: {fixed_cost}")
    print(f"Savings: {(1 - cost / fixed_cost) * 100:.1f}%")
```

**출력:**
```
Huffman Codes:
 Char  Freq   Code  Depth
------------------------
    f    45      0      1
    c    12    100      3
    d    13    101      3
    e    16    111      3
    a     5   1100      4
    b     9   1101      4

Total cost: 224
Fixed 3-bit cost: 300
Savings: 25.3%
```

---

## 6. 복잡도 분석

$n$을 서로 다른 글자의 개수라 하자.

- **처음 힙 세우기:** $O(n)$.
- **큰 되돌이:** $n - 1$ 번 돌며 그때마다 원소 둘을 꺼내고 하나를 넣는다: $O(n \log n)$.
- **부호 뽑아내기:** $O(n)$(모든 잎을 한 번 돌아본다).
- **Total:** $O(n \log n)$.

**공간:** 나무와 부호에 $O(n)$.

---

## 7. 허프먼 부호의 성질

1. **가장 좋음.** 머리말 없는 부호를 모두 통틀어 허프먼 부호가 $\sum f_i d_i$을 가장 작게 한다(허프먼 가장 좋음 증명 쪽에서 밝힌다).
2. **꽉 찬 이진 나무.** 속 마디마다 자식이 정확히 둘이다. 자식이 하나뿐인 마디가 있다면 그 자식의 부호말을 줄일 수 있다.
3. **잦기가 가장 낮은 글자는 가장 깊은 곳의 형제이다.** 욕심쟁이 세우기에서 곧바로 따라 나온다.

---

## 연습문제

**연습문제 1.**
허프먼 부호에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Huffman Coding에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
허프먼 부호이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Huffman Coding에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
허프먼 부호의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(허프먼 부호에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$

## 정리하며

이 마당은 앞가지 없는 부호、가장 좋게 하기 목표、알고리즘、풀이 예제을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16.3절. MIT Press.
- Huffman, D. A. (1952). A method for the construction of minimum-redundancy codes. *Proceedings of the IRE*, 40(9), 1098--1101.
