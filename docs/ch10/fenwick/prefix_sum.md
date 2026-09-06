# 접두사 합 질의

점 갱신을 받을 수 있는 배열 $a[1..n]$이 주어졌을 때 **접두사 합 질의**는 누적합 $\text{prefix}(i) = \sum_{j=1}^{i} a[j]$을 묻는다. 이진 색인 트리(BIT)에서 이 질의는 색인 $i$의 이진 표현을 이용해 $O(\log n)$ 시간에 돈다. 이 쪽은 접두사 질의 알고리즘을 자세히 좇으며 왜 노드를 많아야 $\lfloor \log_2 n \rfloor$개 들르는지 보인다.

## 알고리즘

BIT 항목 `tree[i]`마다 자리 $i$에서 끝나는 이어진 $\text{lowbit}(i) = i \;\&\; (-i)$개 원소의 합을 담음을 떠올리자. 접두사 질의는 겹치지 않는 이 덩어리를 모아 자리 $1$부터 $i$까지를 덮는다.

절차는 다음과 같다.

1. 쌓는 그릇 $s = 0$으로 시작한다.
2. `tree[i]`을 $s$에 더한다.
3. 가장 낮은 켜진 비트를 없앤다: $i \leftarrow i - \text{lowbit}(i)$.
4. $i = 0$이 될 때까지 되풀이한다.

단계마다 $i$에서 비트를 하나 떼므로 고리가 많아야 $\lfloor \log_2 n \rfloor$번 돈다.

## 한 걸음씩 따라가기

$n = 8$인 배열 $a = [1, 3, 5, 7, 9, 2, 4, 6]$을 생각하자. BIT는 다음을 담는다.

| 색인 $i$ | 이진 | $\text{lowbit}(i)$ | 덮는 범위 | `tree[i]` |
|:---------:|:------:|:------------------:|:-------------:|:---------:|
| 1 | `001` | 1 | $[1,1]$ | 1 |
| 2 | `010` | 2 | $[1,2]$ | 4 |
| 3 | `011` | 1 | $[3,3]$ | 5 |
| 4 | `100` | 4 | $[1,4]$ | 16 |
| 5 | `101` | 1 | $[5,5]$ | 9 |
| 6 | `110` | 2 | $[5,6]$ | 11 |
| 7 | `111` | 1 | $[7,7]$ | 4 |
| 8 | `1000` | 8 | $[1,8]$ | 37 |

**질의: prefix(7)**

$a[1] + a[2] + \cdots + a[7] = 1+3+5+7+9+2+4 = 31$을 얻고자 한다.

| 단계 | $i$ (이진) | `tree[i]` | 쌓은 값 $s$ |
|:----:|:------------:|:---------:|:---------------:|
| 1 | 7 (`111`) | 4 | 4 |
| 2 | 6 (`110`) | 11 | 15 |
| 3 | 4 (`100`) | 16 | 31 |
| 4 | 0 (`000`) | — | 끝 |

(7의 이진 표현에 켜진 비트가 셋이므로) 노드를 3개만 들르고 질의가 올바로 31을 돌려준다.

!!! note "단계 수는 켜진 비트의 수와 같다"
    접두사 질의는 $i$의 이진 표현에서 켜진 비트의 수만큼 꼭 그만큼의 BIT 항목을 들른다. $i \leq n$이므로 최악의 경우가 $\lfloor \log_2 n \rfloor$단계이다.

## 범위가 접두사를 꼭 맞게 덮는 까닭

색인 $i$의 BIT 노드마다 자리 $[i - \text{lowbit}(i) + 1, \; i]$을 덮는다. 가장 낮은 켜진 비트를 떼어 $i' = i - \text{lowbit}(i)$을 얻으면 다음 노드가 $i'$에서 끝나는 자리를 덮는다. 가장 낮은 비트를 떼는 것이 바로 앞 덩어리의 끝으로 "뛰는" 것이므로 범위가 이어지고 겹치지 않는다.

정확히 말해 $i = b_k b_{k-1} \cdots b_1$이 이진 표현이고 켜진 비트가 자리 $p_1 < p_2 < \cdots < p_m$에 있다고 하자. 그러면 질의가 접두사 $[1, i]$을 켜진 비트마다 하나씩, 꼭 $m$개의 덩어리로 나눈다.

## 구현

```python
"""
이진 색인 트리의 접두사 합 질의.

켜진 비트가 들르는 BIT 노드의 수를 어떻게 정하는지
한 걸음씩 자세히 따라가며 접두사 질의 알고리즘을
보인다.
"""


# === 질의를 따라가는 펜윅 트리 ===

class FenwickTree:
    """접두사 질의를 따라가는 방식을 선택으로 갖춘 BIT."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """자리 i에 delta를 더한다."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        """접두사 합 a[1] + a[2] + ... + a[i]을 돌려준다."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def query_traced(self, i: int) -> int:
        """단계마다 찍어 가며 따라가는 접두사 질의."""
        s = 0
        step = 1
        print(f"  Query prefix({i}):")
        while i > 0:
            print(f"    Step {step}: i={i} (bin={bin(i)}), "
                  f"tree[{i}]={self.tree[i]}, s={s}+{self.tree[i]}={s + self.tree[i]}")
            s += self.tree[i]
            i -= i & (-i)
            step += 1
        print(f"    Result: {s}")
        return s


# === 시연 ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 2, 4, 6]
    ft = FenwickTree(len(data))
    for i, v in enumerate(data, 1):
        ft.update(i, v)

    print(f"Array: {data}")
    print()

    # 따라가는 질의
    ft.query_traced(7)
    print()
    ft.query_traced(5)
    print()

    # 모든 접두사 합을 확인한다
    print("All prefix sums:")
    for i in range(1, len(data) + 1):
        expected = sum(data[:i])
        actual = ft.query(i)
        print(f"  prefix({i}) = {actual}  (expected {expected})  "
              f"{'OK' if actual == expected else 'MISMATCH'}")
```

**출력:**
```
Array: [1, 3, 5, 7, 9, 2, 4, 6]

  Query prefix(7):
    Step 1: i=7 (bin=0b111), tree[7]=4, s=0+4=4
    Step 2: i=6 (bin=0b110), tree[6]=11, s=4+11=15
    Step 3: i=4 (bin=0b100), tree[4]=16, s=15+16=31
    Result: 31

  Query prefix(5):
    Step 1: i=5 (bin=0b101), tree[5]=9, s=0+9=9
    Step 2: i=4 (bin=0b100), tree[4]=16, s=9+16=25
    Result: 25

All prefix sums:
  prefix(1) = 1  (expected 1)  OK
  prefix(2) = 4  (expected 4)  OK
  prefix(3) = 9  (expected 9)  OK
  prefix(4) = 16  (expected 16)  OK
  prefix(5) = 25  (expected 25)  OK
  prefix(6) = 27  (expected 27)  OK
  prefix(7) = 31  (expected 31)  OK
  prefix(8) = 37  (expected 37)  OK
```

## 복잡도

접두사 질의는 꼭 $\text{popcount}(i)$개, 곧 $i$에서 켜진 비트의 수만큼 노드를 들른다. $i \leq n$이므로 다음과 같다.

$$
\text{Time} = O(\log n) \qquad \text{Space} = O(1)
$$

무작위 정수는 평균적으로 비트의 절반쯤이 켜져 있으므로, 실제 평균 단계 수는 약 $\frac{1}{2} \log_2 n$이다.

## 참고 문헌

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.


## 연습문제

**연습문제 1.**
접두사 합 질의의 짜임을 설명하고 질의와 갱신의 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 짜임은 위계적 분해를 이용해 일차보다 빠른 질의 시간을 이룬다. 대개 질의와 갱신이 모두 $O(\log n)$이고 세우는 데 $O(n)$이나 $O(n\log n)$이 든다.

---

**연습문제 2.**
입력 $[3, 1, 4, 1, 5, 9, 2, 6]$으로 접두사 합 질의를 세워라. 마지막 짜임을 보여라.

??? success "연습문제 2 풀이"
    세우기 알고리즘을 적용하며 중간 상태를 보여라. 트리 짜임이면 트리를 그려라. 배열에 바탕한 짜임이면 부모-자식 관계를 덧붙여 배열의 내용을 보여라.

---

**연습문제 3.**
질의 연산이 올바른 어떤 질의 범위에 대해서도 옳은 결과를 돌려줌을 증명하라.

??? success "연습문제 3 풀이"
    증명에는 대개 트리의 높이에 대한 귀납법을 쓴다. 노드마다 질의가 한 부분 트리 안에 온전히 들거나(재귀한다) 두 부분 트리에 걸친다(부분 결과를 모은다). 모으는 함수(합, 최솟값, 최댓값)가 결합적이므로 올바로 모아진다. $\square$

---

**연습문제 4.**
접두사 합 질의가 질의마다의 계산을 $O(n)$에서 $O(\log n)$으로 빠르게 하는 딥러닝 응용을 설명하라.

??? success "연습문제 4 풀이"
    응용으로는 누적 어텐션 가중치를 위한 접두사 합 질의, 길이가 제각각인 수열에서 효율적인 풀링을 위한 범위 질의, 검색 증강 생성을 위한 최근접 이웃 찾기, 3차원 딥러닝의 점 구름 처리를 위한 공간 색인이 있다.