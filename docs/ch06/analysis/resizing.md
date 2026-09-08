# 크기 조정과 재해싱

칸의 수가 고정된 해시 테이블은 딜레마에 놓인다. 칸을 너무 적게 잡으면 사슬이 길어져 연산이 느려지고, 너무 많이 잡으면 메모리를 낭비한다. **동적 크기 조정**은 저장된 원소의 수가 변할 때 테이블의 크기를 바꾸어 이를 푼다. 적재율 $\alpha = n/m$이 위 문턱값을 넘으면 테이블이 커지고, 아래 문턱값 밑으로 떨어지면 줄어든다. 크기를 조정할 때마다 **재해싱**, 곧 새 크기에 맞추어 저장된 모든 원소의 해시를 다시 계산하는 일이 필요하다.

---

## 1. 재해싱이 필요한 까닭

테이블의 크기가 $m$에서 $m'$으로 바뀌면 해시 함수도 $h(k) = f(k) \bmod m$에서 $h'(k) = f(k) \bmod m'$으로 바뀐다. 대부분의 키에서 $h(k) \neq h'(k)$이므로 모든 원소를 새 테이블에 다시 넣어야 한다. 같은 칸 인덱스로 그냥 복사하면 원소가 엉뚱한 자리에 놓인다.

??? example "재해싱이 칸 배정을 바꾼다"

    $h(k) = k \bmod m$을 쓰며 키 $\{3, 7, 11, 14\}$을 담은 $m = 4$짜리 테이블을 생각해 보자.

    $$
    \begin{array}{rcl}
    h(3) = 3, \quad h(7) = 3, \quad h(11) = 3, \quad h(14) = 2
    \end{array}
    $$

    $m' = 8$으로 두 배 늘리면 다음과 같다.

    $$
    \begin{array}{rcl}
    h'(3) = 3, \quad h'(7) = 7, \quad h'(11) = 3, \quad h'(14) = 6
    \end{array}
    $$

    키 7과 14가 다른 칸으로 옮겨 간다. 재해싱하지 않으면 이 키를 찾을 때 엉뚱한 칸을 뒤져 실패한다.

---

## 2. 크기 조정 알고리즘

크기 조정은 세 단계로 이루어진다.

1. 빈 칸으로 이루어진 크기 $m'$의 새 테이블을 **할당한다**.
2. 옛 테이블의 모든 원소를 **재해싱한다**. $h'(k) = f(k) \bmod m'$을 계산하여 $(k, v)$을 새 테이블에 넣는다.
3. 옛 테이블을 **해제한다**.

크기 조정의 비용은 $\Theta(n + m')$이다. 원소 $n$개를 재해싱하는 데 $\Theta(n)$, 새 테이블을 초기화하는 데 $\Theta(m')$이 든다. $m' = \Theta(n)$이면 전체 비용은 $\Theta(n)$이다.

---

## 3. 늘리기 정책

늘리기 정책은 현재 테이블이 너무 찼을 때 새 크기 $m'$을 정한다. 가장 흔한 정책은 다음과 같다.

**테이블 두 배 늘리기** ($m' = 2m$). 표준적인 선택이다. 두 배로 늘리면 적재율이 절반으로 떨어진다.

$$
\alpha' = \frac{n}{2m} = \frac{\alpha}{2}
$$

위 문턱값이 $\alpha_{\max} = 0.75$이면 두 배 늘린 뒤 $\alpha' \approx 0.375$이 되어 다음 크기 조정까지 많은 삽입을 감당할 여유가 생긴다.

**정확히 늘리기** ($m' = \lceil n / \alpha_{\text{target}} \rceil$). 목표 적재율에 맞추어 테이블의 크기를 잡는다. 두 배 늘리기보다 메모리를 덜 낭비하지만 크기 조정이 더 잦아질 수 있다.

**인수 $c$만큼 늘리기** ($m' = \lceil c \cdot m \rceil$, $c > 1$). 두 배 늘리기의 일반화로, 크기 조정의 잦음과 메모리 부담의 균형을 맞추도록 $c$을 고를 수 있다. $c$이 작으면(예: $c = 1.5$) 메모리를 덜 낭비하지만 크기 조정이 더 잦다.

늘리기 인수의 선택이 상각 비용에 영향을 준다. 두 배 늘리기($c = 2$)는 삽입당 상각 $O(1)$을 이룬다. 상수 인수 $c > 1$이면 어느 것이든 상각 $O(1)$을 이루지만 상수가 달라진다.

**인수 $c$일 때의 상각 비용.** 크기 1에서 시작하여 $n$번 삽입한 뒤 모든 크기 조정의 전체 비용은 다음과 같다.

$$
T_{\text{resize}} = \sum_{i=0}^{\lfloor \log_c n \rfloor} c^i = \frac{c^{\lfloor \log_c n \rfloor + 1} - 1}{c - 1} \leq \frac{cn}{c - 1}
$$

따라서 삽입당 상각 비용은 다음과 같다.

$$
\hat{c} = 1 + \frac{c}{c - 1}
$$

$c = 2$이면 $\hat{c} = 1 + 2 = 3$이고 $c = 1.5$이면 $\hat{c} = 1 + 3 = 4$이다. 늘리기 인수가 작으면 상각 한계의 상수가 커지지만 메모리의 정점 사용량은 줄어든다.

---

## 4. 줄이기 정책

원소를 지우면 적재율이 떨어지고 테이블이 메모리를 낭비한다. 테이블을 줄이면 그 메모리를 되찾는다.

**테이블 절반으로 줄이기** ($\alpha < \alpha_{\min}$일 때 $m' = m/2$). 줄이기 문턱값 $\alpha_{\min}$은 **스래싱**을 막으려면 $\alpha_{\max} / 2$보다 엄격히 작아야 한다. 스래싱은 경계 근처에서 삽입과 삭제를 번갈아 하며 크기 조정이 되풀이되는 병적인 양상이다.

!!! warning "스래싱"

    늘리기 문턱값이 $\alpha_{\max} = 1$이고 줄이기 문턱값이 $\alpha_{\min} = 1/2$이면, 가득 찬 테이블에서 삽입 한 번(두 배 늘리기)과 삭제 한 번(절반 줄이기)을 번갈아 할 때 연산마다 $\Theta(n)$을 치른다. $\alpha_{\min} = 1/4$으로 두면 이를 막을 수 있다. 절반으로 줄인 뒤에는 다음 줄이기까지 삭제가 많이 필요하고, 두 배로 늘린 뒤에는 다음 늘리기까지 삽입이 많이 필요하기 때문이다.

**표준적인 비대칭 문턱값.** $\alpha_{\max} = 1$, $\alpha_{\min} = 1/4$의 조합은 적재율이 $[1/4, 1]$ 안에 머물게 하고, 삽입과 삭제가 뒤섞인 어떤 열에 대해서도 연산당 상각 비용이 $O(1)$임을 보장한다.

---

## 5. 점진적 재해싱

지연에 민감한 응용에서는 크기 조정 한 번의 $\Theta(n)$ 비용이 받아들일 수 없는 멈춤을 일으킬 수 있다. **점진적 재해싱**은 그 비용을 여러 연산에 나누어 편다.

1. 새 표를 잡되 옛 표도 그대로 살려 둔다.
2. 그 뒤 삽입이나 찾기를 할 때마다 옛 표의 원소를 정해진 개수(예를 들어 2개나 4개)만큼 새 표로 다시 해싱한다.
3. 옮기기가 끝날 때까지 찾기는 두 표를 모두 살핀다.
4. 모든 원소를 옮기고 나면 옛 표를 놓아준다.

이 방법은 옮기는 동안 잠시 $O(n)$의 메모리를 더 쓰는 대신 연산마다의 최악 비용을 $O(1)$으로 묶는다(분할 상환 비용은 그대로이다).

Redis는 해시 표 구현에 점진적 재해싱을 써서 크기를 크게 바꿀 때 사건 반복문이 멈추지 않게 한다.

---

## 6. 구현

```python
"""
자동으로 크기를 바꾸고 다시 해싱하는 동적 해시 표.

적재율이 높으면 표를 두 배로, 낮으면 절반으로 줄이는 방식을
보이며, 문턱값을 비대칭으로 두어 요동을
막는다.
"""

# === 동적 해시 표 ===

class DynamicHashTable:
    """성능을 지키려고 스스로 크기를 바꾸는 해시 표."""

    GROW_THRESHOLD = 0.75
    SHRINK_THRESHOLD = 0.25
    MIN_SIZE = 4

    def __init__(self, size: int = 4):
        self.size = size
        self.table: list[list[tuple]] = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key: int) -> int:
        return key % self.size

    def _resize(self, new_size: int) -> None:
        """표의 크기를 바꾸고 모든 원소를 다시 해싱한다."""
        old_table = self.table
        self.size = new_size
        self.table = [[] for _ in range(new_size)]
        self.count = 0
        for chain in old_table:
            for key, value in chain:
                self.put(key, value)

    def put(self, key: int, value) -> None:
        """열쇠-값 쌍을 넣거나 갱신하고, 필요하면 크기를 바꾼다."""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
        self.count += 1

        if self.count / self.size > self.GROW_THRESHOLD:
            self._resize(self.size * 2)

    def delete(self, key: int) -> bool:
        """열쇠를 지우고, 필요하면 표를 줄인다."""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                self.count -= 1
                if (self.size > self.MIN_SIZE
                        and self.count / self.size < self.SHRINK_THRESHOLD):
                    self._resize(max(self.size // 2, self.MIN_SIZE))
                return True
        return False

    def get(self, key: int):
        """열쇠에 딸린 값을 돌려주고, 없으면 None을 돌려준다."""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def load_factor(self) -> float:
        return self.count / self.size

# === 시연 ===

if __name__ == "__main__":
    ht = DynamicHashTable(size=4)

    print("=== Insertions with automatic growth ===")
    for i in range(1, 13):
        ht.put(i, i * 10)
        print(f"Insert {i:2d}: n={ht.count:2d}, m={ht.size:2d}, "
              f"alpha={ht.load_factor():.3f}")

    print("\n=== Deletions with automatic shrinking ===")
    for i in range(1, 10):
        ht.delete(i)
        print(f"Delete {i:2d}: n={ht.count:2d}, m={ht.size:2d}, "
              f"alpha={ht.load_factor():.3f}")
```

**출력:**
```
=== Insertions with automatic growth ===
Insert  1: n= 1, m= 4, alpha=0.250
Insert  2: n= 2, m= 4, alpha=0.500
Insert  3: n= 3, m= 4, alpha=0.750
Insert  4: n= 4, m= 8, alpha=0.500
Insert  5: n= 5, m= 8, alpha=0.625
Insert  6: n= 6, m= 8, alpha=0.750
Insert  7: n= 7, m=16, alpha=0.438
Insert  8: n= 8, m=16, alpha=0.500
Insert  9: n= 9, m=16, alpha=0.562
Insert 10: n=10, m=16, alpha=0.625
Insert 11: n=11, m=16, alpha=0.688
Insert 12: n=12, m=16, alpha=0.750
=== Deletions with automatic shrinking ===
Delete  1: n=11, m=16, alpha=0.688
Delete  2: n=10, m=16, alpha=0.625
Delete  3: n= 9, m=16, alpha=0.562
Delete  4: n= 8, m=16, alpha=0.500
Delete  5: n= 7, m=16, alpha=0.438
Delete  6: n= 6, m=16, alpha=0.375
Delete  7: n= 5, m=16, alpha=0.312
Delete  8: n= 4, m=16, alpha=0.250
Delete  9: n= 3, m= 8, alpha=0.375
```

---

## 연습문제

**연습문제 1.**
크기 조정과 재해싱에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
크기 조정과 재해싱을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
크기 조정과 재해싱은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

동적 크기 조정은 적재율을 일정한 범위 안에 묶어 해시 표가 쓰이는 내내 기대 시간 $O(1)$의 연산을 보장한다. 커질 때 표를 두 배로, 줄어들 때 절반으로 하되 요동을 막으려고 문턱값을 비대칭으로 두면 연산마다 분할 상환 비용이 $O(1)$이 된다. 지연에 민감한 응용에서는 점진적 재해싱이 크기 조정의 $\Theta(n)$ 비용을 여러 연산에 나누어 최악의 연산 비용을 묶는다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Introduction to Algorithms (CLRS), 16장 — 동적 표](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
