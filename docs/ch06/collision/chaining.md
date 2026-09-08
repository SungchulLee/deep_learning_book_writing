# 체이닝

두 키가 같은 칸으로 해시되면 충돌이 일어난다. 체이닝은 같은 인덱스로 가는 키를 모두 그 칸에 딸린 연결 리스트(또는 다른 보조 자료 구조)에 담아 충돌을 해결한다. 충돌이 아무리 많아도 모든 키에 자리가 있으므로 테이블 자체의 공간이 모자라는 일은 없다. 넘치면 사슬이 길어질 뿐이다.

---

## 1. 작동 방식

크기가 $m$인 해시 테이블은 각 항목이 연결 리스트의 머리를 가리키는 배열 $T[0 \ldots m-1]$을 관리한다. 해시 함수 $h$에 대해 세 가지 기본 연산은 다음과 같이 작동한다.

키 $k$ **넣기**: $h(k)$을 계산한 뒤 $T[h(k)]$의 리스트 앞에 $k$을 붙인다. (중복 검사를 하지 않는다면) 최악의 경우에도 $O(1)$이다.

키 $k$ **찾기**: $h(k)$을 계산한 뒤 $T[h(k)]$의 리스트를 훑으며 $k$을 찾는다. 걸리는 시간은 그 사슬의 길이에 달렸다.

키 $k$ **지우기**: $T[h(k)]$의 사슬에서 $k$을 찾아 그 노드를 없앤다. 이중 연결 리스트라면 노드를 찾은 뒤의 삭제 자체는 $O(1)$이다.

---

## 2. 단순 균등 해싱 가정

체이닝의 분석은 **단순 균등 해싱 가정(SUHA)**에 기댄다. 각 키가 다른 키가 어디로 가는지와 무관하게 $m$개의 칸 어디로든 같은 확률로 해시된다는 가정이다. 이 가정 아래 크기 $m$인 테이블에 키 $n$개를 저장할 때 **적재율**은 다음과 같다.

$$
\alpha = \frac{n}{m}
$$

그리고 각 사슬의 기대 길이는 $\alpha$이다.

---

## 3. 기대 탐색 시간

### 탐색 실패

탐색 실패는 $T[h(k)]$ 사슬의 모든 원소를 살핀다. 단순 균등 해싱 가정 아래 기대 사슬 길이가 $\alpha$이므로 해시 계산까지 포함한 기대 시간은 다음과 같다.

$$
\Theta(1 + \alpha)
$$

### 탐색 성공

키 $k$의 탐색 성공은 평균적으로 $k$을 넣은 뒤 같은 사슬에 들어간 원소의 절반과 $k$ 자신을 살핀다. $n$개의 키에 대해 합하고 평균하면 다음을 얻는다.

$$
\Theta\!\left(1 + \frac{\alpha}{2}\right) = \Theta(1 + \alpha)
$$

### 상수 시간 연산

칸의 수 $m$을 키의 수 $n$에 비례하게 잡으면 적재율이 $\alpha = n/m = O(1)$이 된다. 이때 삽입, 탐색, 삭제라는 사전의 모든 연산이 기대 $O(1)$ 시간에 돌아간다.

---

## 4. 최악의 경우의 거동

$n$개의 키가 모두 같은 칸으로 해시되면 사슬 하나의 길이가 $n$이 되어 모든 탐색이 선형 훑기로 퇴화한다. 따라서 탐색과 삭제의 최악의 경우 시간은 다음과 같다.

$$
\Theta(n)
$$

이 병적인 경우 때문에 최악의 충돌이 일어나기 어렵게 하려고 보편 해싱을 비롯한 무작위 해시 족을 쓴다.

---

## 5. 사슬 자료 구조의 선택

연결 리스트가 표준적인 선택이지만 어떤 동적 집합 구조든 사슬로 쓸 수 있다.

| 사슬 구조 | 탐색 | 삽입 | 삭제 | 캐시 거동 |
|---|---|---|---|---|
| 단일 연결 리스트 | $O(\ell)$ | $O(1)$ | $O(\ell)$ | 나쁨 |
| 이중 연결 리스트 | $O(\ell)$ | $O(1)$ | $O(1)$* | 나쁨 |
| 동적 배열 | $O(\ell)$ | 상각 $O(1)$ | $O(\ell)$ | 좋음 |
| 균형 이진 탐색 나무 | $O(\log \ell)$ | $O(\log \ell)$ | $O(\log \ell)$ | 보통 |

여기서 $\ell$은 사슬의 길이이고 $^*$는 노드의 포인터를 이미 안다고 전제한 것이다.

사슬로 균형 이진 탐색 나무를 쓰면 균등성 가정 없이도 최악의 경우 $O(\log n)$ 탐색이 보장되지만 상수가 커진다. 자바의 `HashMap`은 사슬이 여덟 원소를 넘으면 연결 리스트에서 레드-블랙 나무로 바꾼다.

---

## 6. 파이썬 구현

```python
"""
사슬법 기반 해시 표 구현.

칸마다 (열쇠, 값) 쌍의 연결 리스트를 두는
사슬법 충돌 해결을 보인다.
"""

# === 노드와 연결 리스트 ===

class Node:
    """단일 연결 사슬의 노드."""

    __slots__ = ("key", "value", "next")

    def __init__(self, key, value, next_node=None):
        self.key = key
        self.value = value
        self.next = next_node

# === 사슬법을 쓰는 해시 표 ===

class ChainingHashTable:
    """사슬법으로 충돌을 푸는 해시 표."""

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        """열쇠-값 쌍을 넣거나 갱신한다."""
        idx = self._hash(key)
        node = self.table[idx]
        while node is not None:
            if node.key == key:
                node.value = value  # 이미 있는 것 갱신
                return
            node = node.next
        # 새 노드를 앞에 붙이기 (O(1) 삽입)
        self.table[idx] = Node(key, value, self.table[idx])
        self.size += 1

    def search(self, key):
        """열쇠의 값을 돌려주고, 없으면 None을 돌려준다."""
        idx = self._hash(key)
        node = self.table[idx]
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        return None

    def delete(self, key):
        """표에서 열쇠를 없앤다. 있었으면 True를 돌려준다."""
        idx = self._hash(key)
        prev, node = None, self.table[idx]
        while node is not None:
            if node.key == key:
                if prev is None:
                    self.table[idx] = node.next
                else:
                    prev.next = node.next
                self.size -= 1
                return True
            prev, node = node, node.next
        return False

    def load_factor(self):
        """지금 적재율 alpha = n / m을 돌려준다."""
        return self.size / self.capacity

    def chain_lengths(self):
        """살펴볼 수 있도록 사슬 길이의 리스트를 돌려준다."""
        lengths = []
        for head in self.table:
            length, node = 0, head
            while node is not None:
                length += 1
                node = node.next
            lengths.append(length)
        return lengths

# === 시연 ===

if __name__ == "__main__":
    ht = ChainingHashTable(capacity=4)

    # 열쇠 여러 개 삽입 (capacity=4에서는 충돌이 일어나기 쉽다)
    for key, val in [("apple", 1), ("banana", 2), ("cherry", 3),
                     ("date", 4), ("elderberry", 5), ("fig", 6)]:
        ht.insert(key, val)

    print(f"Load factor: {ht.load_factor():.2f}")
    print(f"Chain lengths: {ht.chain_lengths()}")
    print(f"Search 'cherry': {ht.search('cherry')}")
    print(f"Search 'grape': {ht.search('grape')}")

    ht.delete("banana")
    print(f"After deleting 'banana': {ht.search('banana')}")
    print(f"Chain lengths: {ht.chain_lengths()}")
```

**출력:**
```
Load factor: 1.50
Chain lengths: [1, 1, 2, 2]
Search 'cherry': 3
Search 'grape': None
After deleting 'banana': None
Chain lengths: [1, 0, 2, 2]
```

---

## 연습문제

**연습문제 1.**
체이닝에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
체이닝을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
체이닝은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 작동 방식、단순 균등 해싱 가정、기대 탐색 시간、최악의 경우의 거동을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Hashing Technique - Simplified](https://www.youtube.com/watch?v=mFY0J5W8Udk&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=79)
