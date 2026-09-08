# 기호표

모든 프로그래밍 언어 구현에는 **식별자**(변수 이름, 함수 이름, 클래스 이름)와 그 **속성**(자료형, 유효 범위, 메모리 주소, 값)을 잇는 방법이 필요하다. **기호표**는 컴파일이나 해석 내내 이러한 연결을 관리하는 자료 구조이다. 식별자 조회는 컴파일러에서 가장 잦은 연산 가운데 하나이므로, 평균 $O(1)$ 접근을 주는 해시 테이블이 표준 구현이다.

---

## 1. 연산

기호표는 네 가지 핵심 연산을 지원한다.

| 연산 | 설명 | 시간 (해시 기반) |
|---|---|---|
| **insert**(name, attributes) | 새 식별자를 더한다 | 기대 $O(1)$ |
| **lookup**(name) | 식별자의 속성을 가져온다 | 기대 $O(1)$ |
| **update**(name, attributes) | 기존 항목의 속성을 고친다 | 기대 $O(1)$ |
| **delete**(name) | 식별자를 없앤다 (유효 범위를 벗어날 때) | 기대 $O(1)$ |

해시 테이블을 쓰면 모든 연산이 기대 $O(1)$ 시간에 돌아간다. 균형 이진 탐색 나무는 $O(\log n)$, 정렬되지 않은 리스트는 $O(n)$이다.

---

## 2. 유효 범위와 중첩된 기호표

대부분의 언어는 **어휘적 유효 범위**를 지원한다. 블록 안에서 선언한 식별자는 그 블록과 그 안에 중첩된 블록에서만 보인다. 기호표는 두 가지 전략 가운데 하나로 유효 범위를 다룬다.

### 유효 범위 스택

유효 범위의 단계마다 하나씩인 **해시 테이블의 스택**을 관리한다. 새 유효 범위가 열리면(예: 함수 본문이나 반복문에 들어가면) 새 해시 테이블을 쌓는다. 유효 범위가 닫히면 그 테이블을 뺀다.

조회는 스택을 위에서 아래로 훑으며 처음 맞는 것을 돌려준다.

$$
\text{lookup}(x) = \text{first } T_i \text{ in stack where } x \in T_i
$$

중첩의 깊이를 $d$이라 할 때 최악의 경우 $O(d)$이 들지만, $d$은 보통 작다(실무에서 10을 넘는 일이 드물다).

### 유효 범위 사슬을 갖는 단일 테이블

항목마다 유효 범위의 깊이 순으로 정렬된 정의의 연결 리스트를 담는 해시 테이블 하나를 쓴다. 가장 최근의 정의가 앞선 정의를 가린다. 유효 범위가 닫히면 그 깊이의 항목을 모두 없앤다.

---

## 3. 무엇을 담는가

기호표의 항목에는 보통 다음이 들어간다.

| 속성 | 예 |
|---|---|
| 이름 | `"count"` |
| 자료형 | `int`, `float`, `str` |
| 유효 범위 단계 | `0` (전역), `1` (함수), `2` (블록) |
| 메모리 위치 | 스택 프레임 바닥에서의 오프셋 |
| 크기 | 바이트 수 |
| 상수/가변 | 그 이름이 바뀔 수 없는지 |
| 매개변수 | 함수의 경우 매개변수의 자료형과 개수 |

---

## 4. 설계에서의 고려

**해시 함수의 선택**: 실제 프로그램의 식별자는 접두사를 공유하는 일이 많다(`getWidth`, `getHeight`, `getName`). 좋은 해시 함수는 이렇게 비슷한 문자열을 고르게 흩뿌려야 한다. 밑이 소수인 다항 해싱이 잘 통한다.

$$
h(s) = \left(\sum_{i=0}^{|s|-1} s[i] \cdot p^i \right) \bmod m
$$

**테이블의 크기**: 보통의 프로그램에는 식별자가 수백에서 수천 개 있다. 칸이 수백 개이고 적재율이 $0.75$ 아래인 해시 테이블이면 대부분의 경우를 효율적으로 감당한다.

**문자열 인터닝**: 문자열을 되풀이해 비교하지 않으려고 많은 구현이 식별자 문자열을 **인턴**한다. 서로 다른 문자열을 한 번씩만 저장하고 포인터나 참조가 같은지로 비교하는 것이다. 인턴 테이블 자체가 해시 집합이다.

---

## 5. 파이썬 구현

```python
"""
해시 표 기반 유효 범위 스택으로 구현한 기호표.

컴파일러와 해석기가 중첩된 어휘적 유효 범위에 걸쳐
이름의 묶임을 어떻게 다루는지 보인다.
"""

# === 기호표 항목 ===

class Symbol:
    """이름 하나의 속성을 나타낸다."""

    def __init__(self, name, sym_type, scope_level, value=None):
        self.name = name
        self.sym_type = sym_type
        self.scope_level = scope_level
        self.value = value

    def __repr__(self):
        return (f"Symbol({self.name}, type={self.sym_type}, "
                f"scope={self.scope_level}, value={self.value})")

# === 유효 범위가 있는 기호표 ===

class SymbolTable:
    """중첩된 어휘적 유효 범위를 위한 유효 범위 스택이 있는 기호표."""

    def __init__(self):
        self.scopes = [{}]  # 해시 표의 스택
        self.level = 0

    def enter_scope(self):
        """새 유효 범위를 스택에 밀어 넣는다."""
        self.level += 1
        self.scopes.append({})

    def exit_scope(self):
        """지금 유효 범위를 스택에서 꺼낸다."""
        if self.level == 0:
            raise RuntimeError("Cannot exit global scope")
        self.scopes.pop()
        self.level -= 1

    def insert(self, name, sym_type, value=None):
        """지금 유효 범위에 이름을 넣는다."""
        symbol = Symbol(name, sym_type, self.level, value)
        self.scopes[-1][name] = symbol
        return symbol

    def lookup(self, name):
        """가장 안쪽 유효 범위에서 바깥쪽으로 찾아 나가며 이름을 찾는다."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_current_scope(self, name):
        """지금 유효 범위에서만 이름을 찾는다."""
        return self.scopes[-1].get(name)

# === 시연 ===

if __name__ == "__main__":
    st = SymbolTable()

    # 전역 유효 범위
    st.insert("x", "int", 10)
    st.insert("pi", "float", 3.14)
    print(f"Global lookup('x'): {st.lookup('x')}")

    # 함수 유효 범위로 들어가기
    st.enter_scope()
    st.insert("x", "int", 42)  # 전역 x를 가린다
    st.insert("y", "str", "hello")
    print(f"Function lookup('x'): {st.lookup('x')}")
    print(f"Function lookup('pi'): {st.lookup('pi')}")  # 전역에서 찾음

    # 블록 유효 범위로 들어가기
    st.enter_scope()
    st.insert("z", "bool", True)
    print(f"Block lookup('x'): {st.lookup('x')}")  # 여전히 함수의 x
    print(f"Block lookup('z'): {st.lookup('z')}")

    # 블록 유효 범위에서 나오기
    st.exit_scope()
    print(f"After block, lookup('z'): {st.lookup('z')}")  # None

    # 함수 유효 범위에서 나오기
    st.exit_scope()
    print(f"After function, lookup('x'): {st.lookup('x')}")  # 전역 x
```

**출력:**
```
Global lookup('x'): Symbol(x, type=int, scope=0, value=10)
Function lookup('x'): Symbol(x, type=int, scope=1, value=42)
Function lookup('pi'): Symbol(pi, type=float, scope=0, value=3.14)
Block lookup('x'): Symbol(x, type=int, scope=1, value=42)
Block lookup('z'): Symbol(z, type=bool, scope=2, value=True)
After block, lookup('z'): None
After function, lookup('x'): Symbol(x, type=int, scope=0, value=10)
```

---

## 연습문제

**연습문제 1.**
기호표에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
기호표을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
기호표은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 연산、유효 범위와 중첩된 기호표、무엇을 담는가、설계에서의 고려을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Aho, A. V., Lam, M. S., Sethi, R., and Ullman, J. D. *Compilers: Principles, Techniques, and Tools*, 2nd edition.
