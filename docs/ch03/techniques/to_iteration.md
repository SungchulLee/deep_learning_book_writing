# 반복으로 변환하기

모든 재귀 알고리즘은 반복 알고리즘으로 바꿀 수 있고 그 반대도 마찬가지이다. 이 변환은 실무에서 중요하다. 반복적 해법은 스택 넘침의 위험을 피하고 함수 호출 부담이 없어져 더 빠르게 실행되는 경우가 많기 때문이다. 세 가지 주요 기법이 이 변환을 처리한다.

---

## 1. 기법 1: 반복문으로 직접 바꾸기

꼬리 재귀 함수는 반복문으로 곧바로 바뀐다. 누적자가 반복 변수가 된다.

```python
"""꼬리 재귀를 반복으로 변환하기."""

# === 재귀적 계승(꼬리 형태) ===

def factorial_recursive(n, acc=1):
    """꼬리 재귀 계승."""
    if n <= 1:
        return acc
    return factorial_recursive(n - 1, acc * n)

# === 반복적 계승 ===

def factorial_iterative(n):
    """위와 같은 논리를 반복문으로 표현한 것."""
    acc = 1
    while n > 1:
        acc *= n
        n -= 1
    return acc

# === 메인 ===

if __name__ == "__main__":
    for n in [0, 1, 5, 10]:
        r = factorial_recursive(n)
        i = factorial_iterative(n)
        print(f"{n}! = {r} (recursive) = {i} (iterative)")
```

**출력:**
```
0! = 1 (recursive) = 1 (iterative)
1! = 1 (recursive) = 1 (iterative)
5! = 120 (recursive) = 120 (iterative)
10! = 3628800 (recursive) = 3628800 (iterative)
```

---

## 2. 기법 2: 명시적 스택

(트리 순회처럼) 되돌아 나오는 과정이 필요한 꼬리가 아닌 재귀 함수는 명시적 스택 자료구조로 호출 스택을 흉내 낼 수 있다.

```python
# 재귀적 DFS
def dfs_recursive(node):
    visit(node)
    for child in node.children:
        dfs_recursive(child)

# 명시적 스택을 쓰는 반복적 DFS
def dfs_iterative(root):
    stack = [root]
    while stack:
        node = stack.pop()
        visit(node)
        for child in reversed(node.children):
            stack.append(child)
```

---

## 3. 기법 3: 트램펄린

상호 재귀이거나 복잡한 재귀 함수에서는 논리를 재구성하지 않고도 트램펄린 반복문으로 재귀를 대체할 수 있다(꼬리 호출 최적화 절 참고).

---

## 연습문제

**연습문제 1.**
반복으로 변환하기에 대해 기저 사례와 재귀 사례를 찾아라. 모든 유효한 입력에 대해 재귀가 종료됨을 증명하라.

??? success "연습문제 1 풀이"
    기저 사례는 가장 작은 유효한 입력을 직접 처리한다. 재귀 사례는 호출마다 감소하는 잘 정의된 척도로 문제 크기를 줄인다. 이 척도가 (기저 사례의 문턱값으로) 아래로 유계이면서 엄격히 감소하므로 재귀는 반드시 유한한 단계 안에 종료된다.

---

**연습문제 2.**
반복으로 변환하기의 시간 복잡도에 대한 점화식을 유도하고 풀어라.

??? success "연습문제 2 풀이"
    $T(n)$을 재귀 호출과 호출당 일의 함수로 표현한다. 이 점화식은 분기 계수, 부분문제 크기의 감소, 결합 비용을 담는다. 펼치기, 마스터 정리, 치환 중 하나로 풀어 닫힌 형태를 얻는다.

---

**연습문제 3.**
$n = 8$일 때 반복으로 변환하기의 재귀 트리를 그려라. 각 층에서의 일과 전체 일을 계산하라.

??? success "연습문제 3 풀이"
    트리의 깊이는 입력이 기저 사례까지 얼마나 빨리 줄어드는지로 정해진다. 각 층에서 모든 노드의 일을 더한다. 모든 층에 걸친 총합이 실행 시간을 준다. $n = 8$이면 트리가 작아서 전부 열거할 수 있다.

---

**연습문제 4.**
재귀 구현을 반복 버전으로 변환하라. 공간 복잡도를 비교하라.

??? success "연습문제 4 풀이"
    호출 스택을 명시적 스택이나 반복 변수로 대체한다. 꼬리 재귀 형태는 while 반복문으로 곧바로 바뀐다. 꼬리가 아닌 형태는 호출 스택을 흉내 내기 위해 명시적 스택이 필요하다. 반복 버전은 보통 $O(\text{depth})$의 스택 공간을 아낀다.

## 정리하며

이 마당은 기법 1: 반복문으로 직접 바꾸기、기법 2: 명시적 스택、기법 3: 트램펄린을 차례로 짚었다.

**참고 문헌**

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
