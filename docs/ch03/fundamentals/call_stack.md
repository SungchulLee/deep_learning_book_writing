# 호출 스택

함수가 자기 자신을 재귀적으로 호출하면 호출할 때마다 **호출 스택** 에 새 항목이 만들어진다. 호출 스택은 활성 상태인 함수 호출을 추적하는 실행 시간 자료구조이다. 호출 스택을 이해하는 것은 재귀를 따져 보는 데 필수적이다. 각 재귀 호출이 끝난 뒤 컴퓨터가 어디로 돌아갈지 어떻게 기억하는지를 설명해 주기 때문이다.

## 호출 스택의 동작 방식

함수가 호출될 때마다 실행 시간 환경은 다음을 담은 **프레임** 을 호출 스택에 밀어 넣는다.

1. 함수의 지역 변수와 매개변수
2. 반환 주소(호출이 끝난 뒤 재개할 위치)

함수가 반환하면 그 프레임이 스택에서 꺼내지고 반환 주소에서 실행이 재개된다.

## 예제: 호출 스택 따라가기

재귀적 카운트다운 함수를 생각해 보자. 호출할 때마다 스택에 프레임이 하나씩 추가되고, 호출이 반환되면서 프레임이 제거된다.

```python
"""재귀 중 호출 스택의 동작 시연."""


# === 스택을 추적하는 재귀적 카운트다운 ===

def countdown(n, depth=0):
    """스택 깊이를 들여쓰기로 보여주며 카운트다운을 출력한다."""
    indent = "  " * depth
    print(f"{indent}countdown({n}) called  [stack depth: {depth + 1}]")
    if n == 0:
        print(f"{indent}Base case reached, returning")
        return
    countdown(n - 1, depth + 1)
    print(f"{indent}countdown({n}) returning  [stack depth: {depth + 1}]")


# === 메인 ===

if __name__ == "__main__":
    countdown(3)
```

**출력:**
```
countdown(3) called  [stack depth: 1]
  countdown(2) called  [stack depth: 2]
    countdown(1) called  [stack depth: 3]
      countdown(0) called  [stack depth: 4]
      Base case reached, returning
    countdown(1) returning  [stack depth: 3]
  countdown(2) returning  [stack depth: 2]
countdown(3) returning  [stack depth: 1]
```

## 스택 넘침

재귀 함수에 제대로 된 기저 사례가 없거나 기저 사례에 결코 도달하지 못하면, 호출 스택은 시스템의 스택 공간이 바닥날 때까지 끝없이 커진다. 이것이 **스택 넘침(stack overflow)** 오류를 낳는다.

```python
def infinite_recursion(n):
    """이 함수는 결코 종료되지 않는다. n이 커지기만 해서 0에 도달하지 못한다."""
    print("H", end="")
    infinite_recursion(n + 1)  # n이 증가하므로 0의 기저 사례에 도달할 수 없다

# infinite_recursion(1)을 호출하면 다음이 발생한다:
# RecursionError: 되부름 깊이 한도를 넘었다
```

파이썬은 진짜 스택 넘침을 막기 위해 재귀 깊이를 제한한다(기본값 1000). 이 안전장치가 무한 재귀를 `RecursionError`로 바꾸어 준다.

## 참고 자료

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)


## 연습문제

**연습문제 1.**
호출 스택에 대해 기저 사례와 재귀 사례를 찾아라. 모든 유효한 입력에 대해 재귀가 종료됨을 증명하라.

??? success "연습문제 1 풀이"
    기저 사례는 가장 작은 유효한 입력을 직접 처리한다. 재귀 사례는 호출마다 감소하는 잘 정의된 척도로 문제 크기를 줄인다. 이 척도가 (기저 사례의 문턱값으로) 아래로 유계이면서 엄격히 감소하므로 재귀는 반드시 유한한 단계 안에 종료된다.

---

**연습문제 2.**
호출 스택의 시간 복잡도에 대한 점화식을 유도하고 풀어라.

??? success "연습문제 2 풀이"
    $T(n)$을 재귀 호출과 호출당 일의 함수로 표현한다. 이 점화식은 분기 계수, 부분문제 크기의 감소, 결합 비용을 담는다. 펼치기, 마스터 정리, 치환 중 하나로 풀어 닫힌 형태를 얻는다.

---

**연습문제 3.**
$n = 8$일 때 호출 스택의 재귀 트리를 그려라. 각 층에서의 일과 전체 일을 계산하라.

??? success "연습문제 3 풀이"
    트리의 깊이는 입력이 기저 사례까지 얼마나 빨리 줄어드는지로 정해진다. 각 층에서 모든 노드의 일을 더한다. 모든 층에 걸친 총합이 실행 시간을 준다. $n = 8$이면 트리가 작아서 전부 열거할 수 있다.

---

**연습문제 4.**
재귀 구현을 반복 버전으로 변환하라. 공간 복잡도를 비교하라.

??? success "연습문제 4 풀이"
    호출 스택을 명시적 스택이나 반복 변수로 대체한다. 꼬리 재귀 형태는 while 반복문으로 곧바로 바뀐다. 꼬리가 아닌 형태는 호출 스택을 흉내 내기 위해 명시적 스택이 필요하다. 반복 버전은 보통 $O(\text{depth})$의 스택 공간을 아낀다.