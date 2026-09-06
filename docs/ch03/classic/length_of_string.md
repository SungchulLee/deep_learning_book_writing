# 문자열의 길이
<img src="img/Screen Shot 2022-05-01 at 8.01.14 PM.png" width=70%>

```python
def compute_length_of_string_using_recursion(string):
    if string == '':
        return 0
    return 1 + compute_length_of_string_using_recursion(string[1:])

def compute_length_of_string_using_python(string):
    return len(string)

def main():
    # https://edition.cnn.com/2022/04/29/asia/kabul-mosque-explosion-intl/index.html
    strings = """At least 10 people were killed and 30 wounded after an explosion at a mosque in the Afghan capital Kabul after Friday prayers Taliban spokesperson for the chief of police Khalid Zadran said""".split()
    for string in strings:
        result_0 = compute_length_of_string_using_recursion(string)
        result_1 = compute_length_of_string_using_python(string)
        print(f"Computation of length of string '{string}' using recursion : {result_0}")
        print(f"Computation of length of string '{string}' using python    : {result_1}")
        print()
        
        
if __name__ == "__main__":
    main()
```

**출력:**
```
Computation of length of string 'At' using recursion : 2
Computation of length of string 'At' using python    : 2

Computation of length of string 'least' using recursion : 5
Computation of length of string 'least' using python    : 5

Computation of length of string '10' using recursion : 2
Computation of length of string '10' using python    : 2

Computation of length of string 'people' using recursion : 6
Computation of length of string 'people' using python    : 6

Computation of length of string 'were' using recursion : 4
Computation of length of string 'were' using python    : 4

Computation of length of string 'killed' using recursion : 6
Computation of length of string 'killed' using python    : 6

Computation of length of string 'and' using recursion : 3
Computation of length of string 'and' using python    : 3

Computation of length of string '30' using recursion : 2
Computation of length of string '30' using python    : 2

Computation of length of string 'wounded' using recursion : 7
Computation of length of string 'wounded' using python    : 7

Computation of length of string 'after' using recursion : 5
Computation of length of string 'after' using python    : 5

Computation of length of string 'an' using recursion : 2
Computation of length of string 'an' using python    : 2

Computation of length of string 'explosion' using recursion : 9
Computation of length of string 'explosion' using python    : 9

Computation of length of string 'at' using recursion : 2
Computation of length of string 'at' using python    : 2

Computation of length of string 'a' using recursion : 1
Computation of length of string 'a' using python    : 1

Computation of length of string 'mosque' using recursion : 6
Computation of length of string 'mosque' using python    : 6

Computation of length of string 'in' using recursion : 2
Computation of length of string 'in' using python    : 2

Computation of length of string 'the' using recursion : 3
Computation of length of string 'the' using python    : 3

Computation of length of string 'Afghan' using recursion : 6
Computation of length of string 'Afghan' using python    : 6

Computation of length of string 'capital' using recursion : 7
Computation of length of string 'capital' using python    : 7

Computation of length of string 'Kabul' using recursion : 5
Computation of length of string 'Kabul' using python    : 5

Computation of length of string 'after' using recursion : 5
Computation of length of string 'after' using python    : 5

Computation of length of string 'Friday' using recursion : 6
Computation of length of string 'Friday' using python    : 6

Computation of length of string 'prayers' using recursion : 7
Computation of length of string 'prayers' using python    : 7

Computation of length of string 'Taliban' using recursion : 7
Computation of length of string 'Taliban' using python    : 7

Computation of length of string 'spokesperson' using recursion : 12
Computation of length of string 'spokesperson' using python    : 12

Computation of length of string 'for' using recursion : 3
Computation of length of string 'for' using python    : 3

Computation of length of string 'the' using recursion : 3
Computation of length of string 'the' using python    : 3

Computation of length of string 'chief' using recursion : 5
Computation of length of string 'chief' using python    : 5

Computation of length of string 'of' using recursion : 2
Computation of length of string 'of' using python    : 2

Computation of length of string 'police' using recursion : 6
Computation of length of string 'police' using python    : 6

Computation of length of string 'Khalid' using recursion : 6
Computation of length of string 'Khalid' using python    : 6

Computation of length of string 'Zadran' using recursion : 6
Computation of length of string 'Zadran' using python    : 6

Computation of length of string 'said' using recursion : 4
Computation of length of string 'said' using python    : 4
```

# 참고 자료

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)

## 연습문제

**연습문제 1.**
문자열에서 특정 글자가 나타나는 횟수를 세는 재귀 함수를 작성하라.

??? success "연습문제 1 풀이"
    ```python
    def count_char(s, c):
        if not s:
            return 0
        return (1 if s[0] == c else 0) + count_char(s[1:], c)
    ```

---

**연습문제 2.**
문자열이 회문인지 확인하는 재귀 함수를 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def is_palindrome(s):
        if len(s) <= 1:
            return True
        return s[0] == s[-1] and is_palindrome(s[1:-1])
    ```

---

**연습문제 3.**
파이썬 문자열이 불변일 때(슬라이싱이 복사본을 만든다) 재귀적 `length_of_string`의 시간 복잡도는 얼마인가?

??? success "연습문제 3 풀이"
    각 재귀 호출이 슬라이싱으로 길이 $n-1$의 새 문자열을 만들며 $O(n-1)$ 시간이 든다. 총합은 $\sum_{k=1}^{n} k = O(n^2)$이다. $O(n)$을 얻으려면 슬라이싱 대신 인덱스를 넘긴다. `def length(s, i=0): return 0 if i == len(s) else 1 + length(s, i+1)`.

---

**연습문제 4.**
문자열을 뒤집는 재귀 함수를 작성하라.

??? success "연습문제 4 풀이"
    ```python
    def reverse_string(s):
        if len(s) <= 1:
            return s
        return reverse_string(s[1:]) + s[0]
    ```
    문자열 슬라이싱과 이어 붙이기 때문에 시간 복잡도가 $O(n^2)$이다. $O(n)$ 방식은 리스트와 인덱스를 쓴다.
