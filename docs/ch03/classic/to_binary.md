# 이진수로 바꾸기
<img src="img/Screen Shot 2022-05-01 at 8.26.20 PM.png" width=70%>

```python
def change_to_binary_and_print_using_recursion(num):
    if num <= 1:
        print(num, end='') 
        return
    
    change_to_binary_and_print_using_recursion(num//2)
    print(num%2, end='') 

def change_to_binary_and_print_using_python(num):
    print(bin(num)[2:])

def main():
    nums = (
    0,
    10,
    2,
    25,
    1256
    )
    for num in nums:
        print(f"Print number as binary using recursion : ", end='')
        change_to_binary_and_print_using_recursion(num)
        print()
        
        print(f"Print number as binary using python    : ", end='')
        change_to_binary_and_print_using_python(num)
        print()

        
if __name__ == "__main__":
    main()
```

**출력:**
```
Print number as binary using recursion : 0
Print number as binary using python    : 0

Print number as binary using recursion : 1010
Print number as binary using python    : 1010

Print number as binary using recursion : 10
Print number as binary using python    : 10

Print number as binary using recursion : 11001
Print number as binary using python    : 11001

Print number as binary using recursion : 10011101000
Print number as binary using python    : 10011101000
```

# 참고 자료

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)

## 연습문제

**연습문제 1.**
이진 변환을 임의의 밑 $b$(2부터 16까지)를 다루도록 확장하라.

??? success "연습문제 1 풀이"
    ```python
    def to_base(n, base):
        if n == 0:
            return '0'
        digits = '0123456789ABCDEF'
        if n < base:
            return digits[n]
        return to_base(n // base, base) + digits[n % base]
    ```

---

**연습문제 2.**
이진 문자열을 다시 십진수로 바꾸는 재귀 함수를 작성하라.

??? success "연습문제 2 풀이"
    ```python
    def from_binary(s):
        if len(s) == 0:
            return 0
        return int(s[-1]) + 2 * from_binary(s[:-1])
    ```

---

**연습문제 3.**
`to_binary(n)`은 재귀 호출을 몇 번 하는가? $n$으로 표현하라.

??? success "연습문제 3 풀이"
    호출마다 $n$을 2로 나누므로 호출 횟수는 $\lfloor \log_2 n \rfloor + 1$이며, 이는 이진 표현의 비트 수와 같다.

---

**연습문제 4.**
고정된 비트 너비의 2의 보수 표현을 사용해 음수를 다루도록 함수를 수정하라.

??? success "연습문제 4 풀이"
    ```python
    def to_twos_complement(n, bits=8):
        if n < 0:
            n = (1 << bits) + n  # 2^bits + n
        result = ''
        for _ in range(bits):
            result = str(n % 2) + result
            n //= 2
        return result

    assert to_twos_complement(-1, 8) == '11111111'
    assert to_twos_complement(5, 8) == '00000101'
    ```
