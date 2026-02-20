# Length of String


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

**Output:**
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


# Reference

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
