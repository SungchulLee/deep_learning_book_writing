# Print String

<img src="img/Screen Shot 2022-05-01 at 8.13.58 PM.png" width=50%>

```python
def print_string_using_recursion(string):
    if string == '':
        return 
    print(string[0], end='')
    print_string_using_recursion(string[1:])

def print_string_using_using_python(string):
    print(string)

def main():
    # https://edition.cnn.com/2022/04/29/asia/kabul-mosque-explosion-intl/index.html
    string = """At least 10 people were killed and 30 wounded after an explosion at a mosque in the Afghan capital Kabul after Friday prayers Taliban spokesperson for the chief of police Khalid Zadran said"""
    print(f"Print string using recursion :")
    print_string_using_recursion(string)
    print()
    print()
    print(f"Print string using python built-in functionality :")
    print_string_using_using_python(string)
        
if __name__ == "__main__":
    main()
```

**Output:**
```
Print string using recursion :
At least 10 people were killed and 30 wounded after an explosion at a mosque in the Afghan capital Kabul after Friday prayers Taliban spokesperson for the chief of police Khalid Zadran said

Print string using python built-in functionality :
At least 10 people were killed and 30 wounded after an explosion at a mosque in the Afghan capital Kabul after Friday prayers Taliban spokesperson for the chief of police Khalid Zadran said
```

# Reference

[[알고리즘] 제1-2강 Recursion의 개념과 기본 예제들 (2/3)](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
