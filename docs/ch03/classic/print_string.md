# Print String

Printing a string character by character is one of the simplest examples of recursion on a sequence. The recursive approach processes the first character, then recurses on the remaining substring. This pattern — handle one element, recurse on the rest — is the foundation of linear recursion on sequences.

## Recursive Structure

- **Base case**: the string is empty — nothing to print, return
- **Recursive case**: print the first character, then recursively print the rest

```python
"""Print a string character by character using recursion."""


# === Recursive Print ===

def print_string_recursive(string):
    """Print each character of string using recursion."""
    if string == "":
        return
    print(string[0], end="")
    print_string_recursive(string[1:])


# === Built-in Comparison ===

def print_string_builtin(string):
    """Print string using Python built-in."""
    print(string)


# === Main ===

if __name__ == "__main__":
    text = "Recursion breaks a problem into smaller subproblems"
    print("Print string using recursion:")
    print_string_recursive(text)
    print()
    print()
    print("Print string using built-in:")
    print_string_builtin(text)
```

**Output:**
```
Print string using recursion:
Recursion breaks a problem into smaller subproblems

Print string using built-in:
Recursion breaks a problem into smaller subproblems
```

## Complexity

Each recursive call processes one character and creates a new substring of length $n - 1$:

$$
T(n) = T(n - 1) + O(1), \quad T(0) = O(1)
$$

This gives $O(n)$ time complexity. The space complexity is $O(n)$ for the recursion stack, plus $O(n^2)$ total for the substring copies (since Python string slicing creates new strings).

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
