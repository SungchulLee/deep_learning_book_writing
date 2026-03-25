# Print String Reversely

Printing a string in reverse order using recursion demonstrates a subtle but important variation of the linear recursion pattern. Instead of processing the first character and recursing on the rest, the recursive call can be made *before* printing — producing output in reverse. This "print after recursion" pattern appears throughout recursive algorithms where post-order processing is needed.

## Recursive Structure

- **Base case**: the string is empty — nothing to print, return
- **Recursive case**: print the last character, then recursively print the remaining prefix in reverse

An alternative approach reverses the processing order by printing `string[-1]` and recursing on `string[:-1]`:

```python
"""Print a string in reverse using recursion."""


# === Recursive Reverse Print ===

def print_reverse_recursive(string):
    """Print each character of string in reverse using recursion."""
    if string == "":
        return
    print(string[-1], end="")
    print_reverse_recursive(string[:-1])


# === Built-in Comparison ===

def print_reverse_builtin(string):
    """Print reversed string using Python slicing."""
    print(string[::-1])


# === Main ===

if __name__ == "__main__":
    text = "Recursion breaks a problem into smaller subproblems"
    print("Print string reversely using recursion:")
    print_reverse_recursive(text)
    print()
    print()
    print("Print string reversely using built-in:")
    print_reverse_builtin(text)
```

**Output:**
```
Print string reversely using recursion:
smelborpbus rellams otni melborp a skaerb noisruceR

Print string reversely using built-in:
smelborpbus rellams otni melborp a skaerb noisruceR
```

## Complexity

The recurrence is the same as forward printing:

$$
T(n) = T(n - 1) + O(1), \quad T(0) = O(1)
$$

This gives $O(n)$ time complexity with $O(n)$ stack space. The substring slicing also contributes $O(n^2)$ total work for creating intermediate strings.

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
