# Bezout's Identity

**Bezout's Identity** is an important concept in algorithm design and analysis.

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    if b == 0: return a, 1, 0
    g, x, y = extended_gcd(b, a%b)
    return g, y, x - (a//b)*y

print(f"gcd(48,18) = {gcd(48,18)}")
g,x,y = extended_gcd(48,18)
print(f"48*{x} + 18*{y} = {g}")
```

**Output:**
```
gcd(48,18) = 6
48*-1 + 18*3 = 6
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 31](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
