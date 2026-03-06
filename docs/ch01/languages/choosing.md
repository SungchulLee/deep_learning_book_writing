# Choosing a Language

The choice of programming language depends on the context.

## Comparison

$$

\begin{array}{llll}
& \text{Python} & \text{C++} & \text{Java} \\
\hline
\text{Speed} & \text{Slow} & \text{Fast} & \text{Medium} \\
\text{Readability} & \text{High} & \text{Medium} & \text{Medium} \\
\text{Typing} & \text{Dynamic} & \text{Static} & \text{Static} \\
\text{Memory Control} & \text{Auto} & \text{Manual} & \text{Auto (GC)} \\
\text{Best For} & \text{Learning} & \text{Competition} & \text{Enterprise}
\end{array}

$$

## Guidelines

- **Learning algorithms**: Python (clean syntax, focus on logic)
- **Competitive programming**: C++ (fast, STL, short code)
- **Interviews**: Python or the language you're most comfortable with
- **Systems programming**: C++ or Rust

```python
# Python: concise and readable
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

print(f"GCD(48, 18) = {gcd(48, 18)}")
```

**Output:**
```
GCD(48, 18) = 6
```

# Reference

[Competitive Programmer's Handbook — Chapter 1](https://cses.fi/book/book.pdf)
