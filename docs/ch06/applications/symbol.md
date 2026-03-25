# Symbol Tables

Every programming language implementation needs a way to associate **identifiers** (variable names, function names, class names) with their **attributes** (type, scope, memory address, value). A **symbol table** is the data structure that maintains these associations throughout compilation or interpretation. Because identifier lookup is one of the most frequent operations in a compiler, hash tables are the standard implementation, providing $O(1)$ average-case access.

## Operations

A symbol table supports four core operations:

| Operation | Description | Time (hash-based) |
|---|---|---|
| **insert**(name, attributes) | Add a new identifier | $O(1)$ expected |
| **lookup**(name) | Retrieve attributes for an identifier | $O(1)$ expected |
| **update**(name, attributes) | Modify attributes of an existing entry | $O(1)$ expected |
| **delete**(name) | Remove an identifier (scope exit) | $O(1)$ expected |

All operations run in $O(1)$ expected time with a hash table, compared to $O(\log n)$ for a balanced BST or $O(n)$ for an unsorted list.

## Scope and Nested Symbol Tables

Most languages support **lexical scoping**: an identifier declared inside a block is visible only within that block and its nested blocks. Symbol tables handle scoping through one of two strategies.

### Scope Stack

Maintain a **stack of hash tables**, one per scope level. When a new scope opens (e.g., entering a function body or a loop), push a new hash table. When the scope closes, pop the table.

Lookup walks the stack from top to bottom, returning the first match:

$$
\text{lookup}(x) = \text{first } T_i \text{ in stack where } x \in T_i
$$

This takes $O(d)$ time in the worst case where $d$ is the nesting depth, but $d$ is typically small (rarely exceeding 10 in practice).

### Single Table with Scope Chains

Use a single hash table where each entry contains a linked list of definitions, ordered by scope depth. The most recent definition shadows earlier ones. When a scope closes, remove all entries at that depth.

## What Gets Stored

Each symbol table entry typically includes:

| Attribute | Example |
|---|---|
| Name | `"count"` |
| Type | `int`, `float`, `str` |
| Scope level | `0` (global), `1` (function), `2` (block) |
| Memory location | Offset from stack frame base |
| Size | Number of bytes |
| Const/mutable | Whether the binding is immutable |
| Parameters | For functions: parameter types and count |

## Design Considerations

**Hash function choice**: identifiers in real programs tend to share common prefixes (`getWidth`, `getHeight`, `getName`). A good hash function must distribute these similar strings uniformly. Polynomial hashing with a prime base works well:

$$
h(s) = \left(\sum_{i=0}^{|s|-1} s[i] \cdot p^i \right) \bmod m
$$

**Table size**: typical programs contain hundreds to thousands of identifiers. A hash table with a few hundred slots and load factor below $0.75$ handles most cases efficiently.

**String interning**: to avoid repeated string comparisons, many implementations **intern** identifier strings --- storing each unique string once and comparing by pointer/reference equality. The intern table itself is a hash set.

## Python Implementation

```python
"""
Symbol table implementation using hash-table-based scope stack.

Demonstrates how compilers and interpreters manage identifier
bindings across nested lexical scopes.
"""


# === Symbol Table Entry ===

class Symbol:
    """Represents a single identifier's attributes."""

    def __init__(self, name, sym_type, scope_level, value=None):
        self.name = name
        self.sym_type = sym_type
        self.scope_level = scope_level
        self.value = value

    def __repr__(self):
        return (f"Symbol({self.name}, type={self.sym_type}, "
                f"scope={self.scope_level}, value={self.value})")


# === Scoped Symbol Table ===

class SymbolTable:
    """Symbol table with scope stack for nested lexical scoping."""

    def __init__(self):
        self.scopes = [{}]  # stack of hash tables
        self.level = 0

    def enter_scope(self):
        """Push a new scope onto the stack."""
        self.level += 1
        self.scopes.append({})

    def exit_scope(self):
        """Pop the current scope from the stack."""
        if self.level == 0:
            raise RuntimeError("Cannot exit global scope")
        self.scopes.pop()
        self.level -= 1

    def insert(self, name, sym_type, value=None):
        """Insert identifier into the current scope."""
        symbol = Symbol(name, sym_type, self.level, value)
        self.scopes[-1][name] = symbol
        return symbol

    def lookup(self, name):
        """Look up identifier, searching from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_current_scope(self, name):
        """Look up identifier in the current scope only."""
        return self.scopes[-1].get(name)


# === Demonstration ===

if __name__ == "__main__":
    st = SymbolTable()

    # Global scope
    st.insert("x", "int", 10)
    st.insert("pi", "float", 3.14)
    print(f"Global lookup('x'): {st.lookup('x')}")

    # Enter function scope
    st.enter_scope()
    st.insert("x", "int", 42)  # shadows global x
    st.insert("y", "str", "hello")
    print(f"Function lookup('x'): {st.lookup('x')}")
    print(f"Function lookup('pi'): {st.lookup('pi')}")  # found in global

    # Enter block scope
    st.enter_scope()
    st.insert("z", "bool", True)
    print(f"Block lookup('x'): {st.lookup('x')}")  # still function x
    print(f"Block lookup('z'): {st.lookup('z')}")

    # Exit block scope
    st.exit_scope()
    print(f"After block, lookup('z'): {st.lookup('z')}")  # None

    # Exit function scope
    st.exit_scope()
    print(f"After function, lookup('x'): {st.lookup('x')}")  # global x
```

**Output:**
```
Global lookup('x'): Symbol(x, type=int, scope=0, value=10)
Function lookup('x'): Symbol(x, type=int, scope=1, value=42)
Function lookup('pi'): Symbol(pi, type=float, scope=0, value=3.14)
Block lookup('x'): Symbol(x, type=int, scope=1, value=42)
Block lookup('z'): Symbol(z, type=bool, scope=2, value=True)
After block, lookup('z'): None
After function, lookup('x'): Symbol(x, type=int, scope=0, value=10)
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Aho, A. V., Lam, M. S., Sethi, R., and Ullman, J. D. *Compilers: Principles, Techniques, and Tools*, 2nd edition.
