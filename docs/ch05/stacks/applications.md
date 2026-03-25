# Stack Applications

The LIFO property of stacks makes them uniquely suited for any problem where the most recent piece of information is the most relevant. Compilers use stacks to match parentheses and evaluate expressions. Operating systems use them to manage function calls. Graph algorithms use them for depth-first traversal. This page surveys the most important algorithmic applications of stacks, providing concrete examples and complexity analyses for each.

## String Reversal

Because a stack outputs elements in reverse insertion order, pushing every character of a string onto a stack and then popping them all produces the reversed string. This runs in $O(n)$ time and $O(n)$ space, where $n$ is the length of the string.

```python
"""
Stack applications — common algorithmic uses of the stack data structure.

Demonstrates string reversal, undo mechanism, depth-first search,
and next-greater-element problems, all powered by the LIFO property.
"""


# === Stack Implementation =====================================================

class Stack:
    """Minimal stack for demonstration purposes."""

    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


# === Application 1: String Reversal ==========================================

def reverse_string(text):
    """Reverse a string using a stack.

    Push each character, then pop all — LIFO ordering produces the reversal.
    Time: O(n), Space: O(n).
    """
    stack = Stack()
    for ch in text:
        stack.push(ch)

    result = []
    while not stack.is_empty():
        result.append(stack.pop())
    return "".join(result)


# === Application 2: Undo Mechanism ===========================================

def simulate_undo(actions):
    """Simulate an undo mechanism using a stack.

    Each action is pushed. Undo pops the most recent action.
    Returns the list of undone actions in reverse chronological order.
    """
    stack = Stack()
    for action in actions:
        stack.push(action)
        print(f"  Performed: {action}")

    undone = []
    while not stack.is_empty():
        action = stack.pop()
        undone.append(action)
        print(f"  Undo:      {action}")
    return undone


# === Application 3: Depth-First Search =======================================

def dfs_iterative(graph, start):
    """Iterative depth-first search using an explicit stack.

    Replaces the implicit call stack of recursive DFS with a user-managed
    stack, making the LIFO traversal order explicit.
    Time: O(V + E), Space: O(V).
    """
    visited = set()
    stack = Stack()
    stack.push(start)
    order = []

    while not stack.is_empty():
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            # Push neighbors in reverse order for consistent left-to-right traversal
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.push(neighbor)
    return order


# === Application 4: Next Greater Element =====================================

def next_greater_element(arr):
    """Find the next greater element for each position using a monotonic stack.

    For each element, the next greater element is the first element to its
    right that is strictly larger. Uses a stack to achieve O(n) time
    instead of the naive O(n^2) approach.
    """
    n = len(arr)
    result = [-1] * n
    stack = Stack()  # stores indices

    for i in range(n):
        while not stack.is_empty() and arr[stack.peek()] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.push(i)
    return result


# === Demonstration ============================================================

if __name__ == "__main__":
    # String reversal
    original = "stack"
    print(f"Original: '{original}' → Reversed: '{reverse_string(original)}'")
    print()

    # Undo mechanism
    print("Undo mechanism:")
    actions = ["type 'H'", "type 'i'", "bold text", "insert image"]
    simulate_undo(actions)
    print()

    # Iterative DFS
    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": [],
        "F": [],
    }
    print(f"DFS from 'A': {dfs_iterative(graph, 'A')}")
    print()

    # Next greater element
    arr = [4, 5, 2, 10, 8]
    print(f"Array:                {arr}")
    print(f"Next greater element: {next_greater_element(arr)}")
```

**Output:**
```
Original: 'stack' → Reversed: 'kcats'

Undo mechanism:
  Performed: type 'H'
  Performed: type 'i'
  Performed: bold text
  Performed: insert image
  Undo:      insert image
  Undo:      bold text
  Undo:      type 'i'
  Undo:      type 'H'

DFS from 'A': ['A', 'B', 'D', 'E', 'C', 'F']

Array:                [4, 5, 2, 10, 8]
Next greater element: [5, 10, 10, -1, -1]
```

## Undo and History

Text editors, drawing programs, and version control systems all maintain a history of actions. Pushing each action onto a stack and popping on "undo" naturally reverses actions in the correct order. The time complexity is $O(1)$ per undo or redo operation. Many systems extend this with a second stack for redo: when the user undoes an action, it is popped from the undo stack and pushed onto the redo stack.

## Depth-First Search

Recursive DFS implicitly uses the call stack. An iterative implementation replaces this with an explicit stack, making the memory usage controllable and avoiding stack overflow on deep graphs. The LIFO property ensures that the most recently discovered node is explored first, which is exactly the depth-first strategy. The complexity is $O(V + E)$ time and $O(V)$ space.

## Monotonic Stack Problems

A **monotonic stack** maintains elements in sorted order (either increasing or decreasing). This pattern solves several problems in $O(n)$ time that would otherwise require $O(n^2)$:

- **Next greater element**: for each element, find the first larger element to its right
- **Largest rectangle in histogram**: find the maximum area rectangle formed by consecutive bars
- **Stock span problem**: for each day, count consecutive preceding days with lower or equal prices

The key insight is that each element is pushed and popped at most once, giving $O(n)$ total operations regardless of the input.

## Summary of Applications

| Application | Stack Role | Time | Space |
|---|---|---|---|
| String reversal | Reverse character order | $O(n)$ | $O(n)$ |
| Undo mechanism | Track action history | $O(1)$ per op | $O(n)$ |
| Depth-first search | Track unexplored nodes | $O(V + E)$ | $O(V)$ |
| Next greater element | Maintain monotonic order | $O(n)$ | $O(n)$ |
| Balanced parentheses | Match openers with closers | $O(n)$ | $O(n)$ |
| Expression evaluation | Handle operands and operators | $O(n)$ | $O(n)$ |

Detailed treatments of expression evaluation, infix-to-postfix conversion, balanced parentheses checking, and function call simulation appear on their respective sibling pages.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
