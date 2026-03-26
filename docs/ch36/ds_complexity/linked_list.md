# Linked List Complexities

Linked lists store elements in nodes connected by pointers rather than in contiguous
memory. This design makes insertion and deletion at a known position $O(1)$ but
sacrifices the $O(1)$ random access that arrays provide. Understanding when the
pointer-based trade-off is worthwhile requires knowing the exact complexity of each
operation for each list variant.

## Singly Linked List

Each node stores a value and a pointer to the next node. Traversal is possible only
in the forward direction.

| Operation | Time | Notes |
|---|---|---|
| Access by index | $O(n)$ | Must traverse from head |
| Search | $O(n)$ | Linear scan |
| Insert at head | $O(1)$ | Update head pointer |
| Insert at tail | $O(1)$ with tail pointer, $O(n)$ without | Need to traverse without tail pointer |
| Insert after given node | $O(1)$ | Update one pointer |
| Delete head | $O(1)$ | Update head pointer |
| Delete after given node | $O(1)$ | Update one pointer |
| Delete by value | $O(n)$ | Search + delete |
| Find length | $O(n)$ | Must traverse entire list |
| Reverse | $O(n)$ | Three-pointer technique |

!!! tip "Always Maintain a Tail Pointer"
    If your workload includes frequent appends, maintain both head and tail
    pointers. This converts append from $O(n)$ to $O(1)$ at the cost of one extra
    pointer update during insertion and deletion.

## Doubly Linked List

Each node stores pointers to both the next and previous nodes, enabling bidirectional
traversal and $O(1)$ deletion given a node reference.

| Operation | Time | Notes |
|---|---|---|
| Access by index | $O(n)$ | Can start from either end |
| Search | $O(n)$ | Linear scan in either direction |
| Insert at head | $O(1)$ | Update head + new node's pointers |
| Insert at tail | $O(1)$ | Update tail + new node's pointers |
| Insert before/after given node | $O(1)$ | Update two pairs of pointers |
| Delete given node | $O(1)$ | Update neighbors' pointers |
| Delete by value | $O(n)$ | Search + $O(1)$ delete |
| Reverse | $O(n)$ | Swap next and prev for each node |

The key advantage over singly linked lists is $O(1)$ deletion given a reference to
the node, because the previous node is directly accessible.

## Circular Linked List

The last node points back to the first, forming a ring. This variant supports cyclic
iteration without null checks.

| Operation | Time | Notes |
|---|---|---|
| Traverse all nodes | $O(n)$ | Stop when returning to start |
| Insert after given node | $O(1)$ | Same as singly linked |
| Delete after given node | $O(1)$ | Same as singly linked |
| Search | $O(n)$ | Must detect when full cycle is complete |
| Josephus problem | $O(n^2)$ naive, $O(n)$ with formula | Classic circular list application |

## Space Complexity

| Variant | Space per Node | Total Space | Notes |
|---|---|---|---|
| Singly linked | 1 pointer + data | $O(n)$ | Minimal overhead |
| Doubly linked | 2 pointers + data | $O(n)$ | Extra pointer per node |
| Circular singly | 1 pointer + data | $O(n)$ | Same as singly |
| Circular doubly | 2 pointers + data | $O(n)$ | Same as doubly |
| XOR linked list | 1 XOR-pointer + data | $O(n)$ | Saves one pointer but hard to debug |

## Linked List vs Array

| Operation | Array | Singly Linked | Doubly Linked |
|---|---|---|---|
| Access by index | $O(1)$ | $O(n)$ | $O(n)$ |
| Insert at beginning | $O(n)$ | $O(1)$ | $O(1)$ |
| Insert at end | $O(1)$ amort. | $O(1)$ with tail | $O(1)$ |
| Insert at middle | $O(n)$ | $O(1)$ after search | $O(1)$ after search |
| Delete at beginning | $O(n)$ | $O(1)$ | $O(1)$ |
| Delete given element | $O(n)$ | $O(n)$ | $O(1)$ given node |
| Memory layout | Contiguous | Scattered | Scattered |
| Cache performance | Excellent | Poor | Poor |

!!! warning "Cache Performance"
    Linked lists suffer from poor cache locality because nodes are scattered in
    memory. For small lists ($n < 1000$), arrays are almost always faster in
    practice, even for operations where linked lists have better theoretical
    complexity.

## Common Linked List Algorithms

| Algorithm | Time | Space | Description |
|---|---|---|---|
| Floyd's cycle detection | $O(n)$ | $O(1)$ | Slow/fast pointer to detect cycles |
| Find middle node | $O(n)$ | $O(1)$ | Slow/fast pointer |
| Merge two sorted lists | $O(m + n)$ | $O(1)$ | Two-pointer merge |
| Sort a linked list | $O(n \log n)$ | $O(\log n)$ | Merge sort (preferred for lists) |
| Detect intersection | $O(m + n)$ | $O(1)$ | Two-pointer with length alignment |
| Reverse in groups of $k$ | $O(n)$ | $O(1)$ | Iterative reversal |

## When to Use Linked Lists

| Use Case | Recommended? | Why |
|---|---|---|
| LRU cache | Yes | $O(1)$ move-to-front with doubly linked + hash map |
| Undo/redo history | Yes | $O(1)$ insert and remove at ends |
| Polynomial arithmetic | Yes | Variable-length coefficient lists |
| General-purpose list | No | Arrays are faster due to caching |
| Random access needed | No | $O(n)$ access is prohibitive |
| Small fixed-size data | No | Array overhead is lower |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.
