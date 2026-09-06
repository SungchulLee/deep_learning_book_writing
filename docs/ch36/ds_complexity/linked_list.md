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

## Exercises

**Exercise 1.**
Compare singly-linked lists, doubly-linked lists, and arrays for insertion at the front, insertion at the back, deletion by value, and random access.

??? success "Solution to Exercise 1"
    | Operation | Singly LL | Doubly LL | Array |
    |---|---|---|---|
    | Insert front | $O(1)$ | $O(1)$ | $O(n)$ |
    | Insert back | $O(n)$* | $O(1)$** | $O(1)$ amortized |
    | Delete by value | $O(n)$ | $O(n)$ search + $O(1)$ unlink | $O(n)$ |
    | Random access | $O(n)$ | $O(n)$ | $O(1)$ |

    *$O(1)$ with a tail pointer. **With a tail pointer. Linked lists win for front insertion and deletion at known positions. Arrays win for random access and cache performance. $\square$

---

**Exercise 2.**
Explain why linked list deletion given a pointer to the node is $O(1)$ for doubly-linked lists but $O(n)$ for singly-linked lists.

??? success "Solution to Exercise 2"
    To delete a node, we must update the predecessor's `next` pointer. In a doubly-linked list, the node has a `prev` pointer, so the predecessor is accessed in $O(1)$: set `node.prev.next = node.next` and `node.next.prev = node.prev`. In a singly-linked list, there is no `prev` pointer. Finding the predecessor requires traversing from the head until reaching a node whose `next` is the target: $O(n)$ in the worst case. A trick for singly-linked lists: copy the next node's value into the target node and delete the next node. This is $O(1)$ but fails for the last node (no next node to copy). $\square$

---

**Exercise 3.**
Describe how a linked list is used to implement an LRU cache with $O(1)$ operations.

??? success "Solution to Exercise 3"
    Use a doubly-linked list combined with a hash map. The list maintains access order: most recently accessed at the head, least recently at the tail. The hash map maps keys to list nodes. `get(key)`: look up the node in the hash map ($O(1)$). Move the node to the head of the list ($O(1)$ pointer operations). Return the value. `put(key, value)`: if key exists, update and move to head. If new, create a node at the head, add to hash map. If over capacity, remove the tail node from the list and hash map ($O(1)$). All operations are $O(1)$. $\square$

---

**Exercise 4.**
Prove that detecting a cycle in a linked list can be done in $O(n)$ time and $O(1)$ space using Floyd's tortoise and hare algorithm.

??? success "Solution to Exercise 4"
    Use two pointers: slow (advances 1 step) and fast (advances 2 steps). If there is no cycle, fast reaches null in $O(n)$ steps. If there is a cycle of length $c$ starting at position $\mu$: once both pointers enter the cycle, the fast pointer gains 1 step per iteration on the slow pointer. Since the cycle has $c$ nodes, they meet within $c$ steps after slow enters the cycle. Time: slow travels at most $\mu + c$ steps before meeting; fast travels at most $2(\mu + c)$ steps. Both are $O(n)$ since $\mu + c \le n$. Space: only two pointer variables, so $O(1)$. After detecting the cycle, finding its start: reset slow to head. Advance both pointers 1 step at a time. They meet at the cycle start after $\mu$ steps (proof: their distance modulo $c$ equals 0 when slow has traveled $\mu$ steps). $\square$

---

**Exercise 5.**
When should you use a linked list instead of a dynamic array in practice? Give two scenarios where linked lists provide a genuine advantage.

??? success "Solution to Exercise 5"
    (1) **Constant-time splicing**: merging or splitting lists at known positions is $O(1)$ with linked lists (redirect pointers) but $O(n)$ with arrays (copy elements). Example: a text editor's undo buffer, where paragraphs are frequently moved, split, or merged. (2) **Guaranteed $O(1)$ insertion/deletion without amortization**: linked lists never resize or copy. Arrays have $O(n)$ worst-case insertion (during resize). In real-time systems where worst-case latency matters, linked lists provide predictable $O(1)$ operations. In most other cases, arrays are superior due to cache performance. Modern practice often uses arrays (or deques) even when linked lists have better theoretical bounds, because the 10--20x cache performance advantage of arrays outweighs the occasional $O(n)$ resize. $\square$
