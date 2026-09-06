# Stack and Queue

Stacks and queues are restricted-access data structures: stacks enforce last-in
first-out (LIFO) order, while queues enforce first-in first-out (FIFO) order. These
restrictions make every operation $O(1)$, which is why stacks and queues appear as
building blocks inside BFS, DFS, expression evaluation, and many other algorithms.

## Stack Operations

A stack supports two primary operations: push (add to top) and pop (remove from top).

| Operation | Array-Based | Linked-List-Based | Notes |
|---|---|---|---|
| Push | $O(1)$ amortized | $O(1)$ | Array may resize |
| Pop | $O(1)$ | $O(1)$ | Remove top element |
| Peek (top) | $O(1)$ | $O(1)$ | Read without removing |
| Is empty | $O(1)$ | $O(1)$ | Check size or head pointer |
| Size | $O(1)$ | $O(1)$ | Maintained as counter |
| Search | $O(n)$ | $O(n)$ | Must traverse |

Both implementations provide $O(1)$ for the core operations. Array-based stacks have
better cache performance; linked-list-based stacks avoid the amortized resize cost.

## Queue Operations

A queue supports enqueue (add to back) and dequeue (remove from front).

| Operation | Circular Array | Linked List | Notes |
|---|---|---|---|
| Enqueue | $O(1)$ amortized | $O(1)$ | Array may resize |
| Dequeue | $O(1)$ | $O(1)$ | Remove front element |
| Peek (front) | $O(1)$ | $O(1)$ | Read without removing |
| Is empty | $O(1)$ | $O(1)$ | Check size or pointers |
| Size | $O(1)$ | $O(1)$ | Maintained as counter |
| Search | $O(n)$ | $O(n)$ | Must traverse |

!!! warning "Naive Array Queue"
    Using a plain array with dequeue shifting all elements gives $O(n)$ per
    dequeue. Always use a circular buffer or linked list to achieve $O(1)$
    dequeue.

## Deque (Double-Ended Queue)

A deque supports insertion and removal at both ends.

| Operation | Circular Array | Doubly Linked List | Notes |
|---|---|---|---|
| Push front | $O(1)$ amortized | $O(1)$ | |
| Push back | $O(1)$ amortized | $O(1)$ | |
| Pop front | $O(1)$ | $O(1)$ | |
| Pop back | $O(1)$ | $O(1)$ | |
| Peek front | $O(1)$ | $O(1)$ | |
| Peek back | $O(1)$ | $O(1)$ | |
| Access by index | $O(1)$ | $O(n)$ | Array advantage |
| Size | $O(1)$ | $O(1)$ | |

A deque can serve as both a stack (use one end) and a queue (use both ends),
making it a versatile building block.

## Specialized Variants

| Variant | Key Operations | Time | Use Case |
|---|---|---|---|
| Min-stack | Push, pop, get-min | $O(1)$ each | Track minimum in $O(1)$ |
| Max-stack | Push, pop, get-max | $O(1)$ each | Track maximum in $O(1)$ |
| Monotonic stack | Push with eviction | $O(n)$ total for $n$ elements | Next greater/smaller element |
| Monotonic deque | Push/pop with eviction | $O(n)$ total for $n$ elements | Sliding window min/max |
| Two-stack queue | Enqueue, dequeue | $O(1)$ amortized | Queue from two stacks |
| Priority queue | Insert, extract-min | $O(\log n)$ | Not FIFO; ordered by priority |

!!! tip "Min-Stack Trick"
    Maintain a second stack that tracks the current minimum. On push, push the
    min of the new element and the current minimum onto the auxiliary stack. On
    pop, pop from both stacks. This gives $O(1)$ get-min with $O(n)$ extra space.

## Space Complexity

| Structure | Implementation | Space | Overhead |
|---|---|---|---|
| Stack (array) | Dynamic array | $O(n)$ | Up to $n$ wasted slots |
| Stack (linked) | Singly linked list | $O(n)$ | 1 pointer per element |
| Queue (circular array) | Circular buffer | $O(n)$ | Up to $n$ wasted slots |
| Queue (linked) | Singly linked list | $O(n)$ | 1 pointer per element |
| Deque (array) | Circular buffer | $O(n)$ | Up to $n$ wasted slots |
| Deque (linked) | Doubly linked list | $O(n)$ | 2 pointers per element |

## Applications and Their Complexities

| Application | Structure | Per-Operation | Total for $n$ Elements |
|---|---|---|---|
| DFS traversal | Stack | $O(1)$ push/pop | $O(V + E)$ |
| BFS traversal | Queue | $O(1)$ enqueue/dequeue | $O(V + E)$ |
| Parenthesis matching | Stack | $O(1)$ per character | $O(n)$ |
| Infix to postfix | Stack | $O(1)$ per token | $O(n)$ |
| Sliding window max | Monotonic deque | $O(1)$ amortized per element | $O(n)$ |
| Next greater element | Monotonic stack | $O(1)$ amortized per element | $O(n)$ |
| Undo/redo | Two stacks | $O(1)$ per action | $O(n)$ |
| Function call stack | Stack | $O(1)$ per call | $O(d)$, $d$ = max depth |

## Language Implementations

| Language | Stack | Queue | Deque |
|---|---|---|---|
| Python | `list` (append/pop) | `collections.deque` | `collections.deque` |
| C++ | `std::stack` | `std::queue` | `std::deque` |
| Java | `ArrayDeque` | `ArrayDeque` or `LinkedList` | `ArrayDeque` |

!!! warning "Python list as Queue"
    Using Python's `list` as a queue with `pop(0)` is $O(n)$ because it shifts
    all elements. Always use `collections.deque` which provides $O(1)$ `popleft()`.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.

## Exercises

**Exercise 1.**
Implement a queue using two stacks. What are the amortized time complexities for enqueue and dequeue?

??? success "Solution to Exercise 1"
    Use stack `in` for enqueue and stack `out` for dequeue. Enqueue: push to `in` ($O(1)$). Dequeue: if `out` is empty, pop all elements from `in` and push to `out` (reversing the order). Pop from `out`. Amortized analysis: each element is pushed to `in` once ($O(1)$), moved from `in` to `out` once ($O(1)$ amortized), and popped from `out` once ($O(1)$). Total amortized cost per element: $O(1)$ for both enqueue and dequeue. The worst-case dequeue is $O(n)$ (when `out` is empty and all $n$ elements must be transferred), but this is amortized over $n$ operations. $\square$

---

**Exercise 2.**
A monotonic deque maintains elements in non-decreasing order. Describe how to use it to find the maximum element in every sliding window of size $k$ in $O(n)$ total time.

??? success "Solution to Exercise 2"
    Maintain a deque storing indices. Invariant: elements at deque indices are in decreasing order. For each element $a[i]$: (1) Remove indices from the back while $a[\text{back}] \le a[i]$ (they can never be the maximum in any future window). (2) Push $i$ to the back. (3) Remove the front if its index is outside the window ($i - \text{front} \ge k$). (4) The front of the deque is the index of the current window maximum. Each element is pushed and popped at most once, so total operations: $O(n)$. The deque always contains the "candidates" for future window maxima in decreasing order. $\square$

---

**Exercise 3.**
Describe how a stack is used to evaluate postfix (Reverse Polish Notation) expressions in $O(n)$ time.

??? success "Solution to Exercise 3"
    Scan the postfix expression left to right. If the token is a number, push it onto the stack. If the token is an operator ($+, -, \times, /$), pop the top two elements ($b$ then $a$), compute $a \mathbin{\text{op}} b$, and push the result. After processing all tokens, the stack contains exactly one element: the result. Time: $O(n)$ where $n$ is the number of tokens (each token is processed once). Space: $O(n)$ for the stack in the worst case (all operands before any operator). Example: "3 4 + 2 *" evaluates as: push 3, push 4, pop 4 and 3, compute 3+4=7, push 7, push 2, pop 2 and 7, compute 7*2=14. Result: 14. $\square$

---

**Exercise 4.**
A min-stack supports push, pop, top, and get-min all in $O(1)$ time. Describe the design.

??? success "Solution to Exercise 4"
    Maintain two stacks: the main stack and a min-stack. The min-stack's top always holds the current minimum of the main stack. **Push(x)**: push $x$ onto the main stack. If $x \le$ min-stack's top (or min-stack is empty), push $x$ onto the min-stack. **Pop**: pop from the main stack. If the popped value equals the min-stack's top, pop from the min-stack too. **Top**: return main stack's top. **Get-min**: return min-stack's top. All operations are $O(1)$. Space: at most $2n$ total across both stacks. The min-stack tracks the "history of minimums": each entry represents the minimum value at some point in the stack's history. When that minimum is popped from the main stack, it is also removed from the min-stack. $\square$

---

**Exercise 5.**
Compare stack-based DFS and queue-based BFS for graph traversal. What properties of the exploration order differ?

??? success "Solution to Exercise 5"
    **Stack-based DFS**: explores as deep as possible before backtracking. Visits nodes in depth-first order. Properties: discovers back edges (cycles), enables topological sorting, strongly connected components, and tree-edge classification. Uses $O(V)$ space (stack depth bounded by $V$). Does not find shortest paths in unweighted graphs. **Queue-based BFS**: explores all neighbors at distance $d$ before distance $d+1$. Visits nodes in breadth-first (level-order). Properties: finds shortest paths in unweighted graphs, discovers nodes layer by layer, enables bipartiteness checking. Uses $O(V)$ space (queue can hold up to $V$ nodes). Both run in $O(V + E)$ time. Choose DFS for structural properties (cycles, components); BFS for shortest paths and level-order exploration. $\square$
