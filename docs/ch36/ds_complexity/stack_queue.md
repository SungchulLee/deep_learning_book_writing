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
