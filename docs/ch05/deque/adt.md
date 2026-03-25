# Deque ADT

Stacks restrict access to one end, and queues restrict insertion and deletion to opposite ends.  Many algorithms, however, need the flexibility to add or remove elements at *both* ends efficiently.  A **deque** (double-ended queue, pronounced "deck") generalizes both data structures by allowing constant-time insertion and deletion at the front and the back.  This page defines the deque as an abstract data type, specifies its operations, and establishes the time-complexity contract that every concrete implementation must satisfy.

## Definition

A **deque** is a linear collection of elements that supports insertion and removal at both the front and the back.  Formally, a deque $D$ maintains an ordered sequence of elements

$$
D = \langle d_0, d_1, \dots, d_{n-1} \rangle
$$

where $d_0$ is the **front** element and $d_{n-1}$ is the **back** element.  The integer $n = |D|$ is the **size** of the deque.

## Operations

Every deque implementation must provide the following operations with the stated worst-case or amortized time complexities.

| Operation | Description | Time |
|---|---|---|
| `push_front(x)` | Insert element $x$ at the front | $O(1)$ |
| `push_back(x)` | Insert element $x$ at the back | $O(1)$ |
| `pop_front()` | Remove and return the front element | $O(1)$ |
| `pop_back()` | Remove and return the back element | $O(1)$ |
| `front()` | Return the front element without removing it | $O(1)$ |
| `back()` | Return the back element without removing it | $O(1)$ |
| `is_empty()` | Return `True` if the deque has no elements | $O(1)$ |
| `size()` | Return the number of elements | $O(1)$ |

!!! warning "Preconditions"
    `pop_front()`, `pop_back()`, `front()`, and `back()` require the deque to be non-empty.  Calling these on an empty deque is undefined behavior (or raises an exception, depending on the implementation).

## Relationship to Stacks and Queues

A deque subsumes both the stack ADT and the queue ADT:

- **Stack behavior** (LIFO): use only `push_back` and `pop_back` (or only `push_front` and `pop_front`).
- **Queue behavior** (FIFO): use `push_back` for enqueue and `pop_front` for dequeue (or vice versa).

Because every stack operation and every queue operation can be expressed as a deque operation with the same $O(1)$ cost, any deque implementation automatically provides a correct and efficient stack and queue.

??? example "Deque as a stack and as a queue"
    Consider the sequence of operations on an initially empty deque $D$:

    | Step | Operation | Deque state | Returned |
    |------|-----------|-------------|----------|
    | 1 | `push_back(10)` | $\langle 10 \rangle$ | — |
    | 2 | `push_back(20)` | $\langle 10, 20 \rangle$ | — |
    | 3 | `push_back(30)` | $\langle 10, 20, 30 \rangle$ | — |
    | 4 | `pop_back()` | $\langle 10, 20 \rangle$ | 30 |
    | 5 | `pop_front()` | $\langle 20 \rangle$ | 10 |

    Steps 1--4 behave like a stack (LIFO on the back).  Step 5 demonstrates the extra power of a deque: removing from the front, which a plain stack cannot do.

## Common Use Cases

Deques appear naturally in algorithms that process elements from both ends:

- **Sliding window maximum/minimum**: maintaining candidates that may leave from the front (expired) or the back (dominated).  See the [sliding window page](sliding_window.md) for a detailed treatment.
- **Work-stealing schedulers**: idle threads steal tasks from the opposite end of a busy thread's deque.
- **Palindrome checking**: compare characters removed from the front and back simultaneously.
- **BFS variants**: algorithms like 0-1 BFS push to the front or back depending on edge weight.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
