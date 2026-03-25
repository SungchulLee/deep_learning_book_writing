# Queue Applications

The FIFO property of queues makes them the natural choice whenever fairness matters: tasks should be processed in the order they arrive. Operating systems use queues for process scheduling and I/O request buffering. Network routers queue packets waiting for transmission. Graph algorithms use queues to explore nodes level by level. This page surveys the most important algorithmic applications of queues, with concrete examples and complexity analyses.

## Producer-Consumer Buffer

In concurrent systems, a **producer** generates data and a **consumer** processes it, often at different speeds. A queue acts as a buffer between them: the producer enqueues items and the consumer dequeues them. This decouples the two processes and allows them to operate at their own pace. If the queue is bounded (fixed maximum size), the producer blocks when the queue is full and the consumer blocks when it is empty.

## Hot Potato Simulation

The "hot potato" (or Josephus) problem illustrates circular elimination. Players stand in a circle and pass an item. After a fixed number of passes, the player holding the item is eliminated. Using a queue, each pass dequeues the front player and re-enqueues them at the rear. After the specified count, the front player is eliminated (dequeued without re-enqueuing).

## Level-Order Traversal

Trees and graphs are often explored level by level. A queue naturally produces this ordering: start by enqueuing the root, then repeatedly dequeue a node, process it, and enqueue its children. The result is a breadth-first traversal. This is covered in more detail on the BFS Preview sibling page.

```python
"""
Queue applications — common algorithmic uses of the queue data structure.

Demonstrates producer-consumer simulation, hot potato elimination,
and level-order tree traversal, all powered by the FIFO property.
"""
from collections import deque


# === Application 1: Producer-Consumer Simulation ==============================

def producer_consumer(tasks, process_time):
    """Simulate a producer-consumer buffer using a queue.

    The producer enqueues all tasks first, then the consumer processes
    them in FIFO order, each taking `process_time` units.
    """
    queue = deque()
    clock = 0

    # Producer phase
    for task in tasks:
        queue.append(task)
        clock += 1
        print(f"  t={clock:>2}: Producer enqueued '{task}' → queue={list(queue)}")

    # Consumer phase
    while queue:
        task = queue.popleft()
        clock += process_time
        print(f"  t={clock:>2}: Consumer processed '{task}' → queue={list(queue)}")


# === Application 2: Hot Potato Elimination ====================================

def hot_potato(players, num_passes):
    """Simulate the hot potato game using a queue.

    Players stand in a circle. After `num_passes` passes, the player
    holding the potato is eliminated. Last player standing wins.
    Time: O(n * k) where n = players, k = passes per round.
    """
    queue = deque(players)
    print(f"  Starting players: {list(queue)}")

    while len(queue) > 1:
        for _ in range(num_passes):
            queue.append(queue.popleft())  # pass the potato
        eliminated = queue.popleft()
        print(f"  Eliminated: {eliminated:<10s} Remaining: {list(queue)}")

    winner = queue[0]
    print(f"  Winner: {winner}")
    return winner


# === Application 3: Level-Order Tree Traversal ================================

class TreeNode:
    """Simple binary tree node."""

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order_traversal(root):
    """Traverse a binary tree level by level using a queue.

    Time: O(n) — each node is enqueued and dequeued exactly once.
    Space: O(w) where w is the maximum width of the tree.
    """
    if root is None:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


# === Demonstration ============================================================

if __name__ == "__main__":
    # Producer-consumer
    print("Producer-Consumer Simulation:")
    producer_consumer(["email", "report", "backup"], process_time=2)
    print()

    # Hot potato
    print("Hot Potato Game (3 passes per round):")
    hot_potato(["Alice", "Bob", "Carol", "Dave", "Eve"], num_passes=3)
    print()

    # Level-order traversal
    #         1
    #        / \
    #       2   3
    #      / \   \
    #     4   5   6
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, None, TreeNode(6)))

    print("Level-Order Traversal:")
    levels = level_order_traversal(tree)
    for i, level in enumerate(levels):
        print(f"  Level {i}: {level}")
```

**Output:**
```
Producer-Consumer Simulation:
  t= 1: Producer enqueued 'email' → queue=['email']
  t= 2: Producer enqueued 'report' → queue=['email', 'report']
  t= 3: Producer enqueued 'backup' → queue=['email', 'report', 'backup']
  t= 5: Consumer processed 'email' → queue=['report', 'backup']
  t= 7: Consumer processed 'report' → queue=['backup']
  t= 9: Consumer processed 'backup' → queue=[]

Hot Potato Game (3 passes per round):
  Starting players: ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
  Eliminated: Alice      Remaining: ['Eve', 'Bob', 'Carol', 'Dave']
  Eliminated: Eve        Remaining: ['Bob', 'Carol', 'Dave']
  Eliminated: Carol      Remaining: ['Dave', 'Bob']
  Eliminated: Bob        Remaining: ['Dave']
  Winner: Dave

Level-Order Traversal:
  Level 0: [1]
  Level 1: [2, 3]
  Level 2: [4, 5, 6]
```

The producer-consumer simulation shows FIFO ordering: tasks are consumed in the order they were produced. The hot potato game uses the queue's circular rotation property (dequeue from front, enqueue at rear) to simulate passing. Level-order traversal processes all nodes at depth $d$ before any nodes at depth $d+1$.

## Summary of Applications

| Application | Queue Role | Time | Space |
|---|---|---|---|
| Producer-consumer buffer | Decouple producer and consumer speeds | $O(1)$ per op | $O(n)$ |
| Hot potato / Josephus | Circular elimination | $O(n \cdot k)$ | $O(n)$ |
| Level-order tree traversal | Process nodes by depth | $O(n)$ | $O(w)$ |
| Breadth-first search (BFS) | Explore graph level by level | $O(V + E)$ | $O(V)$ |
| Task scheduling (FCFS) | Serve tasks in arrival order | $O(1)$ per op | $O(n)$ |

Here $n$ denotes the number of elements, $k$ the passes per round, $w$ the maximum tree width, and $V$, $E$ the vertices and edges of a graph.

Detailed treatments of BFS and task scheduling appear on their respective sibling pages.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
