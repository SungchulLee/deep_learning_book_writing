# Chaining

When two keys hash to the same slot, a collision occurs. Chaining resolves collisions by storing all keys that map to the same index in a linked list (or another secondary data structure) attached to that slot. Because every key has a home regardless of how many collisions occur, chaining never runs out of space in the table itself --- overflow simply lengthens the chain.

## Mechanism

A hash table of size $m$ maintains an array $T[0 \ldots m-1]$ where each entry points to the head of a linked list. For a hash function $h$, the three fundamental operations work as follows.

**Insert** key $k$: compute $h(k)$, then prepend $k$ to the list at $T[h(k)]$. This takes $O(1)$ worst-case time (assuming duplicate checks are not required).

**Search** for key $k$: compute $h(k)$, then traverse the list at $T[h(k)]$ looking for $k$. The time depends on the length of that chain.

**Delete** key $k$: search for $k$ in the chain at $T[h(k)]$ and remove the node. With a doubly-linked list the deletion itself is $O(1)$ once the node is found.

## Simple Uniform Hashing Assumption

The analysis of chaining relies on the **simple uniform hashing assumption (SUHA)**: each key is equally likely to hash to any of the $m$ slots, independently of where other keys hash. Under SUHA, for $n$ keys stored in a table of size $m$, the **load factor** is

$$
\alpha = \frac{n}{m}
$$

and the expected length of each chain is $\alpha$.

## Expected Search Time

### Unsuccessful Search

An unsuccessful search examines every element in the chain at $T[h(k)]$. Under SUHA the expected chain length is $\alpha$, so the expected time including the hash computation is

$$
\Theta(1 + \alpha)
$$

### Successful Search

A successful search for a key $k$ examines, on average, half the elements that were inserted into the same chain after $k$ was inserted plus one (for $k$ itself). Summing over all $n$ keys and averaging gives

$$
\Theta\!\left(1 + \frac{\alpha}{2}\right) = \Theta(1 + \alpha)
$$

### Constant-Time Operations

When the number of slots $m$ is chosen proportional to the number of keys $n$, the load factor $\alpha = n/m = O(1)$. In this regime, all dictionary operations --- insert, search, and delete --- run in $O(1)$ expected time.

## Worst-Case Behavior

If all $n$ keys hash to the same slot, the single chain has length $n$ and every search degenerates to a linear scan. The worst-case time for search and delete is therefore

$$
\Theta(n)
$$

This pathological case motivates the use of universal hashing or other randomized hash families to make worst-case collisions unlikely.

## Chain Data Structure Choices

Although linked lists are the standard choice, any dynamic set structure can serve as the chain.

| Chain structure | Search | Insert | Delete | Cache behavior |
|---|---|---|---|---|
| Singly-linked list | $O(\ell)$ | $O(1)$ | $O(\ell)$ | Poor |
| Doubly-linked list | $O(\ell)$ | $O(1)$ | $O(1)$* | Poor |
| Dynamic array | $O(\ell)$ | $O(1)$ amortized | $O(\ell)$ | Good |
| Balanced BST | $O(\log \ell)$ | $O(\log \ell)$ | $O(\log \ell)$ | Moderate |

Here $\ell$ denotes the chain length and $^*$assumes the node pointer is already known.

Using balanced BSTs as chains guarantees $O(\log n)$ worst-case search even without uniformity assumptions, at the cost of higher constant factors. Java's `HashMap` switches from linked lists to red-black trees when a chain exceeds eight elements.

## Python Implementation

```python
"""
Chaining-based hash table implementation.

Demonstrates collision resolution by chaining, where each slot
in the table holds a linked list of (key, value) pairs.
"""


# === Node and Linked List ===

class Node:
    """A node in a singly-linked chain."""

    __slots__ = ("key", "value", "next")

    def __init__(self, key, value, next_node=None):
        self.key = key
        self.value = value
        self.next = next_node


# === Hash Table with Chaining ===

class ChainingHashTable:
    """Hash table that resolves collisions via chaining."""

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        """Insert or update a key-value pair."""
        idx = self._hash(key)
        node = self.table[idx]
        while node is not None:
            if node.key == key:
                node.value = value  # update existing
                return
            node = node.next
        # Prepend new node (O(1) insertion)
        self.table[idx] = Node(key, value, self.table[idx])
        self.size += 1

    def search(self, key):
        """Return the value for key, or None if not found."""
        idx = self._hash(key)
        node = self.table[idx]
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        return None

    def delete(self, key):
        """Remove key from the table. Return True if found."""
        idx = self._hash(key)
        prev, node = None, self.table[idx]
        while node is not None:
            if node.key == key:
                if prev is None:
                    self.table[idx] = node.next
                else:
                    prev.next = node.next
                self.size -= 1
                return True
            prev, node = node, node.next
        return False

    def load_factor(self):
        """Return the current load factor alpha = n / m."""
        return self.size / self.capacity

    def chain_lengths(self):
        """Return a list of chain lengths for inspection."""
        lengths = []
        for head in self.table:
            length, node = 0, head
            while node is not None:
                length += 1
                node = node.next
            lengths.append(length)
        return lengths


# === Demonstration ===

if __name__ == "__main__":
    ht = ChainingHashTable(capacity=4)

    # Insert several keys (collisions are likely with capacity=4)
    for key, val in [("apple", 1), ("banana", 2), ("cherry", 3),
                     ("date", 4), ("elderberry", 5), ("fig", 6)]:
        ht.insert(key, val)

    print(f"Load factor: {ht.load_factor():.2f}")
    print(f"Chain lengths: {ht.chain_lengths()}")
    print(f"Search 'cherry': {ht.search('cherry')}")
    print(f"Search 'grape': {ht.search('grape')}")

    ht.delete("banana")
    print(f"After deleting 'banana': {ht.search('banana')}")
    print(f"Chain lengths: {ht.chain_lengths()}")
```

**Output:**
```
Load factor: 1.50
Chain lengths: [1, 1, 2, 2]
Search 'cherry': 3
Search 'grape': None
After deleting 'banana': None
Chain lengths: [1, 0, 2, 2]
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Hashing Technique - Simplified](https://www.youtube.com/watch?v=mFY0J5W8Udk&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=79)
