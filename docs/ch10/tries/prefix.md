# Prefix Search

Search finds a target element by traversing the data structure according to its organizing principle.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self): self.root = TrieNode()
    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children: node.children[c] = TrieNode()
            node = node.children[c]
        node.end = True
    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.children: return False
            node = node.children[c]
        return node.end

t = Trie()
for w in ["apple","app","bat"]: t.insert(w)
for w in ["app","ap","bat","bad"]: print(f"{w}: {t.search(w)}")
```

**Output:**
```
app: True
ap: False
bat: True
bad: False
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
