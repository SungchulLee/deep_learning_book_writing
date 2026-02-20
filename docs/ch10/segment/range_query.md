# Range Queries

**Range Queries** is an important concept in algorithm design and analysis.

```python
class SegTree:
    def __init__(self, n):
        self.n = n; self.tree = [0]*(4*n)
    def update(self, node, lo, hi, idx, val):
        if lo == hi: self.tree[node] = val; return
        mid = (lo+hi)//2
        if idx <= mid: self.update(2*node,lo,mid,idx,val)
        else: self.update(2*node+1,mid+1,hi,idx,val)
        self.tree[node] = self.tree[2*node]+self.tree[2*node+1]
    def query(self, node, lo, hi, l, r):
        if r < lo or hi < l: return 0
        if l <= lo and hi <= r: return self.tree[node]
        mid = (lo+hi)//2
        return self.query(2*node,lo,mid,l,r)+self.query(2*node+1,mid+1,hi,l,r)

st = SegTree(5)
for i,v in enumerate([1,3,5,7,9]): st.update(1,0,4,i,v)
print(f"Sum [1,3]: {st.query(1,0,4,1,3)}")
print(f"Sum [0,4]: {st.query(1,0,4,0,4)}")
```

**Output:**
```
Sum [1,3]: 15
Sum [0,4]: 25
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
