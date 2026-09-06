# 성질
```python
class TreeNode:
    
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None
        
    def __repr__(self):
        return f'TreeNode({self.data})'
        
    def __str__(self):
        return self.data
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)        

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_tree()
```

```python
root = TreeNode("Electronics")

laptop = TreeNode("Laptop")
mac = TreeNode("Mac")
surface = TreeNode("Surface")
thinkpad = TreeNode("Thinkpad")
laptop.add_child(mac)
laptop.add_child(surface)
laptop.add_child(thinkpad)

cellphone = TreeNode("Cell Phone")
iphone = TreeNode("iPhone")
pixel = TreeNode("Pixel")
vivo =TreeNode("Vivo")
cellphone.add_child(iphone)
cellphone.add_child(pixel)
cellphone.add_child(vivo)

tv = TreeNode("TV")
samsung = TreeNode("Samsung")
lg = TreeNode("LG")
tv.add_child(samsung)
tv.add_child(lg)

root.add_child(laptop)
root.add_child(cellphone)
root.add_child(tv)

root.print_tree()
```

**출력:**
```
Electronics
   |__Laptop
      |__Mac
      |__Surface
      |__Thinkpad
   |__Cell Phone
      |__iPhone
      |__Pixel
      |__Vivo
   |__TV
      |__Samsung
      |__LG
```

```python
print(root.parent, root.children, root.get_level())
```

**출력:**
```
None [TreeNode(Laptop), TreeNode(Cell Phone), TreeNode(TV)] 0
```

```python
print(laptop.parent, laptop.children, laptop.get_level())
```

**출력:**
```
Electronics [TreeNode(Mac), TreeNode(Surface), TreeNode(Thinkpad)] 1
```

```python
print(cellphone.parent, cellphone.children, cellphone.get_level())
```

**출력:**
```
Electronics [TreeNode(iPhone), TreeNode(Pixel), TreeNode(Vivo)] 1
```

```python
print(tv.parent, tv.children, tv.get_level())
```

**출력:**
```
Electronics [TreeNode(Samsung), TreeNode(LG)] 1
```

```python
print(samsung.parent, samsung.children, samsung.get_level())
```

**출력:**
```
TV [] 2
```

# 참고 문헌

[Tree (General Tree) - Data Structures & Algorithms Tutorials In Python #9](https://www.youtube.com/watch?v=4r_XR9fUPhQ&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=9)

## 연습문제

**연습문제 1.**
내부 노드가 $n$개인 (포화) 이진 트리에 잎이 $n+1$개 있음을 증명하라.

??? success "연습문제 1 풀이"
    포화 이진 트리에서는 내부 노드마다 자식이 꼭 2개이다. $I$을 내부 노드의 수, $L$을 잎의 수라 하자. 전체 노드는 $I + L$개이다. 전체 변은 $I + L - 1$개이다(뿌리가 아닌 노드마다 들어오는 변이 하나씩 있다). 한편 내부 노드마다 변을 2개씩 내보내므로 나가는 변은 $2I$개이다. 따라서 $2I = I + L - 1$이고 $L = I + 1$이다. $\square$

---

**연습문제 2.**
노드가 $n$개인 완전 이진 트리의 높이는 얼마인가?

??? success "연습문제 2 풀이"
    $h = \lfloor\log_2 n\rfloor$이다. 높이가 $h$인 완전 트리의 노드 수는 $2^h$개 이상 $2^{h+1}-1$개 이하이다. $2^h \leq n < 2^{h+1}$을 $h$에 대해 풀면 $h = \lfloor\log_2 n\rfloor$이다.

---

**연습문제 3.**
노드가 $n$개인 이진 트리의 최소 높이가 $\lceil\log_2(n+1)\rceil - 1$임을 증명하라.

??? success "연습문제 3 풀이"
    높이가 $h$일 때 노드는 많아야 $2^{h+1} - 1$개이다(완벽 트리). $n \leq 2^{h+1} - 1$이어야 하므로 $h \geq \lceil\log_2(n+1)\rceil - 1$이다. 이 최솟값은 완전 이진 트리에서 이루어진다.

---

**연습문제 4.**
노드가 $n$개인 구조적으로 서로 다른 이진 트리는 몇 개인가? 카탈랑 수로 나타내어라.

??? success "연습문제 4 풀이"
    답은 $n$번째 카탈랑 수 $C_n = \frac{1}{n+1}\binom{2n}{n} = \frac{(2n)!}{(n+1)!n!}$이다. $n = 3$이면 $C_3 = 5$개의 서로 다른 이진 트리가 있다.
