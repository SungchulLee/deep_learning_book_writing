# 이진 트리의 정의
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
이진 트리를 엄밀하게 정의하고 일반 트리와 어떻게 다른지 밝혀라.

??? success "연습문제 1 풀이"
    이진 트리는 비어 있거나, 뿌리 노드와 왼쪽 이진 트리와 오른쪽 이진 트리로 이루어진다. 노드마다 자식이 많아야 2개이고 왼쪽과 오른쪽을 구별한다. 일반 트리는 자식의 수에 제한이 없고 왼쪽과 오른쪽을 구별하지 않는다.

---

**연습문제 2.**
노드가 $n$개인 이진 트리에 널 포인터가 정확히 $n+1$개 있음을 증명하라.

??? success "연습문제 2 풀이"
    노드 $n$개마다 포인터 칸이 2개이므로 포인터는 모두 $2n$개이다. 뿌리가 아닌 노드마다 부모의 포인터가 꼭 하나씩 가리키므로 널이 아닌 포인터는 $n-1$개이다. 따라서 널 포인터는 $2n - (n-1) = n + 1$개이다. $\square$

---

**연습문제 3.**
이진 트리의 레벨 $k$에 있을 수 있는 노드의 최대 개수는 얼마인가?

??? success "연습문제 3 풀이"
    레벨 $k$(뿌리는 레벨 0)에는 많아야 $2^k$개의 노드가 있다. 귀납법으로 따라온다. 레벨 0에는 노드가 1개이고, 레벨 $k$의 노드마다 레벨 $k+1$에 많아야 2개의 노드를 더한다.

---

**연습문제 4.**
포화 이진 트리, 완전 이진 트리, 완벽 이진 트리의 차이를 설명하라.

??? success "연습문제 4 풀이"
    포화: 노드마다 자식이 0개나 2개이다(자식이 하나뿐인 노드가 없다). 완전: 마지막 층만 왼쪽부터 차 있고 나머지 층은 모두 꽉 차 있다. 완벽: 내부 노드마다 자식이 2개이고 모든 잎이 같은 레벨에 있다. 관계: 완벽 $\subset$ 완전, 그리고 완전 $\not\subset$ 포화.
