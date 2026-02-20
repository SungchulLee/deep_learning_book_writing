# Properties


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

**Output:**
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

**Output:**
```
None [TreeNode(Laptop), TreeNode(Cell Phone), TreeNode(TV)] 0
```


```python
print(laptop.parent, laptop.children, laptop.get_level())
```

**Output:**
```
Electronics [TreeNode(Mac), TreeNode(Surface), TreeNode(Thinkpad)] 1
```


```python
print(cellphone.parent, cellphone.children, cellphone.get_level())
```

**Output:**
```
Electronics [TreeNode(iPhone), TreeNode(Pixel), TreeNode(Vivo)] 1
```


```python
print(tv.parent, tv.children, tv.get_level())
```

**Output:**
```
Electronics [TreeNode(Samsung), TreeNode(LG)] 1
```


```python
print(samsung.parent, samsung.children, samsung.get_level())
```

**Output:**
```
TV [] 2
```


# Reference

[Tree (General Tree) - Data Structures & Algorithms Tutorials In Python #9](https://www.youtube.com/watch?v=4r_XR9fUPhQ&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=9)
