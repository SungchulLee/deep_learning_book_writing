"""
Binary Tree Definition
"""

# ======================================================================

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


# === Main ===
if __name__ == "__main__":
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


    print(root.parent, root.children, root.get_level())


    print(laptop.parent, laptop.children, laptop.get_level())


    print(cellphone.parent, cellphone.children, cellphone.get_level())


    print(tv.parent, tv.children, tv.get_level())


    print(samsung.parent, samsung.children, samsung.get_level())
