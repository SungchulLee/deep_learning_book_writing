"""밑바닥 엔진의 버그를 재어 본다 — 그리고 고친다.

이 쪽의 본문 구현은 역전파를 **재귀**로 한다. 노드가 제 경사를
부모에게 넘기고, 부모가 다시 제 부모에게 넘긴다. 쓰기 쉽고 작은
그래프에서는 맞는 답을 준다.

그런데 **한 노드가 여러 곳에 쓰이면** 틀린다. 본문의 「교육용
구현의 한계」가 그것을 설명하는데, 설명만 하고 돌려 보지는 않는다.
여기서 돌려 본다.

틀리는 까닭은 하나다. 두 번째로 방문할 때 **새로 들어온 몫**이
아니라 **지금까지 쌓인 전부**를 부모에게 다시 넘긴다.

실행: python autograd_from_scratch.py
"""


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 본문과 같은 방식 — 재귀로 넘긴다
# ========================================================================

class Recursive:
    """본문 구현을 가장 작게 줄인 것. 더하기와 상수배만 있다."""

    def __init__(self, num, parents=(), op=""):
        self.num = num
        self.grad = None
        self.parents = list(parents)
        self.op = op

    def __add__(self, other):
        # type(self) 로 만들어야 물려받은 클래스가 제 backward 를 쓴다
        return type(self)(self.num + other.num, (self, other), "add")

    def __rmul__(self, k):                       # 상수 * 노드
        return type(self)(k * self.num, (self,), f"mul{k}")

    def backward(self, upstream=1.0):
        self.grad = upstream if self.grad is None else self.grad + upstream
        for p in self.parents:
            if self.op == "add":
                p.backward(self.grad)            # <- 쌓인 전부를 넘긴다
            else:
                p.backward(self.grad * float(self.op[3:]))


# ========================================================================
# 고친 것 — 위상 순서로 한 번씩만
# ========================================================================

class Topological:
    """역위상 순서로 노드마다 한 번씩만 처리한다."""

    def __init__(self, num, parents=(), op=""):
        self.num = num
        self.grad = 0.0
        self.parents = list(parents)
        self.op = op

    def __add__(self, other):
        return type(self)(self.num + other.num, (self, other), "add")

    def __rmul__(self, k):
        return type(self)(k * self.num, (self,), f"mul{k}")

    def _order(self):
        seen, order = set(), []

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node.parents:
                visit(p)
            order.append(node)                   # 부모가 먼저 들어간다

        visit(self)
        return order

    def backward(self):
        order = self._order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):             # 자식부터 처리한다
            for p in node.parents:
                if node.op == "add":
                    p.grad += node.grad
                else:
                    p.grad += node.grad * float(node.op[3:])


# ========================================================================
# 1. 공유가 없으면 둘 다 맞는다
# ========================================================================

def no_sharing():
    header("1. 공유가 없을 때")
    for cls, name in ((Recursive, "재귀"), (Topological, "위상")):
        a = cls(2.0)
        (3 * a + 3 * a).backward()               # 3a 를 두 번 따로 만든다
        print(f"  {name}  L = 3a + 3a (따로)  a.grad = {a.grad}   참값 6")
    print("  두 3a 가 서로 다른 객체라 공유가 아니다")


# ========================================================================
# 2. 공유하면 재귀 쪽이 틀린다
# ========================================================================

def with_sharing():
    header("2. b 를 두 번 쓰면")
    for cls, name in ((Recursive, "재귀"), (Topological, "위상")):
        a = cls(2.0)
        b = 3 * a
        (b + b).backward()
        print(f"  {name}  L = b + b, b = 3a   a.grad = {a.grad}   참값 6")
    print("  재귀 쪽이 9 다. 두 번째 방문에서 b.grad=2 를 통째로 다시 넘긴다")
    print("  3*1 + 3*2 = 9 — 첫 몫이 두 번 세어졌다")


# ========================================================================
# 3. 깊어지면 더 벌어진다
# ========================================================================

def deeper():
    header("3. 더 깊이 공유하면")
    for cls, name in ((Recursive, "재귀"), (Topological, "위상")):
        a = cls(2.0)
        b = 3 * a
        c = b + b
        d = c + c
        d.backward()
        print(f"  {name}  d = (b+b)+(b+b)   a.grad = {a.grad}   참값 12")
    print("  재귀 쪽이 39 다. 틀린 정도가 깊이에 따라 커진다")


# ========================================================================
# 4. 왜 작은 보기에서는 맞았는가
# ========================================================================

def why_it_passed():
    header("4. 본문의 보기들은 왜 맞았는가")
    print("  본문 1~3부의 식들은 중간 노드를 한 번씩만 쓴다")
    print("    c = 4a + 3        a 는 한 번")
    print("    e = (4a+3)(a+2)   a 가 두 번이지만 잎이라 더 넘길 데가 없다")
    print("  잎에서 쌓이는 것은 맞는 동작이다 — 경사 누적")
    print("  틀리는 것은 **중간 노드**가 공유될 때뿐이다")
    print("  그래서 PyTorch 와 대조한 검사들이 전부 통과했다")
    print("  통과한 검사가 구현의 일반적 정확성을 보증하지 않는다")


# ========================================================================
# 5. 고침의 핵심
# ========================================================================

def the_fix():
    header("5. 고치는 두 가지 길")
    print("  (가) 위상 순서 — 자식을 다 처리한 뒤에 부모로 넘긴다")
    print("       노드마다 정확히 한 번씩만 넘기므로 중복이 없다")
    print("       파이토치가 쓰는 방법이기도 하다")
    print("  (나) 새로 들어온 몫만 넘기기 — upstream 을 그대로 쓴다")
    print("       재귀를 유지할 수 있지만 같은 노드를 여러 번 방문한다")
    print("       그래프가 깊고 넓으면 방문 횟수가 지수로 는다")
    a = Recursive(2.0)
    b = 3 * a
    (b + b).backward()
    print(f"  (나) 를 본문 코드에 적용하려면 p.backward(self.grad) 를")
    print(f"       p.backward(upstream) 으로 고치면 된다")


def main():
    no_sharing()
    with_sharing()
    deeper()
    why_it_passed()
    the_fix()


if __name__ == "__main__":
    main()
