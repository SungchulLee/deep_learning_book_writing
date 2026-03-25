# Static Arrays

Arrays are the simplest and most widely used data structure in computing. A **static array** allocates a fixed number of elements in a single contiguous block of memory at creation time. Because every element occupies the same number of bytes and sits next to its neighbors, any element can be reached in constant time through a single arithmetic calculation. This predictable memory layout also makes static arrays exceptionally cache-friendly, which is why they remain the backbone of high-performance numerical computing and the internal storage behind many higher-level data structures.

## Memory Layout

A static array of $n$ elements, each occupying $w$ bytes, is stored as a single contiguous block of $n \cdot w$ bytes starting at some base address $b$. The address of the element at index $i$ (using zero-based indexing) is computed by the formula

$$
\text{addr}(i) = b + i \cdot w
$$

where $0 \le i \le n - 1$. Because this formula uses only a multiplication and an addition, both of which execute in constant time on modern hardware, accessing any element takes $O(1)$ time regardless of the array's size.

??? example "Address Computation Example"

    Consider an array of 5 integers, each 4 bytes wide, starting at base address 1000:

    | Index $i$ | Address $= 1000 + i \times 4$ | Value |
    |-----------|-------------------------------|-------|
    | 0         | 1000                          | 10    |
    | 1         | 1004                          | 20    |
    | 2         | 1008                          | 30    |
    | 3         | 1012                          | 40    |
    | 4         | 1016                          | 50    |

    To access the element at index 3, the hardware computes $1000 + 3 \times 4 = 1012$ and reads 4 bytes starting at that address.

## Declaration and Initialization

In languages like C, a static array is declared with a fixed size known at compile time. Python does not have a built-in static array type, but the `array` module and NumPy's `ndarray` provide fixed-type contiguous storage that behaves similarly.

```python
"""Static array operations using Python's array module and ctypes."""

import array
import ctypes

# === Fixed-type array using the array module ===
# 'i' specifies signed 32-bit integers
int_array = array.array('i', [10, 20, 30, 40, 50])

# O(1) access by index
print(f"Element at index 2: {int_array[2]}")   # 30
print(f"Element at index 4: {int_array[4]}")   # 50

# === Low-level static allocation using ctypes ===
# Allocate exactly 5 integers — size is fixed at creation
ArrayType = ctypes.c_int * 5
static = ArrayType(10, 20, 30, 40, 50)

print(f"ctypes element at index 3: {static[3]}")  # 40
```

**Output:**
```
Element at index 2: 30
Element at index 4: 50
ctypes element at index 3: 40
```

## Operations and Complexity

The fixed size of a static array constrains which operations are efficient. Reading or writing at a known index is instantaneous, but operations that change the logical size or shift elements require linear work.

| Operation         | Time Complexity | Description                                       |
|-------------------|-----------------|---------------------------------------------------|
| Access by index   | $O(1)$          | Direct address computation                        |
| Update by index   | $O(1)$          | Write to a computed address                       |
| Search (unsorted) | $O(n)$          | Must scan every element in the worst case         |
| Search (sorted)   | $O(\log n)$     | Binary search on a sorted array                   |
| Insert at end     | $O(1)$          | Only if there is unused capacity                  |
| Insert at index   | $O(n)$          | Must shift all subsequent elements right           |
| Delete at index   | $O(n)$          | Must shift all subsequent elements left            |

!!! warning "No Built-in Resizing"

    A static array cannot grow or shrink after creation. Attempting to add a sixth element to a five-element array requires allocating a new, larger array and copying all existing elements. This limitation motivates the **dynamic array** (covered in the next section), which automates resizing with amortized constant-time appends.

## Advantages and Limitations

Static arrays excel when the number of elements is known in advance and does not change during execution.

**Advantages:**

- **Constant-time access**: $O(1)$ read and write at any index, the fastest possible for random access.
- **Cache efficiency**: contiguous memory layout exploits spatial locality, so sequential scans run close to hardware speed.
- **Minimal overhead**: no pointers, no metadata beyond the base address and length. Memory usage is exactly $n \cdot w$ bytes plus a small constant.
- **Predictable performance**: no amortized costs or worst-case surprises from resizing.

**Limitations:**

- **Fixed capacity**: the size must be decided at allocation time. Overestimating wastes memory; underestimating requires reallocation.
- **Expensive insertion and deletion**: inserting or removing an element in the middle requires shifting $O(n)$ elements.
- **No built-in bounds checking** (in C/C++): accessing an out-of-bounds index causes undefined behavior rather than a clean error.

## Connection to Deep Learning

Static arrays are directly relevant to deep learning through tensors. A PyTorch or NumPy tensor is fundamentally a static array: a contiguous block of memory with fixed size and element type. The address computation formula above generalizes to multiple dimensions via strides, which Chapter 2 covers in detail. Understanding the flat, contiguous nature of static arrays is essential for reasoning about memory layout, cache performance, and GPU memory access patterns in neural network training.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
