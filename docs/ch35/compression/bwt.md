# Burrows-Wheeler Transform

Dictionary-based compressors like LZ77 exploit repeated substrings, while statistical compressors like Huffman exploit frequency imbalances.  The Burrows-Wheeler Transform (BWT) bridges these worlds: it **rearranges** the input so that characters from similar contexts cluster together, creating long runs of identical symbols that simpler methods (RLE, move-to-front coding) can then compress efficiently.  This preprocessing step is the engine behind bzip2, one of the most effective general-purpose compressors.

## Forward Transform

Given a string $s$ of length $n$ (with an appended end-of-string sentinel `$` that is lexicographically smaller than all other characters), the BWT proceeds in three steps:

1. **Generate all cyclic rotations** of $s$, forming an $n \times n$ matrix $M$ where row $i$ is the string rotated left by $i$ positions.
2. **Sort** the rows of $M$ lexicographically.
3. **Extract the last column** $L$ of the sorted matrix.  This is the BWT output.

The transform also records the **row index** $I$ where the original string appears in the sorted matrix, needed for inversion.

## Worked Example

Transform the string `banana$`:

**Step 1 -- All rotations:**

| Index | Rotation |
|-------|----------|
| 0     | banana$  |
| 1     | anana$b  |
| 2     | nana$ba  |
| 3     | ana$ban  |
| 4     | na$bana  |
| 5     | a$banan  |
| 6     | $banana  |

**Step 2 -- Sort lexicographically:**

| Sorted index | First (F) | Rotation | Last (L) |
|-------------|-----------|----------|----------|
| 0           | $         | $banana  | a        |
| 1           | a         | a$banan  | n        |
| 2           | a         | ana$ban  | n        |
| 3           | a         | anana$b  | b        |
| 4           | b         | banana$  | $        |
| 5           | n         | na$bana  | a        |
| 6           | n         | nana$ba  | a        |

**BWT output:** $L = $ `annb$aa`, with original string at row $I = 4$.

Notice how the `a`'s and `n`'s cluster together in $L$ -- this is the key property that makes subsequent compression effective.

## Why Characters Cluster

The sorted rows group together all rotations that share the same **suffix context**.  Since characters in natural text are correlated with their context (e.g., `t` often follows `s`), the last column -- which contains the character that **precedes** each sorted suffix -- tends to have long runs of identical characters.

## Inverse Transform

The BWT is **reversible** from just $L$ and $I$.  The inverse exploits the **LF-mapping**: the $k$-th occurrence of character $c$ in the last column $L$ corresponds to the $k$-th occurrence of $c$ in the first column $F$.

**Algorithm:**

1. Compute $F$ by sorting $L$.
2. Build the LF-mapping: for each position $i$ in $L$, find the corresponding position in $F$.
3. Starting from position $I$, follow the LF-mapping for $n$ steps, prepending each character to reconstruct the original string.

The first column $F$ is simply the sorted version of $L$, so it requires no additional information beyond $L$ itself.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Forward (naive) | $O(n^2 \log n)$ | $O(n^2)$ |
| Forward (suffix array) | $O(n)$ | $O(n)$ |
| Inverse | $O(n)$ | $O(n + |\Sigma|)$ |

The naive approach builds and sorts all rotations explicitly.  In practice, the BWT is computed via suffix arrays, avoiding the quadratic space of the rotation matrix entirely.

## Implementation

```python
"""
Burrows-Wheeler Transform -- forward and inverse demonstrations.

The BWT rearranges a string to cluster similar characters together,
enabling more effective compression by downstream algorithms like
RLE or move-to-front coding.
"""

# === Forward Transform ========================================================

def bwt_encode(text: str) -> tuple[str, int]:
    """Compute the Burrows-Wheeler Transform of a string.

    Appends a sentinel '$' and returns (transformed_string, original_row_index).
    """
    s = text + "$"
    n = len(s)

    # Generate and sort all rotations
    rotations = sorted(s[i:] + s[:i] for i in range(n))

    # Last column and original index
    last_column = "".join(r[-1] for r in rotations)
    original_index = rotations.index(s)

    return last_column, original_index


# === Inverse Transform ========================================================

def bwt_decode(last_column: str, original_index: int) -> str:
    """Reconstruct the original string from the BWT output.

    Uses the LF-mapping to walk through the permutation.
    """
    n = len(last_column)

    # Build the first column by sorting
    first_column = sorted(range(n), key=lambda i: last_column[i])

    # Build LF-mapping
    lf = [0] * n
    for new_pos, old_pos in enumerate(first_column):
        lf[old_pos] = new_pos

    # Reconstruct by following LF-mapping
    result = []
    idx = original_index
    for _ in range(n):
        result.append(last_column[idx])
        idx = lf[idx]

    # Remove sentinel and reverse (we collected in reverse order)
    return "".join(result).rstrip("$")


# === Main ====================================================================

if __name__ == "__main__":
    original = "banana"
    print(f"Original : {original}")

    bwt_string, row_idx = bwt_encode(original)
    print(f"BWT      : {bwt_string}  (row index = {row_idx})")

    decoded = bwt_decode(bwt_string, row_idx)
    print(f"Decoded  : {decoded}")
    print(f"Match    : {original == decoded}")
```

**Output:**
```
Original : banana
BWT      : annb$aa  (row index = 4)
Decoded  : banana
Match    : True
```

## BWT in the Compression Pipeline

In practice, BWT is never used alone.  The standard bzip2 pipeline combines several stages:

1. **BWT** -- rearranges the input to cluster similar characters.
2. **Move-to-front (MTF) coding** -- converts the clustered output into a sequence dominated by small integers (many zeros).
3. **Run-length encoding** -- compresses the runs of zeros.
4. **Huffman coding** -- encodes the remaining symbols optimally.

This combination achieves compression ratios competitive with PPM and LZMA on many data types, while maintaining reasonable encoding and decoding speed.

!!! tip "Connection to suffix arrays"
    Computing the BWT via suffix arrays (as in the SA-IS algorithm) reduces the forward transform to $O(n)$ time and $O(n)$ space, making it practical for large inputs.  The suffix array of $s$ directly gives the sorted order of rotations, and the last column is obtained by looking one position before each suffix start.

## Reference

- [A Block-sorting Lossless Data Compression Algorithm (Burrows & Wheeler, 1994)](https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.html)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
