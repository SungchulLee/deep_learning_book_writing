# Run-Length Encoding

Many real-world data sources contain long runs of repeated values -- pixel rows in monochrome images, silence in audio streams, or repeated characters in log files.  Run-length encoding (RLE) exploits this structure by replacing each maximal run of identical symbols with a single (count, value) pair, often achieving dramatic compression when repetitions dominate.  This page formalizes the encoding, analyzes its complexity, and examines when RLE helps versus when it hurts.

## Encoding Definition

Given an input sequence of $n$ symbols from an alphabet $\Sigma$, RLE partitions the sequence into **maximal runs** -- contiguous subsequences of identical symbols.

For input $s = s_1 s_2 \dots s_n$, define the $k$-th run as $(c_k, v_k)$ where $v_k \in \Sigma$ is the repeated symbol and $c_k \geq 1$ is the run length, such that

$$
s = \underbrace{v_1 v_1 \cdots v_1}_{c_1} \underbrace{v_2 v_2 \cdots v_2}_{c_2} \cdots \underbrace{v_r v_r \cdots v_r}_{c_r}
$$

where $r$ is the total number of runs and $\sum_{k=1}^{r} c_k = n$.  Adjacent runs must differ: $v_k \neq v_{k+1}$ for all $k$.

The encoded output is the sequence of pairs $(c_1, v_1), (c_2, v_2), \dots, (c_r, v_r)$.

## Worked Example

Consider the string `AAABBBCCDDDDDDAA`:

| Run | Symbol ($v_k$) | Count ($c_k$) |
|-----|---------------|---------------|
| 1   | A             | 3             |
| 2   | B             | 3             |
| 3   | C             | 2             |
| 4   | D             | 6             |
| 5   | A             | 2             |

Encoded: `3A3B2C6D2A` -- 10 characters instead of 16, a compression ratio of $10/16 = 0.625$.

## Compression Ratio Analysis

If the input has $n$ symbols and $r$ maximal runs, the encoded representation stores $r$ pairs.  The compression ratio depends on the encoding format for counts:

- **Fixed-width counts**: each pair uses $\lceil \log_2 n \rceil + \lceil \log_2 |\Sigma| \rceil$ bits, giving total size $r \cdot (\lceil \log_2 n \rceil + \lceil \log_2 |\Sigma| \rceil)$.
- **Variable-length counts**: using a prefix-free encoding for counts (such as Elias gamma coding) reduces overhead when most runs are short.

RLE achieves compression when $r \ll n$.  In the worst case, every symbol differs from its neighbors, so $r = n$ and the encoded output is larger than the input.  This worst case makes RLE a poor choice for data without long runs (e.g., natural language text).

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Encode    | $O(n)$ | $O(r)$ |
| Decode    | $O(n)$ | $O(n)$ |

Both encoding and decoding perform a single linear scan.  No auxiliary data structures are needed beyond the output buffer, making RLE one of the simplest compression algorithms.

## Implementation

```python
"""
Run-Length Encoding (RLE) -- encode and decode demonstration.

RLE replaces consecutive runs of identical symbols with (count, symbol)
pairs.  This module shows both encoding and decoding in a single pass.
"""

# === Encoder =================================================================

def rle_encode(data: str) -> list[tuple[int, str]]:
    """Encode a string using run-length encoding.

    Returns a list of (count, character) pairs representing maximal runs.
    """
    if not data:
        return []

    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append((count, data[i - 1]))
            count = 1
    encoded.append((count, data[-1]))
    return encoded


# === Decoder =================================================================

def rle_decode(encoded: list[tuple[int, str]]) -> str:
    """Decode an RLE-encoded list back to the original string."""
    return "".join(char * count for count, char in encoded)


# === Main ====================================================================

if __name__ == "__main__":
    original = "AAABBBCCDDDDDDAA"
    print(f"Original : {original}  (length {len(original)})")

    encoded = rle_encode(original)
    print(f"Encoded  : {encoded}")

    compact = "".join(f"{c}{v}" for c, v in encoded)
    print(f"Compact  : {compact}  (length {len(compact)})")

    decoded = rle_decode(encoded)
    print(f"Decoded  : {decoded}")
    print(f"Match    : {original == decoded}")
```

**Output:**
```
Original : AAABBBCCDDDDDDAA  (length 16)
Encoded  : [(3, 'A'), (3, 'B'), (2, 'C'), (6, 'D'), (2, 'A')]
Compact  : 3A3B2C6D2A  (length 10)
Decoded  : AAABBBCCDDDDDDAA
Match    : True
```

## When RLE Helps and When It Hurts

!!! tip "Best-case scenarios for RLE"
    - **Binary images**: large regions of black or white pixels produce long runs.
    - **Sparse data**: matrices with many repeated zeros compress well.
    - **Preprocessing step**: RLE after the Burrows-Wheeler Transform (BWT) exploits the clustering BWT creates.

!!! warning "Worst case"
    When every adjacent pair differs ($r = n$), RLE *expands* the data because each symbol now requires an additional count field.  For general-purpose compression, RLE alone is insufficient.

## Applications

- **BMP and TIFF image formats**: use RLE for simple lossless compression of pixel rows.
- **PackBits**: a byte-oriented RLE variant used in early Macintosh graphics.
- **Fax machines (Group 3/4 encoding)**: compress scanned pages that are mostly white with sparse black text, achieving ratios of 10:1 or better.
- **Preprocessing for BWT-based compressors**: tools like bzip2 apply RLE before and after the Burrows-Wheeler Transform.

## Reference

- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
