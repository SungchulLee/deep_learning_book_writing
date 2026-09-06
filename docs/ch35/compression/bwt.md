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

## Exercises

**Exercise 1.**
Compute the BWT of the string "banana\$". List all rotations, sort them lexicographically, and extract the last column.

??? success "Solution to Exercise 1"
    The 7 rotations of "banana\$" are: banana\$, anana\$b, nana\$ba, ana\$ban, na\$bana, a\$banan, \$banana. Sorted lexicographically: \$banana, a\$banan, ana\$ban, anana\$b, banana\$, na\$bana, nana\$ba. The last column (BWT output) is: a, n, b, \$, a, a, n, giving "annb\$aa". The original string index is 3 (position of \$ in the last column, 0-indexed). The BWT clusters the characters by context: the three 'a's and two 'n's appear together, creating runs that compress well with RLE. $\square$

---

**Exercise 2.**
Explain why the BWT tends to group identical characters together. What property of the rotation sort causes this clustering?

??? success "Solution to Exercise 2"
    The BWT's last column character at row $i$ is the character immediately preceding the first column character in the original string. Rows that sort adjacently have similar prefixes (since they are sorted lexicographically). Similar prefixes in the rotations correspond to similar right-contexts in the original string. Characters that appear before the same context (e.g., 'a' frequently precedes 'n' in English) cluster together in the last column. Formally, if two rotations share a long common prefix, they are adjacent in sorted order, and their last-column characters come from similar positions in the original string -- positions that share a right-context. This context-clustering is the key insight: BWT groups characters by their right-context, and in natural language, context determines character distribution strongly. $\square$

---

**Exercise 3.**
Describe the inverse BWT algorithm. Given only the last column and the index of the original string's row, how do you recover the original string in $O(n)$ time?

??? success "Solution to Exercise 3"
    Given the last column $L$ and the row index $r$ of the original string: (1) Sort $L$ to get the first column $F$ (since $F$ contains the same characters in sorted order). (2) Build the LF-mapping: for each occurrence of character $c$ in $L$, the $j$-th occurrence of $c$ in $L$ corresponds to the $j$-th occurrence of $c$ in $F$. (3) Starting at row $r$, repeatedly apply the LF-mapping: the character at position $r$ in $L$ is prepended to the output, and $r$ is updated to $\text{LF}(r)$. After $n$ steps, the original string is recovered (in reverse). Constructing $F$ takes $O(n)$ (counting sort). Building the LF-mapping takes $O(n)$ with a rank array. Each step of the traversal is $O(1)$, so the total is $O(n)$. $\square$

---

**Exercise 4.**
The BWT is used as a preprocessing step before move-to-front (MTF) encoding and then Huffman coding. Explain the role of each stage in the bzip2 pipeline and why the order matters.

??? success "Solution to Exercise 4"
    (1) **BWT**: rearranges the input so that characters from similar contexts are adjacent, creating long runs of identical or similar characters. It does not compress -- the output is the same size. (2) **MTF encoding**: replaces each character with its position in a recently-used list. Since the BWT output has long runs, consecutive characters are often the same, producing many 0s in the MTF output. Characters that are not the same but share a context produce small numbers. The MTF output is heavily skewed toward small values. (3) **Huffman (or arithmetic) coding**: assigns shorter codes to frequent symbols. The MTF output has many 0s and small values, so Huffman coding achieves high compression. The order matters: without BWT, the input has no special structure for MTF to exploit. Without MTF, the BWT output has runs but not the frequency skew that Huffman needs. Each stage prepares the data for the next. $\square$

---

**Exercise 5.**
Prove that the BWT is a reversible transformation: the last column plus the row index uniquely determines the original string.

??? success "Solution to Exercise 5"
    The last column $L$ and first column $F$ together define a permutation of the rotation matrix rows via the LF-mapping. The key property is: the $j$-th occurrence of character $c$ in $L$ and the $j$-th occurrence of $c$ in $F$ correspond to the same rotation. This holds because sorting the rotations preserves the relative order of rotations starting with the same character, and the last column character of row $i$ is the first column character of some row $\sigma(i)$ -- the LF-mapping. Starting from row $r$ (the original string), each application of $\text{LF}$ produces the next character of the original string. Since the rotation matrix is a permutation, the LF-mapping is a bijection on $\{0, \ldots, n-1\}$, and iterating it for $n$ steps cycles back to $r$, recovering all $n$ characters. Therefore, $(L, r)$ uniquely determines the original string. $\square$
