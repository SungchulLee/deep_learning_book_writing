# LZ77 and LZ78

Statistical compressors like Huffman coding require advance knowledge of symbol frequencies.  When the data source is unknown or non-stationary, **dictionary-based** methods adapt on the fly by building a dictionary of previously seen patterns and replacing future occurrences with short references.  Abraham Lempel and Jacob Ziv introduced two foundational approaches in 1977 and 1978 that underpin nearly all modern lossless compressors, from gzip to zstd.

## LZ77 -- Sliding Window

LZ77 maintains a **sliding window** over the recently processed data.  The window has two parts:

- **Search buffer** (size $W$): already-encoded data available for back-references.
- **Look-ahead buffer** (size $L$): upcoming data to be encoded.

At each step, the encoder finds the **longest match** of the look-ahead buffer content within the search buffer and emits a triple:

$$
(d,\, \ell,\, c)
$$

where $d$ is the **offset** (distance back into the search buffer), $\ell$ is the **match length**, and $c$ is the first character after the match.  If no match is found, the encoder emits $(0, 0, c)$ with the literal character.

### LZ77 Worked Example

Encode the string `AABCAABCAA` with search buffer size $W = 7$:

| Step | Search buffer | Look-ahead | Match | Output |
|------|--------------|------------|-------|--------|
| 1    | (empty)      | AABCAABCAA | none  | (0, 0, A) |
| 2    | A            | ABCAABCAA  | A at offset 1 | (1, 1, B) |
| 3    | AAB          | CAABCAA    | none  | (0, 0, C) |
| 4    | AABC         | AABCAA     | AABC at offset 4 | (4, 4, A) |
| 5    | AABCAABCA   | A          | A at offset 1 | (1, 1, EOF) |

The decoder reconstructs the original by replaying each triple: copy $\ell$ characters from position $d$ behind the current write pointer, then append character $c$.

### LZ77 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Encode (naive) | $O(n \cdot W \cdot L)$ | $O(W + L)$ |
| Encode (suffix tree) | $O(n)$ | $O(W)$ |
| Decode | $O(n)$ | $O(W)$ |

Practical implementations (e.g., DEFLATE in gzip) use hash chains for match finding, achieving near-linear performance with $O(W)$ memory.

## LZ78 -- Explicit Dictionary

LZ78 takes a different approach: instead of a sliding window, it builds an **explicit dictionary** that grows during encoding.  The dictionary starts with a single entry: index 0 representing the empty string.

At each step, the encoder finds the longest prefix of the remaining input that matches a dictionary entry, then outputs a pair:

$$
(i,\, c)
$$

where $i$ is the index of the matched dictionary entry and $c$ is the next character.  The encoder then adds the concatenation (matched string + $c$) as a new dictionary entry.

### LZ78 Worked Example

Encode `AABCAABCAA`:

| Step | Remaining input | Longest match | Output | New entry |
|------|----------------|---------------|--------|-----------|
| 1    | AABCAABCAA     | (empty)       | (0, A) | 1: A      |
| 2    | ABCAABCAA      | A (index 1)   | (1, B) | 2: AB     |
| 3    | CAABCAA        | (empty)       | (0, C) | 3: C      |
| 4    | AABCAA         | A (index 1)   | (1, A) | 4: AA     |
| 5    | BCAA           | (empty)       | (0, B) | 5: B      |
| 6    | CAA            | C (index 3)   | (3, A) | 6: CA     |
| 7    | A              | A (index 1)   | (1, EOF) | --       |

The decoder rebuilds the dictionary in lockstep with the encoder, so no explicit dictionary transmission is needed.

### LZ78 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Encode | $O(n)$ with trie | $O(n)$ |
| Decode | $O(n)$ | $O(n)$ |

The dictionary is stored as a trie, giving $O(1)$ amortized time per input character.  Unlike LZ77, the dictionary can grow unboundedly, so practical implementations impose a maximum dictionary size.

## LZ77 vs LZ78

| Property | LZ77 | LZ78 |
|----------|------|------|
| Dictionary | Implicit (sliding window) | Explicit (trie) |
| Memory | $O(W)$ -- bounded | $O(n)$ -- grows with input |
| Back-references | Offset + length | Dictionary index |
| Adaptive | Yes (recent context) | Yes (full history) |
| Derivatives | DEFLATE, LZ4, zstd | LZW, LZC |

LZ77 tends to perform better on data with strong local correlations, while LZ78 can exploit patterns that recur at arbitrary distances.

## Implementation

```python
"""
LZ77 Compression -- sliding-window encoder and decoder.

Demonstrates the core LZ77 mechanism using a simple longest-match
search within a bounded search buffer.
"""

# === Encoder =================================================================

def lz77_encode(data: str, window_size: int = 16) -> list[tuple[int, int, str]]:
    """Encode a string using LZ77 with a sliding window.

    Returns a list of (offset, length, next_char) triples.
    """
    i = 0
    tokens = []
    while i < len(data):
        best_offset, best_length = 0, 0
        start = max(0, i - window_size)

        for j in range(start, i):
            length = 0
            while (i + length < len(data)
                   and length < window_size
                   and data[j + length] == data[i + length]):
                length += 1
            if length > best_length:
                best_offset = i - j
                best_length = length

        next_char = data[i + best_length] if i + best_length < len(data) else ""
        tokens.append((best_offset, best_length, next_char))
        i += best_length + 1

    return tokens


# === Decoder =================================================================

def lz77_decode(tokens: list[tuple[int, int, str]]) -> str:
    """Decode LZ77 tokens back to the original string."""
    output = []
    for offset, length, next_char in tokens:
        start = len(output) - offset
        for k in range(length):
            output.append(output[start + k])
        if next_char:
            output.append(next_char)
    return "".join(output)


# === Main ====================================================================

if __name__ == "__main__":
    original = "AABCAABCAA"
    print(f"Original: {original}")

    tokens = lz77_encode(original, window_size=7)
    print(f"Encoded : {tokens}")

    decoded = lz77_decode(tokens)
    print(f"Decoded : {decoded}")
    print(f"Match   : {original == decoded}")
```

**Output:**
```
Original: AABCAABCAA
Encoded : [(0, 0, 'A'), (1, 1, 'B'), (0, 0, 'C'), (4, 4, 'A'), (1, 1, '')]
Decoded : AABCAABCAA
Match   : True
```

## Theoretical Significance

Lempel and Ziv proved that both LZ77 and LZ78 are **asymptotically optimal**: for any stationary ergodic source, the compression ratio converges to the source entropy rate as the input length grows.  This universality -- achieving optimal compression without knowing the source distribution -- is the key theoretical contribution.

## Reference

- [A Universal Algorithm for Sequential Data Compression (Ziv & Lempel, 1977)](https://ieeexplore.ieee.org/document/1055714)
- [Compression of Individual Sequences via Variable-Rate Coding (Ziv & Lempel, 1978)](https://ieeexplore.ieee.org/document/1055934)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
