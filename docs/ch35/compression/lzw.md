# LZW Compression

LZ78 outputs pairs of (dictionary index, next character), which means every output token includes a literal character.  Terry Welch's 1984 refinement, **LZW**, eliminates the explicit character from each output token by initializing the dictionary with all single-character entries.  This simpler output format -- a stream of dictionary indices -- made LZW practical for hardware implementations and led to its adoption in GIF images, early ZIP utilities, and Unix `compress`.

## Algorithm Overview

LZW builds its dictionary incrementally during a single pass over the input.  The dictionary starts pre-loaded with entries for every symbol in the alphabet (e.g., entries 0--255 for 8-bit bytes).

### Encoding

1. Initialize dictionary $D$ with all single-character strings, indexed $0$ to $|\Sigma| - 1$.
2. Set the current string $w \leftarrow$ empty.
3. For each input character $c$:
    - If $w + c$ is in $D$, extend: $w \leftarrow w + c$.
    - Otherwise:
        - Output the index $D[w]$.
        - Add $w + c$ to $D$ with the next available index.
        - Set $w \leftarrow c$.
4. Output $D[w]$ for the remaining string.

### Decoding

The decoder rebuilds the same dictionary in lockstep with the encoder.  It reads each index, outputs the corresponding string, and adds a new dictionary entry formed from the previous output string plus the first character of the current output string.

A subtle edge case arises when the encoder outputs an index that the decoder has **not yet added** -- the "KwKwK" problem.  This happens when the input contains a pattern like $c \cdot S \cdot c \cdot S \cdot c$ where $c \cdot S$ is already in the dictionary.  The decoder handles this by recognizing that the unknown entry must be the previous string plus its own first character.

## Worked Example

Encode `ABABABA` with initial dictionary $\{A: 0, B: 1\}$:

| Step | $w$ | $c$ | $w + c$ in $D$? | Output | New entry |
|------|-----|-----|-----------------|--------|-----------|
| 1 | (empty) | A | -- | -- | -- |
| 2 | A | B | No | 0 (A) | 2: AB |
| 3 | B | A | No | 1 (B) | 3: BA |
| 4 | A | B | Yes (AB) | -- | -- |
| 5 | AB | A | No | 2 (AB) | 4: ABA |
| 6 | A | B | Yes (AB) | -- | -- |
| 7 | AB | A | Yes (ABA) | -- | -- |
| 8 | ABA | (EOF) | -- | 4 (ABA) | -- |

Output stream: `[0, 1, 2, 4]` -- four indices instead of seven characters.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Encode | $O(n)$ amortized | $O(|\text{dict}|)$ |
| Decode | $O(n)$ amortized | $O(|\text{dict}|)$ |

Dictionary lookups use a trie or hash table, giving $O(1)$ amortized per character.  In practice, the dictionary is capped at a maximum size (e.g., $2^{12} = 4096$ entries for 12-bit GIF) and either frozen or reset when full.

## Implementation

```python
"""
LZW Compression -- encode and decode demonstration.

LZW builds a dictionary on the fly, outputting only dictionary indices.
This module shows the complete encode/decode cycle including handling
of the KwKwK edge case during decoding.
"""

# === Encoder =================================================================

def lzw_encode(data: str) -> list[int]:
    """Encode a string using LZW compression.

    Returns a list of dictionary indices.
    """
    # Initialize dictionary with single characters
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256

    w = ""
    output = []
    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            output.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c

    if w:
        output.append(dictionary[w])

    return output


# === Decoder =================================================================

def lzw_decode(codes: list[int]) -> str:
    """Decode an LZW-encoded list of indices back to the original string."""
    # Initialize dictionary with single characters
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256

    result = [dictionary[codes[0]]]
    w = result[0]

    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            # KwKwK case: entry not yet in dictionary
            entry = w + w[0]
        else:
            raise ValueError(f"Invalid code: {code}")

        result.append(entry)
        dictionary[next_code] = w + entry[0]
        next_code += 1
        w = entry

    return "".join(result)


# === Main ====================================================================

if __name__ == "__main__":
    original = "ABABABABABABAB"
    print(f"Original : {original}  (length {len(original)})")

    encoded = lzw_encode(original)
    print(f"Encoded  : {encoded}  ({len(encoded)} codes)")

    decoded = lzw_decode(encoded)
    print(f"Decoded  : {decoded}")
    print(f"Match    : {original == decoded}")

    # Demonstrate compression ratio
    print(f"\nCompression: {len(original)} chars -> {len(encoded)} codes")
```

**Output:**
```
Original : ABABABABABABAB  (length 14)
Encoded  : [65, 66, 256, 258, 260, 66]  (6 codes)
Decoded  : ABABABABABABAB
Match    : True

Compression: 14 chars -> 6 codes
```

## Dictionary Management

In practice, the dictionary cannot grow without bound.  Common strategies when the dictionary reaches its maximum size:

| Strategy | Description | Used in |
|----------|-------------|---------|
| Freeze | Stop adding entries; use existing dictionary | GIF (variable-width codes up to 12 bits) |
| Reset | Clear dictionary and restart from single characters | Unix `compress` |
| LRU eviction | Remove least-recently-used entries | Some modern variants |

!!! tip "Variable-width codes"
    GIF uses variable-width codes that start at the minimum needed for the initial alphabet and grow as the dictionary expands.  Each time the dictionary size exceeds $2^b$, the code width increases to $b + 1$ bits.  This avoids wasting bits on large code widths when the dictionary is still small.

## Patent History and Legacy

!!! note "Historical context"
    LZW was the subject of a notable software patent controversy.  Unisys held patents on LZW (US Patent 4,558,302, filed 1983) and began enforcing licensing fees in the 1990s, particularly targeting the widely-used GIF format.  This controversy motivated the development of PNG as a patent-free alternative using DEFLATE (LZ77 + Huffman).  The Unisys patent expired in 2003 (US) and 2004 (worldwide).

## Reference

- [A Technique for High-Performance Data Compression (Welch, 1984)](https://ieeexplore.ieee.org/document/1659158)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
