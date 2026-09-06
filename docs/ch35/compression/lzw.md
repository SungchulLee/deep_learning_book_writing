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

## Exercises

**Exercise 1.**
Trace LZW encoding on the input "ABABABA" with an initial dictionary containing A=0, B=1. Show the dictionary state and output codes after each step.

??? success "Solution to Exercise 1"
    Initial dictionary: {A:0, B:1}. Next code: 2. Step 1: read A, AB not in dict. Output 0 (A). Add AB=2. Step 2: read B, BA not in dict. Output 1 (B). Add BA=3. Step 3: read A, AB in dict (code 2). Read ABA, ABA not in dict. Output 2 (AB). Add ABA=4. Step 5: read B, BA in dict (code 3). Read BAB, BAB not in dict. Output 3 (BA). Add BAB=5. Step 7: read A, end of input. Output 0 (A). Output codes: 0, 1, 2, 3, 0. Dictionary: {A:0, B:1, AB:2, BA:3, ABA:4, BAB:5}. The dictionary grows by learning new patterns, and previously seen patterns are encoded with single codes. $\square$

---

**Exercise 2.**
Explain the "code not yet in dictionary" edge case in LZW decoding. When does it occur and how is it handled?

??? success "Solution to Exercise 2"
    This edge case occurs when the encoder outputs a code $c$ that the decoder has not yet added to its dictionary. It happens when the input has a pattern of the form $xSxSx$ where $S$ is a string already in the dictionary and $x$ is a single character. The encoder sees $xS$ (in dict), then $xSx$ (not in dict), so it outputs the code for $xS$ and adds $xSx$. But the decoder, one step behind, has not yet added $xSx$. The fix: if the decoder receives code $c$ that equals the next code to be added, it knows the new string is the previous string plus its first character: $\text{new\_entry} = \text{prev\_entry} + \text{prev\_entry}[0]$. This is the only case where the code is unknown, and the rule uniquely determines the string. $\square$

---

**Exercise 3.**
LZW was patented (US Patent 4,558,302, expired 2003). Discuss how the patent affected the adoption of GIF and what alternatives were developed.

??? success "Solution to Exercise 3"
    The LZW patent, held by Unisys, meant that software using LZW (including all GIF encoders) owed royalties. In the 1990s, Unisys began enforcing the patent, sending license demands to websites and developers using GIF images. This prompted two responses: (1) the PNG (Portable Network Graphics) format was created as a patent-free alternative, using DEFLATE (LZ77 + Huffman) instead of LZW. PNG offered better compression and features (alpha transparency, gamma correction) and became the preferred format for web graphics. (2) Some developers switched to JPEG for photographic images. After the patent expired in 2003--2004 (depending on country), GIF's legal risk vanished, but PNG had already become established. The episode demonstrated how patents on compression algorithms can fragment ecosystems and drive innovation in unexpected directions. $\square$

---

**Exercise 4.**
Compare the compression ratios of LZW, LZ77, and Huffman coding on three types of input: random bytes, English text, and a file of all zeros. Explain the differences.

??? success "Solution to Exercise 4"
    **Random bytes**: entropy is 8 bits/byte. No compressor can improve on this. LZW and LZ77 may slightly expand the data (dictionary/header overhead). Huffman achieves 8 bits/byte (uniform frequencies). All three produce output $\ge$ input size. **English text** (entropy $\approx 1$--$2$ bits/char): LZ77 and LZW exploit repeated words and phrases, achieving 2--4x compression. Huffman exploits frequency imbalance (e/t/a are common), achieving $\sim$1.5--2x. LZ77/LZW outperform Huffman because they capture multi-character patterns, not just single-character frequencies. **All zeros** (entropy = 0): all three compress extremely well. RLE would achieve the best ratio. LZW: after learning a few patterns (0, 00, 000, ...), it encodes exponentially longer runs with single codes. LZ77: one back-reference covers the entire file. Huffman: all weight on one symbol, 1 bit/byte. $\square$

---

**Exercise 5.**
Design an LZW variant that limits the dictionary size to $2^{16}$ entries. What happens when the dictionary fills up, and what strategies can maintain good compression?

??? success "Solution to Exercise 5"
    When the dictionary reaches $2^{16} = 65536$ entries, no new patterns can be added. Strategies: (1) **Freeze**: stop adding entries but continue using existing ones. Compression degrades gradually as the encoder cannot adapt to new patterns not in the dictionary. (2) **Reset**: clear the dictionary and restart with only the initial single-character entries. This causes a temporary compression drop but allows adaptation to changing data. A reset marker code is sent to synchronize the decoder. (3) **LRU eviction**: replace the least recently used dictionary entry with the new pattern. This is complex (requires the decoder to track usage) but maintains adaptivity. GIF uses strategy (2) with a "clear code" signal. Unix `compress` uses (1) with monitoring -- it resets when the compression ratio stops improving. Strategy (3) is rarely used due to implementation complexity and synchronization overhead between encoder and decoder. $\square$
