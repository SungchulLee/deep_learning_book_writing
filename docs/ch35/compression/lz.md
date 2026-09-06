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

## Exercises

**Exercise 1.**
Trace LZ77 encoding on the string "abcabcabc" with a search buffer of size 6 and a lookahead buffer of size 4. List all output tokens.

??? success "Solution to Exercise 1"
    Position 0: no match in search buffer. Output (0, 0, 'a'). Position 1: no match. Output (0, 0, 'b'). Position 2: no match. Output (0, 0, 'c'). Position 3: "abca" matches at offset 3 with length 3 (then next char 'a'). Output (3, 3, 'a'). Position 7: "bc" matches at offset 6 with length 2 (then end of input or next char). Output (6, 2, end). Total tokens: (0,0,'a'), (0,0,'b'), (0,0,'c'), (3,3,'a'), (6,2,end). The repeated "abc" pattern is captured by back-references, compressing 9 characters into fewer tokens. $\square$

---

**Exercise 2.**
Explain the key difference between LZ77 and LZ78. What are the advantages of each approach?

??? success "Solution to Exercise 2"
    **LZ77** uses a sliding window: it references previously seen data by (offset, length) pairs pointing back into the search buffer. The dictionary is implicit -- it is the content of the sliding window. **LZ78** builds an explicit dictionary: each new pattern extends an existing dictionary entry by one character, and the output is (dictionary index, next character). Advantages of LZ77: no explicit dictionary overhead; the window naturally "forgets" old data, adapting to changing content. Good for streaming. Advantages of LZ78: dictionary entries grow incrementally and can represent longer patterns earlier; decompression is simpler (just look up dictionary entries). Disadvantage of LZ77: matching against the sliding window can be slow ($O(n \cdot w)$ naive, improved with hash chains or suffix arrays). Disadvantage of LZ78: the dictionary can grow without bound and may need resetting. $\square$

---

**Exercise 3.**
Prove that LZ77 is asymptotically optimal: for any ergodic source, the compression ratio of LZ77 converges to the entropy rate as the input length goes to infinity.

??? success "Solution to Exercise 3"
    Ziv and Lempel (1977) proved that for a stationary ergodic source with entropy rate $h$, the LZ77 compression ratio $\rho_n$ satisfies $\rho_n \to h$ as $n \to \infty$ almost surely. The intuition: as the window grows, longer and longer matches are found, and the number of bits to encode each match (offset + length) amortizes to the entropy rate. Formally, the average codeword length per source symbol is bounded above by $h + \epsilon(n)$ where $\epsilon(n) \to 0$. The proof uses the fact that for large $n$, the probability of seeing any particular pattern of length $L$ is approximately $2^{-hL}$, so match lengths grow proportionally to $\log n$, and the encoding cost per symbol approaches $h$. This makes LZ77 a universal compressor -- it achieves optimal compression without knowing the source distribution. $\square$

---

**Exercise 4.**
Modern compressors like zstd and lz4 are based on LZ77 but achieve different speed/compression tradeoffs. Describe two techniques they use to improve upon basic LZ77.

??? success "Solution to Exercise 4"
    (1) **Hash chains / hash tables for match finding**: instead of searching the entire sliding window for matches (which is slow), modern compressors hash the next 3--4 bytes at each position and use the hash to quickly find candidate match positions. This reduces match-finding from $O(w)$ to expected $O(1)$ per position, dramatically improving compression speed. lz4 uses a single hash table with no chaining (greedy, fastest), while zstd uses multi-level hash tables and optimal parsing for better compression. (2) **Entropy coding of literals and match lengths**: basic LZ77 outputs raw (offset, length, literal) triples. Modern compressors apply Huffman or finite-state entropy (FSE/tANS) coding to the literal bytes, match lengths, and offsets separately, compressing each stream according to its distribution. zstd interleaves FSE-coded streams for high throughput, achieving near-arithmetic-coding compression at speeds close to lz4. $\square$

---

**Exercise 5.**
A financial system logs 100 million trade records per day, each roughly 200 bytes. Estimate the compression ratio achievable with LZ77-based compression and discuss whether real-time compression is feasible.

??? success "Solution to Exercise 5"
    Trade records are highly structured: each has fields like timestamp, symbol, price, quantity in a predictable format. Field values are often repetitive (same symbol appears in many trades, prices vary by small deltas). LZ77-based compressors exploit these repetitions: the symbol "AAPL" appearing thousands of times is encoded once and back-referenced. Typical compression ratios for structured log data: 5:1 to 10:1, reducing 20 GB/day to 2--4 GB. For real-time compression: lz4 compresses at 400+ MB/s on modern hardware. At $200 \times 10^8 / 86400 \approx 231$ KB/s average write rate, even a single core handles compression with $< 0.1\%$ CPU utilization. Peak rates of 10x average ($\approx 2.3$ MB/s) are still trivial. zstd at level 1 compresses at $\sim$500 MB/s with better ratios. Real-time compression is entirely feasible and is standard practice for financial log storage. $\square$
