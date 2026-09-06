# Inverted Index

Given a collection of documents, a search engine must quickly find all documents containing a query term. A naive approach scans every document for every query -- $O(N \cdot D)$ where $N$ is the number of documents and $D$ is the average document length. An **inverted index** preprocesses the collection so that queries run in time proportional to the number of matching documents, not the total collection size.

## Structure

An inverted index maps each term to a **postings list** -- a sorted list of document IDs containing that term:

- **Dictionary**: A hash table or sorted array of all distinct terms.
- **Postings list**: For each term $t$, a list $[d_1, d_2, \ldots, d_k]$ of document IDs where $t$ appears.

Optionally, each posting stores the **term frequency** $\text{tf}(t, d)$ and the positions where $t$ appears in $d$.

## Construction

Given $N$ documents with a total of $T$ tokens:

1. **Tokenize** each document into terms.
2. For each (term, doc\_id) pair, add doc\_id to the term's postings list.
3. Sort each postings list by doc\_id.

$$
T_{\text{build}} = O(T \log T), \quad S = O(T)
$$

## Query Processing

### Single-Term Query

Look up the term in the dictionary and return its postings list. With a hash table:

$$
T_{\text{single query}} = O(1 + |\text{postings}|)
$$

### Boolean AND Query

For terms $t_1, t_2$, intersect their postings lists. Since lists are sorted, use a merge-based intersection:

$$
T_{\text{AND}} = O(|P_1| + |P_2|)
$$

where $|P_i|$ is the length of term $t_i$'s postings list.

### Boolean OR Query

Merge the postings lists (union):

$$
T_{\text{OR}} = O(|P_1| + |P_2|)
$$

## TF-IDF Scoring

To rank results by relevance, assign each (term, document) pair a **TF-IDF** score:

$$
\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \log \frac{N}{\text{df}(t)}
$$

where $\text{tf}(t, d)$ is the number of times term $t$ appears in document $d$, $\text{df}(t)$ is the number of documents containing $t$, and $N$ is the total number of documents.

The IDF factor $\log(N / \text{df}(t))$ downweights common terms (like "the") and upweights rare, informative terms.

!!! tip "Cosine similarity"
    To compute similarity between a query $q$ and document $d$, represent both as TF-IDF vectors and compute their cosine similarity: $\cos(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|}$.

## Implementation

```python
"""
Inverted Index -- construction, boolean queries, and TF-IDF scoring.

Builds an inverted index from a document collection, supports
boolean AND/OR queries, and ranks results by TF-IDF score.
"""

from __future__ import annotations
import math
from collections import defaultdict


# === Inverted Index ===========================================================

class InvertedIndex:
    """Inverted index with TF-IDF scoring."""

    def __init__(self):
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_count = 0
        self.doc_lengths: dict[int, int] = {}

    def add_document(self, doc_id: int, text: str) -> None:
        """Index a document by tokenizing and building postings."""
        tokens = text.lower().split()
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

        # Count term frequencies
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        for term, freq in tf.items():
            self.index[term].append((doc_id, freq))

    def search_and(self, terms: list[str]) -> list[int]:
        """Boolean AND: return doc_ids containing ALL terms."""
        if not terms:
            return []
        sets = []
        for term in terms:
            term = term.lower()
            doc_ids = {doc_id for doc_id, _ in self.index.get(term, [])}
            sets.append(doc_ids)
        result = sets[0]
        for s in sets[1:]:
            result &= s
        return sorted(result)

    def search_or(self, terms: list[str]) -> list[int]:
        """Boolean OR: return doc_ids containing ANY term."""
        result: set[int] = set()
        for term in terms:
            term = term.lower()
            for doc_id, _ in self.index.get(term, []):
                result.add(doc_id)
        return sorted(result)

    def tfidf_rank(self, query: list[str]) -> list[tuple[int, float]]:
        """Rank documents by TF-IDF score for the query terms."""
        scores: dict[int, float] = defaultdict(float)
        for term in query:
            term = term.lower()
            postings = self.index.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(self.doc_count / df)
            for doc_id, tf in postings:
                scores[doc_id] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked


# === Main =====================================================================

if __name__ == "__main__":
    idx = InvertedIndex()

    documents = {
        0: "the quick brown fox jumps over the lazy dog",
        1: "the fox hunts the rabbit in the forest",
        2: "a lazy dog sleeps in the sun",
        3: "the quick rabbit runs from the fox",
    }

    for doc_id, text in documents.items():
        idx.add_document(doc_id, text)

    print(f"Indexed {idx.doc_count} documents\n")

    # Boolean queries
    and_result = idx.search_and(["fox", "the"])
    print(f"AND('fox', 'the'): docs {and_result}")

    or_result = idx.search_or(["lazy", "rabbit"])
    print(f"OR('lazy', 'rabbit'): docs {or_result}")

    # TF-IDF ranking
    print("\nTF-IDF ranking for 'fox rabbit':")
    for doc_id, score in idx.tfidf_rank(["fox", "rabbit"]):
        print(f"  Doc {doc_id}: score={score:.3f}  \"{documents[doc_id][:40]}...\"")
```

**Output:**

```
Indexed 4 documents

AND('fox', 'the'): docs [0, 1, 3]
OR('lazy', 'rabbit'): docs [0, 1, 2, 3]

TF-IDF ranking for 'fox rabbit':
  Doc 1: score=0.981  "the fox hunts the rabbit in the forest..."
  Doc 3: score=0.981  "the quick rabbit runs from the fox..."
  Doc 0: score=0.288  "the quick brown fox jumps over the lazy ..."
```

Documents 1 and 3 rank highest because they contain both "fox" and "rabbit." Document 0 scores lower because it contains only "fox." The TF-IDF scores correctly reflect term rarity: "rabbit" (appearing in only 2 of 4 documents) contributes more to the score than a common term.

## Reference

- Manning, C.D., Raghavan, P., and Schutze, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008
- Zobel, J. and Moffat, A. "Inverted Files for Text Search Engines." *ACM Computing Surveys*, 2006

## Exercises

**Exercise 1.**
Build an inverted index for the following three documents: D1="the cat sat", D2="the dog sat", D3="cat and dog". Show the posting lists for each term.

??? success "Solution to Exercise 1"
    Terms and posting lists (document IDs): "the" -> [D1, D2], "cat" -> [D1, D3], "sat" -> [D1, D2], "dog" -> [D2, D3], "and" -> [D3]. To answer query "cat AND dog": intersect posting lists [D1, D3] and [D2, D3] = [D3]. Document D3 contains both terms. With TF (term frequency) augmentation: "the" -> [(D1,1), (D2,1)], "cat" -> [(D1,1), (D3,1)], etc. This supports TF-IDF ranking in addition to boolean queries. $\square$

---

**Exercise 2.**
Describe an efficient algorithm for intersecting two sorted posting lists of lengths $m$ and $n$ ($m \le n$). What is the time complexity?

??? success "Solution to Exercise 2"
    Use a merge-based intersection: maintain two pointers, one for each list. Compare the current elements. If equal, add to the result and advance both pointers. If the left element is smaller, advance the left pointer. Otherwise, advance the right pointer. Time: $O(m + n)$. When $m \ll n$, a more efficient approach is binary search: for each element in the shorter list, binary search for it in the longer list. Time: $O(m \log n)$. This is better when $m \ll n / \log n$. An adaptive approach: use galloping search (exponential search + binary search): for each element in the short list, exponentially probe the long list to find the neighborhood, then binary search. Average case: $O(m \log(n/m))$, which smoothly interpolates between the two extremes. $\square$

---

**Exercise 3.**
Explain how posting list compression reduces the storage size of an inverted index. Describe delta encoding and variable-byte encoding.

??? success "Solution to Exercise 3"
    Posting lists store sorted document IDs, which increase monotonically. **Delta encoding**: store the differences between consecutive IDs rather than absolute IDs. For list [3, 5, 20, 21, 23]: deltas are [3, 2, 15, 1, 2]. Deltas are smaller numbers, requiring fewer bits. **Variable-byte encoding**: encode each delta using a variable number of bytes. Use the high bit of each byte as a continuation flag (1 = more bytes, 0 = last byte). Small deltas (< 128) use 1 byte; larger deltas use 2--4 bytes. For the deltas [3, 2, 15, 1, 2]: each fits in 1 byte (7 data bits), total 5 bytes vs. 20 bytes for 32-bit integers. Compression ratio: typically 4--8x for web-scale indices. More aggressive schemes (PForDelta, Simple-9, SIMD-based) achieve better ratios with even faster decompression. $\square$

---

**Exercise 4.**
A search engine indexes 10 billion documents. The term "the" appears in 8 billion documents. Discuss why this term should be treated specially and how stop-word handling affects index size and query performance.

??? success "Solution to Exercise 4"
    The posting list for "the" contains 8 billion entries, consuming $\sim$8 GB even with compression. This single term accounts for a disproportionate fraction of the index. Handling: (1) **Stop-word removal**: exclude "the" from the index entirely. Reduces index size significantly. Queries containing "the" ignore the term (e.g., "the matrix" becomes just "matrix"). This can cause incorrect results for queries where the stop word is meaningful (e.g., "The Who," "to be or not to be"). (2) **Tiered indexing**: index "the" but store it in a lower tier. For conjunctive queries (AND), skip the posting list for "the" because every document matches. Only access it for phrase queries or proximity queries where position matters. (3) **Frequency-based pruning**: keep only the top-ranked documents in "the"'s posting list (e.g., top 1 million by PageRank), since lower-ranked documents add no value to ranking. $\square$

---

**Exercise 5.**
Compare inverted indexes with forward indexes. When is each appropriate, and how does a search engine use both?

??? success "Solution to Exercise 5"
    **Inverted index**: maps terms to documents. Efficient for finding which documents contain a query term ($O(1)$ lookup per term + posting list traversal). Essential for query processing. **Forward index**: maps documents to their terms (with positions, frequencies). Efficient for computing document-level features (e.g., document length, term frequency for a specific document). Essential for scoring and snippet generation. A search engine uses both: (1) the inverted index identifies candidate documents matching the query (retrieval phase). (2) The forward index computes detailed relevance scores for each candidate (ranking phase) and generates the snippet shown in search results. The inverted index is the larger structure ($\sim$100 TB for a web-scale engine) stored on disk/SSD. The forward index is smaller (only the subset of documents returned needs scoring) and is often memory-mapped. $\square$
