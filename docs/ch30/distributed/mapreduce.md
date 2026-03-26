# MapReduce

Processing terabytes of data on a single machine is infeasible.  MapReduce
is a programming model that distributes computation across thousands of
machines by expressing the task as two simple functions: **Map** and
**Reduce**.  The framework handles parallelization, fault tolerance, and
data distribution automatically.

## Programming Model

A MapReduce computation transforms a set of input key-value pairs into
a set of output key-value pairs through two user-defined functions.

### Map

$$
\text{Map}: (k_1, v_1) \to \text{list}(k_2, v_2)
$$

The Map function processes each input pair independently and emits zero
or more intermediate key-value pairs.

### Reduce

$$
\text{Reduce}: (k_2, \text{list}(v_2)) \to \text{list}(v_3)
$$

The Reduce function receives all intermediate values associated with a
given key and combines them into a (typically smaller) set of output values.

Between Map and Reduce, a **shuffle** phase groups intermediate pairs by
key and distributes them to the appropriate reducers.

## Execution Flow

1. **Split.**  The input is divided into $M$ splits, each assigned to a
   mapper.
2. **Map.**  Each mapper applies the Map function to its split, producing
   intermediate pairs written to local disk.
3. **Shuffle/Sort.**  The framework partitions intermediate pairs by key
   (typically via $\text{hash}(k_2) \bmod R$, where $R$ is the number of
   reducers) and sorts them.
4. **Reduce.**  Each reducer applies the Reduce function to all values
   for each key in its partition.
5. **Output.**  Reducer results are written to the distributed file system.

!!! note "Fault Tolerance"
    If a mapper or reducer fails, the framework re-executes its task on
    another machine.  Map outputs are stored on local disk, so only the
    failed task needs re-execution.  This is possible because both Map and
    Reduce are **deterministic** and **side-effect-free**.

## Classic Example: Word Count

```python
"""
MapReduce word count (simulated single-machine version).

Demonstrates the Map and Reduce functions for counting word frequencies.
"""

from collections import defaultdict


# === Map Function ===
def map_fn(doc_id: str, text: str) -> list[tuple[str, int]]:
    """Emit (word, 1) for each word in the document."""
    pairs = []
    for word in text.lower().split():
        pairs.append((word, 1))
    return pairs


# === Reduce Function ===
def reduce_fn(word: str, counts: list[int]) -> int:
    """Sum all counts for a given word."""
    return sum(counts)


# === MapReduce Simulation ===
def mapreduce(documents: dict[str, str]) -> dict[str, int]:
    """Simulate MapReduce word count."""
    # Map phase
    intermediate: defaultdict[str, list[int]] = defaultdict(list)
    for doc_id, text in documents.items():
        for word, count in map_fn(doc_id, text):
            intermediate[word].append(count)

    # Reduce phase
    result = {}
    for word, counts in sorted(intermediate.items()):
        result[word] = reduce_fn(word, counts)
    return result


# === Example ===
if __name__ == "__main__":
    docs = {
        "doc1": "the cat sat on the mat",
        "doc2": "the dog sat on the log",
        "doc3": "the cat and the dog",
    }
    counts = mapreduce(docs)
    for word, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {word}: {count}")
```

## Complexity Model

In the MapReduce complexity model, the key metrics are:

| Metric | Description |
|---|---|
| **Rounds** | Number of MapReduce rounds (sequential steps) |
| **Communication** | Total data sent during shuffle |
| **Map work** | Total computation across all mappers |
| **Reduce work** | Total computation across all reducers |

Many problems can be solved in $O(1)$ MapReduce rounds (e.g., word count,
inverted index), while graph problems often require $O(\log n)$ or more
rounds.

## MapReduce Algorithms

### Sorting

- Map: emit $(k, v)$ where $k$ is the sort key.
- The shuffle phase sorts by key automatically.
- Reduce: identity function.

Total communication: $O(N)$ where $N$ is the input size.

### Matrix Multiplication

To compute $C = AB$ where $A$ is $m \times p$ and $B$ is $p \times n$:

- Map: for each $A_{ij}$, emit $((i, k), A_{ij})$ for all $k$; for each
  $B_{jk}$, emit $((i, k), B_{jk})$ for all $i$.
- Reduce: for key $(i, k)$, compute $C_{ik} = \sum_j A_{ij} B_{jk}$.

One round, $O(mpn)$ communication.

## Limitations and Extensions

- **Iterative algorithms.**  Graph algorithms (PageRank, connected
  components) require multiple MapReduce rounds, each incurring shuffle
  overhead.
- **Stragglers.**  A slow mapper delays the entire computation.  Speculative
  execution (running duplicate tasks) mitigates this.
- **Beyond MapReduce.**  Systems like Spark (resilient distributed datasets),
  Dryad (DAG execution), and Pregel (vertex-centric graph processing)
  address these limitations.

!!! tip "Combiners"
    A **combiner** is a local reduce applied on each mapper before the
    shuffle.  For associative-commutative operations like sum, this
    dramatically reduces network traffic.

## Reference

- Dean, J. & Ghemawat, S. "MapReduce: Simplified Data Processing on Large
  Clusters." OSDI 2004.
- Leskovec, J., Rajaraman, A., & Ullman, J. D. *Mining of Massive Datasets*.
  Cambridge University Press.
