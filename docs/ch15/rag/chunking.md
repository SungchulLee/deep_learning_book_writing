# Document Chunking

## Learning Objectives

- Understand chunking strategies and their trade-offs
- Implement different chunking approaches
- Apply chunking to financial documents

## Why Chunking Matters

LLMs have limited context windows, and retrieval works best with focused, coherent text segments. **Chunking** is the process of splitting documents into retrieval units.

The chunk size creates a fundamental trade-off:

- **Too small**: Loses context, fragments meaning
- **Too large**: Dilutes relevance, wastes context window

## Chunking Strategies

### Fixed-Size Chunking

Split by character/token count with overlap:

```python
def fixed_size_chunk(text, chunk_size=512, overlap=64):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

### Sentence-Based Chunking

Split on sentence boundaries, group into chunks:

```python
import re

def sentence_chunk(text, max_sentences=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks
```

### Semantic Chunking

Split where embedding similarity between adjacent sentences drops:

```python
def semantic_chunk(sentences, embeddings, threshold=0.5):
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))
    return chunks
```

### Hierarchical Chunking

Create chunks at multiple granularities (paragraph, section, document) with parent-child relationships.

## Financial Document Chunking

| Document Type | Recommended Strategy | Chunk Size | Notes |
|--------------|---------------------|-----------|-------|
| SEC 10-K | Section-based | 500-1000 tokens | Split by Item number |
| Earnings transcript | Speaker-turn based | 200-500 tokens | Preserve Q&A pairs |
| Research report | Section + semantic | 300-600 tokens | Keep tables intact |
| News articles | Paragraph-based | 200-400 tokens | Include headline |

```python
def chunk_sec_filing(text):
    """Chunk SEC filing by Item sections."""
    # Split on common 10-K section headers
    items = re.split(
        r'(Item\s+\d+[A-Z]?\.\s+[^\n]+)',
        text, flags=re.IGNORECASE
    )
    chunks = []
    for i in range(1, len(items), 2):
        header = items[i].strip()
        content = items[i + 1].strip() if i + 1 < len(items) else ""
        # Further split large sections
        if len(content.split()) > 1000:
            sub_chunks = fixed_size_chunk(content, chunk_size=500, overlap=50)
            for j, sc in enumerate(sub_chunks):
                chunks.append({"header": header, "part": j + 1, "text": sc})
        else:
            chunks.append({"header": header, "part": 1, "text": content})
    return chunks
```

## References

1. Gao, Y., et al. (2024). "Retrieval-Augmented Generation for LLMs: A Survey."
