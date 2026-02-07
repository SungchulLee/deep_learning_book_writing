# Training Data at Scale

## Learning Objectives

- Understand the scale and composition of modern LLM training corpora
- Describe the data curation pipeline: collection, filtering, deduplication
- Analyze the relationship between data quality and model performance
- Identify contamination risks and mitigation strategies

## Scale of Modern Training Data

| Model | Year | Training Tokens | Data Sources |
|-------|------|----------------|-------------|
| GPT-2 | 2019 | ~10B | WebText (Reddit-filtered) |
| GPT-3 | 2020 | 300B | CommonCrawl, WebText2, Books, Wikipedia |
| Chinchilla | 2022 | 1.4T | MassiveText |
| LLaMA | 2023 | 1.0-1.4T | CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange |
| LLaMA 2 | 2023 | 2.0T | Undisclosed web mix |
| LLaMA 3 | 2024 | 15T+ | Expanded web + code + multilingual |

## Common Data Sources

### Web Crawls

- **CommonCrawl**: Petabytes of raw HTML from web crawls; requires heavy filtering
- **C4 (Colossal Clean Crawled Corpus)**: Cleaned version of CommonCrawl with heuristic filters
- **RefinedWeb**: High-quality web data with aggressive deduplication (used by Falcon)

### Curated Sources

- **Wikipedia**: High-quality encyclopedic text (~4B tokens English)
- **Books**: BookCorpus, Books3, Gutenberg (~30B tokens)
- **ArXiv**: Scientific papers (~30B tokens)
- **GitHub**: Open-source code (~100B+ tokens)
- **StackExchange**: Q&A format technical discussions

## Data Curation Pipeline

Raw web data is extremely noisy. Modern LLMs use multi-stage curation:

```
Raw Web Crawl -> Language ID -> URL Filtering -> Content Extraction
     -> Quality Filtering -> Deduplication -> Toxicity Filtering
     -> PII Removal -> Domain Mixing -> Final Corpus
```

### Quality Filtering

```python
from collections import Counter


def quality_filter(document: str) -> bool:
    # Length filter
    words = document.split()
    if len(words) < 50 or len(words) > 100000:
        return False

    # Repetition filter
    lines = document.split('\n')
    line_counts = Counter(lines)
    if max(line_counts.values()) > 3:
        return False

    # Symbol-to-word ratio
    symbols = sum(1 for w in words if not w.isalpha())
    if symbols / max(len(words), 1) > 0.3:
        return False

    return True
```

### Deduplication

- **Exact deduplication**: Hash-based removal of identical documents
- **Near-deduplication**: MinHash + LSH to detect near-duplicate documents
- **Substring deduplication**: Suffix array-based removal of repeated passages

### Domain Mixing

| Source | LLaMA Mix | Typical Weight |
|--------|-----------|---------------|
| CommonCrawl | 67% | 60-80% |
| C4 | 15% | 10-20% |
| GitHub | 4.5% | 3-10% |
| Wikipedia | 4.5% | 3-5% |
| Books | 4.5% | 3-5% |
| ArXiv | 2.5% | 1-3% |
| StackExchange | 2.0% | 1-3% |

## Data Quality vs. Quantity

1. **Quality > Quantity**: The Phi series (Microsoft) demonstrated that carefully curated "textbook-quality" data can match much larger models trained on noisier data
2. **Diminishing returns**: Beyond ~1T tokens, marginal improvements from additional data decrease
3. **Data repetition**: Training for multiple epochs on limited data degrades performance (Muennighoff et al., 2023)

## Contamination and Benchmark Integrity

**Contamination** occurs when benchmark test sets appear in training data, inflating reported metrics. Detection uses N-gram overlap analysis; mitigation includes held-out decontamination sets, canary strings, and dynamic benchmarks.

## Financial Data Considerations

For finance-specific LLMs:

- **SEC filings**: EDGAR database provides decades of 10-K, 10-Q, 8-K filings
- **Financial news**: Reuters, Bloomberg, WSJ archives
- **Research reports**: Analyst reports (often proprietary)
- **Regulatory text**: FINRA, SEC, CFTC rule texts

Most high-quality financial data is proprietary. Public financial LLMs (FinGPT, BloombergGPT) typically underperform on specialized tasks compared to models with proprietary data access.

## References

1. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*.
2. Penedo, G., et al. (2023). "The RefinedWeb Dataset for Falcon LLM." *NeurIPS*.
3. Lee, K., et al. (2022). "Deduplicating Training Data Makes Language Models Better." *ACL*.
4. Muennighoff, N., et al. (2023). "Scaling Data-Constrained Language Models." *NeurIPS*.
