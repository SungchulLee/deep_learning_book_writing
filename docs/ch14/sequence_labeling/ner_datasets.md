# NER Datasets

## Overview

Standard NER datasets provide benchmarks for comparing models:

| Dataset | Language | Entity Types | Train Sentences | Domain |
|---------|----------|-------------|----------------|--------|
| CoNLL-2003 | English/German | 4 (PER, ORG, LOC, MISC) | 14,987 | News (Reuters) |
| OntoNotes 5.0 | English + 5 langs | 18 | 76,714 | Mixed |
| WNUT-17 | English | 6 | 3,394 | Social media |
| Few-NERD | English | 66 fine-grained | 188,238 | Wikipedia |
| CoNLL++ | English | 4 | 14,987 (corrected) | News |

## Domain-Specific Datasets

**Biomedical**: JNLPBA (gene/protein), NCBI Disease, BC5CDR (chemical-disease), GENIA

**Financial**: FiNER (financial NER), SEC-NER (SEC filings), FinBERT-NER

**Legal**: LegalNER, CUAD (contract understanding)

## Data Format (CoNLL)

```
-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
```

Each line: `token POS chunk NER_tag`, sentences separated by blank lines.

## Key Considerations

1. **Train/dev/test splits** must be respected for fair comparison
2. **Label noise** exists in all benchmarks—CoNLL++ provides corrected annotations
3. **Domain mismatch** between training (news) and target domain is a major challenge
4. **Cross-lingual** evaluation uses CoNLL-2002 (Spanish, Dutch) and WikiANN (282 languages)

---

## References

1. Tjong Kim Sang, E. F. (2003). CoNLL-2003 Shared Task. *CoNLL*.
2. Ding, N., et al. (2021). Few-NERD: A Few-shot NER Dataset. *ACL*.
