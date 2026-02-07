# Multi-Hop QA

## Overview

Multi-hop QA requires reasoning over multiple pieces of evidence to answer a question:

"What country is the birthplace of the director of Inception?"
→ Step 1: Director of Inception = Christopher Nolan
→ Step 2: Birthplace of Christopher Nolan = London, England

## Datasets

| Dataset | Hops | Evidence |
|---------|------|---------|
| HotpotQA | 2 | Wikipedia paragraphs |
| MuSiQue | 2-4 | Composable questions |
| 2WikiMultiHopQA | 2 | Wikipedia |

## Approaches

### Chain-of-Thought

Prompt LLMs to reason step-by-step:

```
Q: What country is the birthplace of the director of Inception?
Let me think step by step:
1. The director of Inception is Christopher Nolan.
2. Christopher Nolan was born in London, England.
3. London is in the United Kingdom.
Answer: United Kingdom
```

### Iterative Retrieval

Retrieve evidence, extract partial answer, reformulate query, retrieve more evidence.

### Graph Neural Networks

Model entity relations as a graph, perform reasoning via message passing.

## Summary

1. Multi-hop QA tests compositional reasoning ability
2. Chain-of-thought prompting enables LLMs to perform step-by-step reasoning
3. Iterative retrieval decomposes complex questions into simpler sub-queries
