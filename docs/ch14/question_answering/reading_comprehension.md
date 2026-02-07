# Reading Comprehension

## Learning Objectives

- Survey major reading comprehension benchmarks
- Understand the challenges of machine reading
- Distinguish between different comprehension skills tested

## What Is Reading Comprehension?

Reading comprehension evaluates whether a model can understand text well enough to answer questions about it. Unlike open-domain QA, the context passage is always provided — the challenge is pure comprehension, not retrieval.

## Key Benchmarks

### SQuAD 1.1 (Rajpurkar et al., 2016)

The foundational extractive QA dataset: 100,000+ questions on 500+ Wikipedia articles. All questions are answerable from the given passage.

- **Task**: Extract answer span from passage
- **Metric**: Exact Match (EM) and F1
- **Human performance**: 82.3 EM / 91.2 F1
- **SOTA**: ~95 EM / 97 F1 (surpasses human)

### SQuAD 2.0 (Rajpurkar et al., 2018)

Adds 50,000+ unanswerable questions designed to look plausible. Models must distinguish "I can answer this" from "the passage doesn't say."

### RACE (Lai et al., 2017)

Multiple-choice reading comprehension from Chinese middle and high school English exams. Tests more complex reasoning: inference, summarization, and attitude detection.

### CoQA (Reddy et al., 2019)

Conversational QA — questions in a dialog context where answers depend on conversation history. Tests coreference resolution and pragmatic understanding.

### Natural Questions (Kwiatkowski et al., 2019)

Real Google search queries paired with Wikipedia articles. Distinguishes "short answers" (entities/phrases) from "long answers" (paragraphs). More realistic than SQuAD since questions were asked before seeing the passage.

## Comprehension Skills

| Skill | Example |
|-------|---------|
| Lexical matching | "capital" in question matches "capital" in passage |
| Paraphrase | "founded" in question, "established" in passage |
| Single-sentence reasoning | Answer within one sentence |
| Multi-sentence reasoning | Must combine information across sentences |
| Coreference | Must resolve "he", "the company" to find answer |
| Numerical reasoning | "How many more?" requires subtraction |
| Temporal reasoning | "What happened before/after X?" |

## Adversarial Evaluation

Models can exploit superficial patterns (lexical overlap, entity type matching) rather than truly comprehending text. Adversarial evaluation adds distracting sentences to test robustness:

- **AddSent** (Jia & Liang, 2017): Append misleading sentences that share words with the question
- **AddOneSent**: Add a single adversarial distractor

These reveal that many models rely on shallow heuristics rather than deep understanding.

## Financial Reading Comprehension

Financial RC presents unique challenges: understanding numerical tables, temporal references in earnings reports, and domain-specific terminology. Datasets like FinQA (Chen et al., 2021) test numerical reasoning over financial documents.

## References

1. Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension. *EMNLP*.
2. Rajpurkar, P., et al. (2018). Know What You Don't Know. *ACL*.
3. Jia, R., & Liang, P. (2017). Adversarial Examples for Evaluating Reading Comprehension Systems. *EMNLP*.
