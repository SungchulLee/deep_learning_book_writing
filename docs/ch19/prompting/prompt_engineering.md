# Prompt Engineering

## Learning Objectives

- Master core principles of effective prompt design
- Apply structural patterns for consistent LLM outputs
- Design financial domain-specific prompt templates

## Core Principles

### 1. Be Specific and Unambiguous

```
# Bad
Analyze this stock.

# Good
Analyze AAPL's Q3 FY2024 earnings report. Focus on:
1. Revenue growth vs. consensus estimates
2. Services segment margin trends
3. Forward guidance changes from Q2
```

### 2. Provide Output Format Specifications

```
Extract financial metrics from the text. Return JSON:
{
    "revenue": {"value": float, "unit": "USD_millions", "yoy_change": float},
    "eps": {"value": float, "adjusted": bool},
    "guidance": {"raised": bool, "details": string}
}
```

### 3. Use Role Prompting

```
You are a senior quantitative analyst at a systematic hedge fund.
You specialize in factor-based equity strategies. When analyzing data:
- Consider statistical significance
- Evaluate out-of-sample robustness
- Account for transaction costs
```

### 4. Set Constraints and Guardrails

```
Rules:
- Do NOT speculate about future price movements
- Cite specific numbers from the filing
- If unavailable, say "Not disclosed"
- Tag confidence: HIGH (stated), MEDIUM (inferred), LOW (estimated)
```

## Structural Patterns

### Delimited Input

```
Summarize the earnings call between the triple backticks.
Focus on margin outlook.

\```
[transcript text]
\```
```

### Step-by-Step Decomposition

```
Analyze this M&A announcement in three steps:
Step 1 - Deal Structure: acquirer, target, price, premium, payment
Step 2 - Strategic Rationale: why this deal makes sense
Step 3 - Risk Assessment: top 3 risks to completion
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Vague instructions | Specify aspect, timeframe, format |
| No output format | Provide JSON schema or template |
| Information overload | Extract relevant sections first |
| Missing edge cases | Add "If unavailable, state N/A" |

## References

1. White, J., et al. (2023). "A Prompt Pattern Catalog to Enhance Prompt Engineering." *arXiv*.
