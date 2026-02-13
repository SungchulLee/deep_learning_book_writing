# Transfer Learning for Finance

## Text-Based Transfer

Pre-trained language models transfer effectively to financial text tasks. FinBERT (BERT fine-tuned on financial text) handles sentiment analysis on earnings calls and analyst reports. BloombergGPT and FinGPT demonstrate domain-specific LLM capabilities for financial analysis and QA.

## Time Series Transfer

### Cross-Asset Transfer
Models trained on liquid assets (S&P 500 constituents) can be fine-tuned for less liquid assets. Shared factors (momentum, value, size) transfer well; market microstructure features require re-learning.

### Cross-Market Transfer
Models from one market (US equities) adapted to another (emerging markets). Common factors transfer; market-specific features need adaptation.

## Temporal Domain Shift

Financial data is non-stationary. Strategies include rolling fine-tuning (periodic updates on recent data), adversarial domain adaptation (minimize distribution gap between training and deployment periods), and ensemble methods with recency weighting.

## Pre-Trained Models

| Model | Base | Domain | Tasks |
|-------|------|--------|-------|
| FinBERT | BERT | Financial text | Sentiment, NER |
| BloombergGPT | LLM | Bloomberg data | General finance |
| FinGPT | LLaMA | Financial text | Analysis, QA |
| TFT | Transformer | Time series | Forecasting |

## Challenges

Data leakage (pre-training data may include future info), regime dependence (transfer performance varies across market regimes), and limited labeled data for financial prediction tasks.
