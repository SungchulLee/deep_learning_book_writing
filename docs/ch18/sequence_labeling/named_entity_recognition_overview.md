# Module 38: Named Entity Recognition (NER)

## Overview
Named Entity Recognition (NER) is a fundamental Natural Language Processing (NLP) task that identifies and classifies named entities (such as persons, organizations, locations, dates, etc.) in unstructured text. This module provides a comprehensive introduction to NER from basic rule-based approaches to state-of-the-art transformer models.

## Learning Objectives
By completing this module, you will:
- Understand the fundamentals of NER and sequence labeling
- Learn various tagging schemes (IOB, BIOES)
- Implement rule-based and dictionary-based NER systems
- Build traditional machine learning models for NER (CRF)
- Develop deep learning models (BiLSTM-CRF, Transformers)
- Apply pre-trained models for production use
- Evaluate NER systems using appropriate metrics

## Prerequisites
- Python programming fundamentals
- Basic NLP concepts (tokenization, part-of-speech tagging)
- Understanding of neural networks (Module 20)
- Familiarity with RNNs and LSTMs (Module 28-29)
- Basic knowledge of transformers (Module 26)

## Mathematical Foundations

### 1. Sequence Labeling Problem
NER is formulated as a sequence labeling task where we assign a label to each token in a sequence.

**Problem Statement:**
- Input: Sequence of tokens X = [x₁, x₂, ..., xₙ]
- Output: Sequence of labels Y = [y₁, y₂, ..., yₙ]
- Objective: Find Y* = argmax P(Y|X)

### 2. IOB Tagging Scheme
- **B-TAG**: Beginning of entity
- **I-TAG**: Inside entity (continuation)
- **O**: Outside entity (not part of any entity)

Example: "Barack Obama visited New York"
```
Barack  → B-PER
Obama   → I-PER
visited → O
New     → B-LOC
York    → I-LOC
```

### 3. Conditional Random Fields (CRF)

**Linear-chain CRF:**
```
P(Y|X) = (1/Z(X)) * exp(∑ᵢ ∑ₖ λₖfₖ(yᵢ₋₁, yᵢ, X, i))

where:
- Z(X) is normalization constant
- fₖ are feature functions
- λₖ are learned weights
```

**Feature Functions:**
- Emission features: f(yᵢ, X, i)
- Transition features: f(yᵢ₋₁, yᵢ)

**Viterbi Decoding:**
Used to find the most likely sequence of labels.

### 4. BiLSTM-CRF Architecture

**Forward Pass:**
```
Forward LSTM:  h⃗ᵢ = LSTM(xᵢ, h⃗ᵢ₋₁)
Backward LSTM: h⃖ᵢ = LSTM(xᵢ, h⃖ᵢ₊₁)
Concatenation: hᵢ = [h⃗ᵢ; h⃖ᵢ]
Emission scores: Eᵢ = Whᵢ + b
```

**CRF Layer:**
```
Score(X, Y) = ∑ᵢ Eᵢ(yᵢ) + ∑ᵢ T(yᵢ₋₁, yᵢ)

where:
- E are emission scores from BiLSTM
- T are transition scores (learned)
```

**Loss Function:**
```
L = -log P(Y*|X) = -[Score(X, Y*) - log(∑ᵧ exp(Score(X, Y')))]
```

### 5. Transformer-based NER

**Architecture:**
```
Input: [CLS] x₁ x₂ ... xₙ [SEP]
       ↓
   BERT/RoBERTa/etc.
       ↓
   h₁, h₂, ..., hₙ  (contextualized embeddings)
       ↓
   Linear classifier for each token
       ↓
   y₁, y₂, ..., yₙ
```

**Subword Handling:**
- Only predict on first subword token
- Or aggregate subword predictions

## Module Structure

```
38_named_entity_recognition/
│
├── README.md                          # This file
│
├── beginner/
│   ├── 01_ner_basics.py              # Introduction to NER concepts
│   ├── 02_iob_tagging.py             # Tagging schemes (IOB, BIOES)
│   ├── 03_rule_based_ner.py          # Simple rule-based NER
│   └── 04_dictionary_ner.py          # Dictionary/gazetteer-based NER
│
├── intermediate/
│   ├── 05_feature_extraction.py      # Feature engineering for NER
│   ├── 06_crf_ner.py                 # Conditional Random Fields
│   ├── 07_evaluation_metrics.py      # Precision, Recall, F1 for NER
│   └── 08_dataset_creation.py        # Creating and formatting NER datasets
│
├── advanced/
│   ├── 09_bilstm_ner.py              # BiLSTM-based NER
│   ├── 10_bilstm_crf_ner.py          # BiLSTM-CRF model
│   ├── 11_transformer_ner.py         # BERT/RoBERTa for NER
│   ├── 12_custom_entities.py         # Training on custom entity types
│   ├── 13_nested_ner.py              # Handling nested entities
│   ├── 14_multilingual_ner.py        # Cross-lingual NER
│   └── 15_production_ner.py          # Production-ready NER pipeline
│
├── data/
│   ├── sample_data.conll             # Sample CoNLL format data
│   ├── custom_entities.json          # Custom entity examples
│   └── entity_dictionary.txt         # Entity dictionary for lookup
│
├── utils/
│   ├── data_loader.py                # Data loading utilities
│   ├── metrics.py                    # Evaluation metrics
│   └── visualization.py              # Visualization tools
│
└── requirements.txt                   # Python dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv ner_env
source ner_env/bin/activate  # On Windows: ner_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Common Entity Types

### Standard CoNLL Entity Types
- **PER** (Person): Names of people
- **ORG** (Organization): Companies, agencies, institutions
- **LOC** (Location): Countries, cities, geographical regions
- **MISC** (Miscellaneous): Other named entities

### Extended Entity Types
- **DATE**: Temporal expressions
- **TIME**: Time expressions
- **MONEY**: Monetary values
- **PERCENT**: Percentages
- **PRODUCT**: Product names
- **EVENT**: Named events
- **WORK_OF_ART**: Titles of creative works
- **LAW**: Legal documents
- **LANGUAGE**: Language names

## Evaluation Metrics

### Token-level Metrics
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Entity-level Metrics
- **Exact Match**: Entity boundaries and type must match exactly
- **Partial Match**: Overlapping entities count as partial matches
- **Type Match**: Entity type correct, boundaries may differ

### Strict vs Relaxed Evaluation
- **Strict**: Exact boundary and type match required
- **Relaxed**: Partial overlap acceptable

## Best Practices

1. **Data Quality**
   - Ensure consistent annotation guidelines
   - Handle ambiguous cases explicitly
   - Include diverse examples

2. **Model Selection**
   - Rule-based: High precision, low recall, fast
   - CRF: Good balance, interpretable features
   - BiLSTM-CRF: Better context, requires more data
   - Transformers: State-of-the-art, requires significant compute

3. **Feature Engineering (for traditional models)**
   - Word shape features (capitalization patterns)
   - Prefix/suffix features
   - POS tags
   - Context windows
   - Gazetteers/dictionaries

4. **Common Challenges**
   - Ambiguous entities (e.g., "Washington" - person or location?)
   - Nested entities
   - Long-distance dependencies
   - Domain-specific entities
   - Rare entity types

## Usage Examples

### Quick Start - Rule-based NER
```python
from beginner.rule_based_ner import RuleBasedNER

# Initialize
ner = RuleBasedNER()

# Extract entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
entities = ner.extract_entities(text)

for entity in entities:
    print(f"{entity['text']}: {entity['type']}")
```

### Advanced - Transformer-based NER
```python
from advanced.transformer_ner import TransformerNER

# Initialize with pre-trained model
ner = TransformerNER(model_name='bert-base-cased')

# Fine-tune on custom data
ner.train(train_data, val_data, epochs=3)

# Predict
entities = ner.predict(text)
```

## Learning Path

### Week 1: Fundamentals (Beginner)
1. Understanding NER and entity types
2. IOB tagging schemes
3. Rule-based approaches
4. Dictionary-based methods

### Week 2: Traditional ML (Intermediate)
1. Feature extraction techniques
2. Conditional Random Fields
3. Evaluation metrics
4. Dataset preparation

### Week 3-4: Deep Learning (Advanced)
1. BiLSTM for sequence labeling
2. BiLSTM-CRF architecture
3. Transformer-based models
4. Custom entity training
5. Production deployment

## Applications

- **Information Extraction**: Extracting structured information from text
- **Question Answering**: Identifying entities in questions and documents
- **Content Classification**: Categorizing documents based on entities
- **Knowledge Graph Construction**: Building entity relationships
- **Document Understanding**: Analyzing contracts, medical records, legal documents
- **Customer Support**: Extracting key information from customer queries
- **Financial Analysis**: Identifying companies, products, metrics in reports

## References

### Papers
1. "Neural Architectures for Named Entity Recognition" (Lample et al., 2016)
2. "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (Ma & Hovy, 2016)
3. "BERT for Named Entity Recognition" (Devlin et al., 2018)
4. "Named Entity Recognition with Bidirectional LSTM-CNNs" (Chiu & Nichols, 2016)

### Datasets
- **CoNLL 2003**: Standard benchmark for NER
- **OntoNotes 5.0**: Large-scale corpus with 18 entity types
- **WikiNER**: Automatically annotated from Wikipedia
- **Few-NERD**: Few-shot NER dataset

### Tools & Libraries
- **spaCy**: Production-ready NER
- **Hugging Face Transformers**: State-of-the-art models
- **Stanford NER**: Java-based NER system
- **NLTK**: Basic NER functionality

## Troubleshooting

### Common Issues
1. **Low Recall**: Add more training data, improve tokenization
2. **Low Precision**: Refine entity definitions, add negative examples
3. **Overfitting**: Use regularization, more training data, data augmentation
4. **Slow Inference**: Use quantization, distillation, or lighter models

## Contributing
This module is part of a larger deep learning curriculum. Suggestions for improvements are welcome.

## License
Educational use only.

---

**Next Module**: 39_principal_component_analysis  
**Previous Module**: 37_language_modeling
