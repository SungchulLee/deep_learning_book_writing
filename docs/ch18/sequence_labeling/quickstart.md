# Quick Start Guide - Module 38: Named Entity Recognition

## 5-Minute Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for advanced features)
python -m spacy download en_core_web_sm
```

### 2. Run the Complete Demo
```bash
python demo_complete_ner.py
```

This will run demonstrations of all NER approaches covered in the module.

### 3. Try Individual Modules

#### Rule-Based NER (Beginner)
```bash
python beginner/03_rule_based_ner.py
```

#### Dictionary-Based NER (Beginner)
```bash
python beginner/04_dictionary_ner.py
```

#### Feature Extraction (Intermediate)
```bash
python intermediate/05_feature_extraction.py
```

## Learning Path

### Week 1: Fundamentals (4-6 hours)
- **Day 1-2**: `01_ner_basics.py` - Understanding NER concepts
- **Day 3-4**: `02_iob_tagging.py` - Tagging schemes
- **Day 5-6**: `03_rule_based_ner.py` and `04_dictionary_ner.py`
- **Day 7**: Practice and review

### Week 2: Traditional ML (6-8 hours)
- **Day 1-2**: `05_feature_extraction.py` - Feature engineering
- **Day 3-4**: `06_crf_ner.py` - CRF models
- **Day 5-6**: `07_evaluation_metrics.py` - Evaluation
- **Day 7**: `08_dataset_creation.py` - Data preparation

### Week 3-4: Deep Learning (10-15 hours)
- **Week 3**: `10_bilstm_crf_ner.py` - BiLSTM-CRF implementation
- **Week 4**: `11_transformer_ner.py` - Transformer models
- **Final**: `15_production_ner.py` - Production pipeline

## Quick Examples

### Example 1: Extract Entities from Text
```python
from beginner.rule_based_ner_03_rule_based_ner import RuleBasedNER

ner = RuleBasedNER()
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
entities = ner.extract_entities(text)

for entity in entities:
    print(f"{entity['text']}: {entity['type']}")
```

### Example 2: Evaluate NER Performance
```python
from intermediate.evaluation_metrics_07_evaluation_metrics import NERMetrics

y_true = [["B-PER", "I-PER", "O", "B-ORG"]]
y_pred = [["B-PER", "I-PER", "O", "B-ORG"]]

metrics = NERMetrics.compute_metrics(y_true, y_pred)
print(f"F1 Score: {metrics['f1']:.3f}")
```

## Common Issues

### Issue: Import errors
**Solution**: Make sure you're running from the module root directory and have installed all requirements.

### Issue: Missing spaCy models
**Solution**: Run `python -m spacy download en_core_web_sm`

### Issue: CUDA/GPU errors in deep learning modules
**Solution**: Set `device='cpu'` in model initialization if GPU not available

## Next Steps

1. **Complete all beginner modules** to understand fundamentals
2. **Practice with intermediate modules** on your own data
3. **Experiment with advanced modules** for production use
4. **Read the full README.md** for comprehensive documentation

## Getting Help

- Check module docstrings for detailed explanations
- Review inline comments in each file
- Refer to README.md for mathematical foundations
- See demo_complete_ner.py for usage examples

## Module Structure
```
38_named_entity_recognition/
├── beginner/           # Start here
├── intermediate/       # Move here next
├── advanced/          # Advanced implementations
├── utils/             # Helper functions
├── data/              # Sample data
└── README.md          # Full documentation
```

Happy learning!
