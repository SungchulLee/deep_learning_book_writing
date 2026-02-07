# Extractive Question Answering

## Learning Objectives

- Understand span extraction as a QA formulation
- Implement BERT-based extractive QA
- Handle unanswerable questions

## Task Formulation

Given a question $q$ and context passage $c$, extract the answer span $a = c_{i:j}$ where $i$ and $j$ are the start and end token positions:

$$\hat{a} = \arg\max_{(i,j): i \leq j \leq i + L_{\max}} P(\text{start}=i \mid q, c) \cdot P(\text{end}=j \mid q, c)$$

## BERT for Extractive QA

BERT (Devlin et al., 2019) encodes the question and context as a single sequence:

```
[CLS] question tokens [SEP] context tokens [SEP]
```

Two linear heads predict start and end positions over the context tokens:

$$P(\text{start}=i) = \frac{\exp(\mathbf{w}_s^T \mathbf{h}_i)}{\sum_k \exp(\mathbf{w}_s^T \mathbf{h}_k)}$$

$$P(\text{end}=j) = \frac{\exp(\mathbf{w}_e^T \mathbf{h}_j)}{\sum_k \exp(\mathbf{w}_e^T \mathbf{h}_k)}$$

### Implementation

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt",
                      max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    # Decode answer span
    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# Example
context = """Tesla reported Q3 2024 revenue of $25.18 billion,
up 8% year-over-year. Net income was $2.17 billion."""

print(answer_question("What was Tesla's Q3 revenue?", context))
# "25.18 billion"
```

### Custom Model Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class ExtractiveQAModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # start, end

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        logits = self.qa_outputs(outputs.last_hidden_state)  # (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

## Handling Unanswerable Questions

SQuAD 2.0 includes unanswerable questions. The model must learn to abstain when the context does not contain the answer.

### No-Answer Score

Compare the best span score against the `[CLS]` score (representing "no answer"):

$$s_{\text{span}} = \max_{i,j} (s_{\text{start},i} + s_{\text{end},j})$$
$$s_{\text{null}} = s_{\text{start},[\text{CLS}]} + s_{\text{end},[\text{CLS}]}$$

Predict "unanswerable" if $s_{\text{null}} > s_{\text{span}} + \tau$ where $\tau$ is a threshold tuned on the dev set.

## Long Document Handling

BERT's 512-token limit requires chunking long documents with sliding windows:

1. Split context into overlapping chunks (stride = 128 tokens)
2. Run QA model on each chunk
3. Select the answer span with the highest confidence across all chunks

## References

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
2. Rajpurkar, P., et al. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. *ACL*.
