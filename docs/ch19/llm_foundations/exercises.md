# 6.11.10 Hands-On Exercises

These exercises provide practical experience with building, fine-tuning, and evaluating conversational AI systems. Each exercise includes complete, runnable Python code with detailed explanations.

!!! note "Prerequisites"
    These exercises require PyTorch, Hugging Face Transformers, and basic Python proficiency. Install dependencies with:
    ```bash
    pip install torch transformers datasets evaluate rouge-score bert-score
    ```

---

## Exercise 1: Building a Rule-Based Chatbot

Before working with neural models, it is instructive to build a simple **rule-based chatbot** to understand the fundamental concepts of intent recognition and response generation.

### 1.1 Pattern-Matching Chatbot

```python
import re
from typing import Optional

class RuleBasedChatbot:
    """A simple pattern-matching chatbot demonstrating intent recognition."""
    
    def __init__(self):
        # Each rule: (compiled regex pattern, response template)
        self.rules = [
            (re.compile(r"\b(hello|hi|hey|greetings)\b", re.IGNORECASE),
             "Hello! How can I help you today?"),
            (re.compile(r"\b(bye|goodbye|quit|exit)\b", re.IGNORECASE),
             "Goodbye! Have a great day."),
            (re.compile(r"book.*flight.*to\s+(.+)", re.IGNORECASE),
             "I'd be happy to help you book a flight to {0}. What dates work for you?"),
            (re.compile(r"weather.*in\s+(.+)", re.IGNORECASE),
             "Let me check the weather in {0} for you."),
            (re.compile(r"(order|tracking)\s*(status|number)?\s*#?(\w+)", re.IGNORECASE),
             "Let me look up the status of order {2}."),
            (re.compile(r"\b(help|support)\b", re.IGNORECASE),
             "I can help with: booking flights, checking weather, "
             "and order tracking. What do you need?"),
        ]
        self.default_response = (
            "I'm not sure I understand. Could you rephrase that? "
            "Type 'help' to see what I can do."
        )
    
    def respond(self, user_input: str) -> str:
        """Match user input against rules and return a response."""
        for pattern, response_template in self.rules:
            match = pattern.search(user_input)
            if match:
                # Fill in captured groups if the template uses them
                return response_template.format(*match.groups())
        return self.default_response


# Interactive demo
if __name__ == "__main__":
    bot = RuleBasedChatbot()
    print("Rule-Based Chatbot (type 'bye' to exit)")
    print("-" * 45)
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        response = bot.respond(user_input)
        print(f"Bot: {response}")
        if re.search(r"\b(bye|goodbye|quit|exit)\b", user_input, re.IGNORECASE):
            break
```

### 1.2 Discussion

This exercise demonstrates the **limitations** of rule-based approaches:

- **Brittle pattern matching.** Small variations in phrasing (e.g., "I want to fly to Paris" vs. "Book me a flight to Paris") require separate rules or increasingly complex regex patterns.
- **No generalization.** The system cannot handle queries outside its predefined rules.
- **No context.** Each turn is processed independently; the system has no memory of previous exchanges.

These limitations motivate the use of neural language models for conversational AI.

---

## Exercise 2: Text Generation with GPT-2

This exercise demonstrates autoregressive text generation using a pre-trained GPT-2 model with different decoding strategies.

### 2.1 Basic Generation with Decoding Strategies

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(
    prompt: str,
    strategy: str = "nucleus",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    num_return_sequences: int = 1,
    model_name: str = "gpt2",
) -> list[str]:
    """
    Generate text using GPT-2 with configurable decoding strategies.
    
    Args:
        prompt: Input text to condition generation on.
        strategy: One of 'greedy', 'temperature', 'top_k', 'nucleus'.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Number of top tokens to consider for top-k sampling.
        top_p: Cumulative probability threshold for nucleus sampling.
        num_return_sequences: Number of independent sequences to generate.
        model_name: Hugging Face model identifier.
    
    Returns:
        List of generated text strings.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Configure generation parameters based on strategy
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if strategy == "greedy":
        gen_kwargs["do_sample"] = False
    elif strategy == "temperature":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    elif strategy == "top_k":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_k"] = top_k
        gen_kwargs["temperature"] = temperature
    elif strategy == "nucleus":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = top_p
        gen_kwargs["temperature"] = temperature
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]


# Compare decoding strategies
if __name__ == "__main__":
    prompt = "The future of artificial intelligence is"
    
    strategies = ["greedy", "temperature", "top_k", "nucleus"]
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        results = generate_text(prompt, strategy=strategy, max_new_tokens=60)
        for i, text in enumerate(results):
            print(f"\n{text}")
```

### 2.2 Observing the Effect of Temperature

```python
# Temperature comparison: factual vs. creative settings
prompt = "Machine learning models learn by"

print("Low temperature (tau=0.2) — more deterministic, focused:")
for text in generate_text(prompt, strategy="nucleus", temperature=0.2, top_p=0.9):
    print(f"  {text}\n")

print("Medium temperature (tau=0.7) — balanced:")
for text in generate_text(prompt, strategy="nucleus", temperature=0.7, top_p=0.9):
    print(f"  {text}\n")

print("High temperature (tau=1.2) — more creative, diverse:")
for text in generate_text(prompt, strategy="nucleus", temperature=1.2, top_p=0.95):
    print(f"  {text}\n")
```

---

## Exercise 3: Supervised Fine-Tuning (SFT)

This exercise demonstrates fine-tuning GPT-2 on a small instruction-following dataset — the first step of the ChatGPT alignment pipeline.

### 3.1 Preparing an Instruction Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

class InstructionDataset(Dataset):
    """A simple instruction-response dataset for SFT demonstration."""
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Small demonstration dataset
        # In practice, this would contain thousands of examples
        self.examples = [
            {"instruction": "Explain what a neural network is.",
             "response": "A neural network is a computational model inspired by "
                        "the structure of biological neurons. It consists of layers "
                        "of interconnected nodes that learn to transform inputs into "
                        "outputs by adjusting connection weights during training."},
            {"instruction": "What is the capital of France?",
             "response": "The capital of France is Paris. It is the largest city "
                        "in France and serves as the country's political, economic, "
                        "and cultural center."},
            {"instruction": "Write a short poem about the ocean.",
             "response": "Beneath the endless sky of blue, the ocean whispers "
                        "something true. Its waves reach out to touch the shore, "
                        "then pull away to rise once more."},
            {"instruction": "Summarize the concept of gradient descent.",
             "response": "Gradient descent is an optimization algorithm that "
                        "iteratively adjusts model parameters to minimize a loss "
                        "function. At each step, it computes the gradient of the loss "
                        "with respect to the parameters and moves in the direction "
                        "of steepest descent, scaled by a learning rate."},
            {"instruction": "What are the benefits of exercise?",
             "response": "Regular exercise improves cardiovascular health, "
                        "strengthens muscles and bones, enhances mental health by "
                        "reducing anxiety and depression, improves sleep quality, "
                        "and helps maintain a healthy body weight."},
        ]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        # Format as instruction-response pair
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['response']}"
            f"{self.tokenizer.eos_token}"
        )
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": input_ids.clone()}
```

### 3.2 Fine-Tuning Loop

```python
def fine_tune_gpt2(
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    model_name: str = "gpt2",
):
    """Fine-tune GPT-2 on instruction-response data."""
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.train()
    
    dataset = InstructionDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Fine-tuning on {len(dataset)} examples for {num_epochs} epochs")
    print(f"Device: {device}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Mask padding tokens in labels (set to -100 so they are ignored)
            labels[attention_mask == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Average Loss: {avg_loss:.4f}")
    
    return model, tokenizer


# Run fine-tuning
if __name__ == "__main__":
    model, tokenizer = fine_tune_gpt2(num_epochs=3)
    
    # Test the fine-tuned model
    model.eval()
    prompt = "### Instruction:\nExplain what machine learning is.\n\n### Response:\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nGenerated response:\n{generated}")
```

---

## Exercise 4: Computing Evaluation Metrics

### 4.1 Perplexity

```python
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compute_perplexity(text: str, model_name: str = "gpt2") -> float:
    """
    Compute the perplexity of a text under a GPT-2 model.
    
    Perplexity is defined as:
        PPL = exp( -1/T * sum_{t=1}^{T} log p(x_t | x_{<t}) )
    
    Args:
        text: Input text to evaluate.
        model_name: Hugging Face model identifier.
    
    Returns:
        Perplexity score (lower is better).
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    T = input_ids.size(1)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is the average negative log-likelihood per token
        neg_log_likelihood = outputs.loss.item()
    
    perplexity = math.exp(neg_log_likelihood)
    
    print(f"Text: '{text[:60]}...' " if len(text) > 60 else f"Text: '{text}'")
    print(f"  Tokens: {T}")
    print(f"  Avg NLL: {neg_log_likelihood:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    
    return perplexity


# Compare perplexity of fluent vs. disfluent text
if __name__ == "__main__":
    # Well-formed English should have lower perplexity
    compute_perplexity("The cat sat on the mat and looked out the window.")
    print()
    compute_perplexity("Mat the on sat cat the window out looked and.")
    print()
    compute_perplexity(
        "Machine learning is a subset of artificial intelligence that "
        "enables systems to learn and improve from experience."
    )
```

### 4.2 BERTScore

```python
from evaluate import load

def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict:
    """
    Compute BERTScore between predictions and references.
    
    BERTScore computes token-level cosine similarity between
    contextual embeddings, then aggregates via greedy matching:
        P_BERT = (1/|pred|) * sum_i max_j cos(e_i, e_j)
        R_BERT = (1/|ref|)  * sum_j max_i cos(e_i, e_j)
        F_BERT = 2 * P * R / (P + R)
    
    Args:
        predictions: List of generated responses.
        references: List of reference responses.
        model_type: Embedding model to use.
    
    Returns:
        Dictionary with precision, recall, and F1 arrays.
    """
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=model_type,
    )
    
    for i, (p, r, f) in enumerate(
        zip(results["precision"], results["recall"], results["f1"])
    ):
        print(f"Pair {i+1}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")
    
    return results


# Example: compare semantic similarity
if __name__ == "__main__":
    predictions = [
        "Paris is the capital city of France.",
        "Neural networks are computational models based on brain structure.",
    ]
    references = [
        "The capital of France is Paris, located in northern France.",
        "A neural network is a machine learning model inspired by neurons.",
    ]
    compute_bertscore(predictions, references)
```

---

## Exercise 5: A/B Testing Framework for Chatbot Evaluation

This exercise implements a lightweight A/B testing framework to compare different chatbot configurations.

```python
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from scipy import stats

@dataclass
class ChatbotConfig:
    """Configuration for a chatbot variant in an A/B test."""
    name: str
    temperature: float
    top_p: float
    system_prompt: str = ""
    
    def __repr__(self):
        return (f"Config(name='{self.name}', temp={self.temperature}, "
                f"top_p={self.top_p})")

@dataclass
class InteractionResult:
    """Result of a single user-chatbot interaction."""
    config_name: str
    query: str
    response: str
    satisfaction_score: float   # 1-5 Likert scale
    response_time_ms: float
    task_completed: bool

class ABTestFramework:
    """
    A/B testing framework for comparing chatbot configurations.
    
    Randomly assigns users to variants and collects metrics
    for statistical comparison.
    """
    
    def __init__(self, configs: list[ChatbotConfig], seed: int = 42):
        self.configs = {c.name: c for c in configs}
        self.results: list[InteractionResult] = []
        self.rng = random.Random(seed)
    
    def assign_variant(self) -> ChatbotConfig:
        """Randomly assign a user to a configuration variant."""
        name = self.rng.choice(list(self.configs.keys()))
        return self.configs[name]
    
    def record_interaction(self, result: InteractionResult):
        """Record the result of an interaction."""
        self.results.append(result)
    
    def analyze(self) -> dict:
        """
        Perform statistical analysis comparing variants.
        
        Returns summary statistics and two-sample t-test results
        for satisfaction scores between all variant pairs.
        """
        # Group results by variant
        groups = {}
        for r in self.results:
            groups.setdefault(r.config_name, []).append(r)
        
        analysis = {"variants": {}, "comparisons": []}
        
        # Per-variant statistics
        for name, results in groups.items():
            scores = [r.satisfaction_score for r in results]
            completion = [r.task_completed for r in results]
            times = [r.response_time_ms for r in results]
            
            analysis["variants"][name] = {
                "n": len(results),
                "mean_satisfaction": np.mean(scores),
                "std_satisfaction": np.std(scores, ddof=1),
                "completion_rate": np.mean(completion),
                "mean_response_time_ms": np.mean(times),
            }
        
        # Pairwise comparisons (two-sample t-test)
        variant_names = list(groups.keys())
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                name_a, name_b = variant_names[i], variant_names[j]
                scores_a = [r.satisfaction_score for r in groups[name_a]]
                scores_b = [r.satisfaction_score for r in groups[name_b]]
                
                t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                
                analysis["comparisons"].append({
                    "variant_a": name_a,
                    "variant_b": name_b,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant_at_005": p_value < 0.05,
                    "mean_diff": np.mean(scores_a) - np.mean(scores_b),
                })
        
        return analysis
    
    def print_report(self):
        """Print a formatted analysis report."""
        analysis = self.analyze()
        
        print("=" * 60)
        print("A/B TEST REPORT")
        print("=" * 60)
        
        for name, stats_dict in analysis["variants"].items():
            print(f"\nVariant: {name}")
            print(f"  Interactions:       {stats_dict['n']}")
            print(f"  Mean Satisfaction:  {stats_dict['mean_satisfaction']:.2f} "
                  f"(± {stats_dict['std_satisfaction']:.2f})")
            print(f"  Completion Rate:    {stats_dict['completion_rate']:.1%}")
            print(f"  Avg Response Time:  {stats_dict['mean_response_time_ms']:.0f} ms")
        
        print(f"\n{'─' * 60}")
        print("Pairwise Comparisons (Welch's t-test)")
        print(f"{'─' * 60}")
        
        for comp in analysis["comparisons"]:
            sig = "✓" if comp["significant_at_005"] else "✗"
            print(f"\n  {comp['variant_a']} vs {comp['variant_b']}")
            print(f"    Mean diff:  {comp['mean_diff']:+.3f}")
            print(f"    t-stat:     {comp['t_statistic']:.3f}")
            print(f"    p-value:    {comp['p_value']:.4f}")
            print(f"    Sig (α=.05): {sig}")


# Demonstration with simulated data
if __name__ == "__main__":
    configs = [
        ChatbotConfig("creative", temperature=1.0, top_p=0.95),
        ChatbotConfig("balanced", temperature=0.7, top_p=0.90),
        ChatbotConfig("factual",  temperature=0.3, top_p=0.80),
    ]
    
    framework = ABTestFramework(configs, seed=42)
    
    # Simulate 150 interactions (50 per variant on average)
    rng = random.Random(42)
    test_queries = [
        "What is quantum computing?",
        "Help me plan a weekend trip.",
        "Explain gradient descent.",
        "Write a creative story opening.",
        "What are the symptoms of the flu?",
    ]
    
    for _ in range(150):
        config = framework.assign_variant()
        query = rng.choice(test_queries)
        
        # Simulate satisfaction scores with variant-dependent distributions
        if config.name == "creative":
            score = min(5, max(1, rng.gauss(3.5, 1.0)))
        elif config.name == "balanced":
            score = min(5, max(1, rng.gauss(3.9, 0.8)))
        else:  # factual
            score = min(5, max(1, rng.gauss(3.7, 0.9)))
        
        result = InteractionResult(
            config_name=config.name,
            query=query,
            response=f"[Simulated response from {config.name}]",
            satisfaction_score=round(score, 1),
            response_time_ms=rng.gauss(200 + config.temperature * 100, 50),
            task_completed=rng.random() < (0.85 if config.name == "balanced" else 0.75),
        )
        framework.record_interaction(result)
    
    framework.print_report()
```

---

## Summary

These exercises progressively build practical skills:

| Exercise | Concept | Key Takeaway |
|----------|---------|--------------|
| 1 | Rule-based chatbot | Limitations of pattern matching motivate neural approaches |
| 2 | GPT-2 generation | Decoding strategies critically affect output quality and diversity |
| 3 | Supervised fine-tuning | SFT adapts a base model to instruction-following behavior |
| 4 | Evaluation metrics | Perplexity measures fluency; BERTScore captures semantic similarity |
| 5 | A/B testing | Statistical rigor is essential for comparing chatbot configurations |

!!! tip "Further Exploration"
    To extend these exercises: (1) scale the SFT dataset using the Alpaca or OpenAssistant datasets, (2) implement a simple reward model using pairwise comparison data, (3) add ROUGE-L to the evaluation metrics, and (4) deploy a chatbot via a Gradio or Streamlit interface for live user testing.
