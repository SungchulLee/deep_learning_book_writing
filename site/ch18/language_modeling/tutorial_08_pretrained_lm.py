"""
Tutorial 08: Fine-tuning Pretrained Language Models
====================================================

Learn to leverage pretrained models like GPT-2 for downstream tasks.
Transfer learning enables using knowledge from large-scale pretraining.

Key Concepts:
- Pretrained models (GPT-2, BERT, RoBERTa)
- Fine-tuning vs. feature extraction
- Domain adaptation
- Few-shot learning

Pretrained Model Benefits:
1. Learned linguistic knowledge
2. Reduced training time/data
3. Better generalization
4. State-of-the-art performance
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset for fine-tuning pretrained models."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        
        for text in texts:
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.inputs.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]


def finetune_gpt2(train_texts, model_name='gpt2', epochs=3):
    """
    Fine-tune GPT-2 on custom text data.
    
    Args:
        train_texts: List of training texts
        model_name: Pretrained model name ('gpt2', 'gpt2-medium', etc.)
        epochs: Number of training epochs
    """
    print(f"Fine-tuning {model_name}")
    print("=" * 60)
    
    # Load pretrained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Prepare dataset
    train_dataset = TextDataset(train_texts, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=100,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Fine-tune
    trainer.train()
    
    return model, tokenizer


def generate_from_pretrained(model, tokenizer, prompt, max_length=50,
                             temperature=1.0, top_k=50, top_p=0.95):
    """
    Generate text from fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt: Starting text
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    """
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def demonstrate_pretrained():
    """Demonstrate using pretrained GPT-2."""
    
    print("Loading pretrained GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence is",
        "In a distant galaxy"
    ]
    
    print("\nGenerating from pretrained model:\n")
    for prompt in prompts:
        text = generate_from_pretrained(model, tokenizer, prompt, 
                                       max_length=30)
        print(f"Prompt: {prompt}")
        print(f"Generated: {text}\n")


if __name__ == "__main__":
    print("""
Pretrained Language Models
==========================

Popular Models:
1. GPT (Generative Pretrained Transformer)
   - Autoregressive, left-to-right
   - Good for generation tasks
   
2. BERT (Bidirectional Encoder)
   - Masked language modeling
   - Good for understanding tasks

3. T5 (Text-to-Text)
   - Unified framework
   - All tasks as text generation

Fine-tuning Strategies:
1. Full fine-tuning: Update all parameters
2. Adapter modules: Add small trainable layers
3. Prompt tuning: Optimize continuous prompts
4. LoRA: Low-rank adaptation of weights

Best Practices:
- Use smaller learning rate than pretraining
- Freeze early layers for small datasets
- Implement gradient accumulation for large models
- Use mixed precision training
- Monitor for catastrophic forgetting

EXERCISES:
1. Fine-tune GPT-2 on domain-specific text
2. Compare different model sizes (base, medium, large)
3. Implement adapter-based fine-tuning
4. Try prompt-based few-shot learning
5. Evaluate on perplexity and generation quality
6. Implement knowledge distillation from large to small model
    """)
    
    # demonstrate_pretrained()
    print("\nNote: Requires 'transformers' library: pip install transformers")
