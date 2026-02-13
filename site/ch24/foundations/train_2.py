"""
Training Script for Character-Level Language Model

This script demonstrates:
1. Loading and preparing text data
2. Training an autoregressive character model
3. Generating new text samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CharRNN, SimpleCharTransformer
from data import CharacterDataset, load_sample_text, train_test_split


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The language model
        dataloader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        # Move to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        if isinstance(model, CharRNN):
            output, _ = model(batch_x)
        else:  # Transformer
            output = model(batch_x)
        
        # Compute loss
        loss = criterion(output, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        # This is especially important for RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            criterion: nn.Module,
            device: str) -> float:
    """
    Evaluate the model on test data.
    
    Args:
        model: The language model
        dataloader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss on test set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if isinstance(model, CharRNN):
                output, _ = model(batch_x)
            else:
                output = model(batch_x)
            
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_text(model: nn.Module,
                 dataset: CharacterDataset,
                 seed_text: str,
                 length: int = 200,
                 temperature: float = 0.8,
                 device: str = 'cpu') -> str:
    """
    Generate text autoregressively from a seed.
    
    Args:
        model: Trained language model
        dataset: Dataset (for encoding/decoding)
        seed_text: Starting text
        length: Number of characters to generate
        temperature: Sampling temperature
        device: Device to run on
        
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode seed text
    seed_indices = dataset.encode(seed_text)
    seed_tensor = torch.LongTensor(seed_indices).to(device)
    
    # Generate
    if isinstance(model, CharRNN):
        generated = model.generate(seed_tensor, length, temperature)
    else:
        # For transformer, we need to implement generation differently
        # This is a simplified version
        generated = seed_tensor.clone()
        
        with torch.no_grad():
            for _ in range(length):
                # Take last sequence_length characters
                input_seq = generated[-dataset.sequence_length:].unsqueeze(0)
                
                # Predict next character
                output = model(input_seq)
                output = output / temperature
                probs = torch.softmax(output, dim=-1)
                next_char = torch.multinomial(probs, 1).squeeze()
                
                # Append to generated sequence
                generated = torch.cat([generated, next_char.unsqueeze(0)])
    
    # Decode to text
    generated_text = dataset.decode(generated.cpu().tolist())
    
    return generated_text


def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("Character-Level Autoregressive Language Model Training")
    print("=" * 70)
    
    # ==================== Setup ====================
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    SEQUENCE_LENGTH = 50  # Length of input sequences
    BATCH_SIZE = 64       # Batch size
    EMBEDDING_DIM = 128   # Embedding dimension
    HIDDEN_DIM = 256      # Hidden dimension (for RNN)
    N_LAYERS = 2          # Number of layers
    N_EPOCHS = 50         # Training epochs
    LEARNING_RATE = 0.001
    
    print(f"\nHyperparameters:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ==================== Load Data ====================
    print(f"\n{'='*70}")
    print("Step 1: Loading and preparing text data...")
    print(f"{'='*70}")
    
    # Load sample text (Shakespeare)
    text = load_sample_text()
    
    # You can also load from a file:
    # text = load_text_file('your_text_file.txt')
    
    print(f"\nSample of text:")
    print(text[:200])
    print()
    
    # Create dataset
    dataset = CharacterDataset(text, sequence_length=SEQUENCE_LENGTH)
    
    # Create sequences
    X, y = dataset.create_sequences()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.9)
    
    print(f"\nData prepared:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    print(f"  Vocabulary size: {dataset.vocab_size}")
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # ==================== Initialize Model ====================
    print(f"\n{'='*70}")
    print("Step 2: Initializing model...")
    print(f"{'='*70}")
    
    # Choose model type
    # Option 1: RNN (LSTM)
    model = CharRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS
    ).to(device)
    model_name = "CharRNN"
    
    # Option 2: Transformer (uncomment to use)
    # model = SimpleCharTransformer(
    #     vocab_size=dataset.vocab_size,
    #     embedding_dim=EMBEDDING_DIM,
    #     n_heads=4,
    #     n_layers=N_LAYERS,
    #     max_seq_length=SEQUENCE_LENGTH
    # ).to(device)
    # model_name = "Transformer"
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==================== Training ====================
    print(f"\n{'='*70}")
    print("Step 3: Training model...")
    print(f"{'='*70}\n")
    
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            
            # Generate sample text
            seed = "To be"
            sample = generate_text(model, dataset, seed, length=100, 
                                 temperature=0.8, device=device)
            print(f"\n  Sample generation (seed: '{seed}'):")
            print(f"  {sample}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final test loss: {test_losses[-1]:.4f}")
    
    # ==================== Visualization ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name}: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✓ Saved training_history.png")
    
    # ==================== Generate Samples ====================
    print(f"\n{'='*70}")
    print("Step 5: Generating text samples...")
    print(f"{'='*70}")
    
    seeds = ["To be", "Whether", "And by"]
    temperatures = [0.5, 0.8, 1.2]
    
    print("\nGenerated samples with different temperatures:\n")
    
    for temp in temperatures:
        print(f"Temperature: {temp}")
        print("-" * 70)
        for seed in seeds:
            sample = generate_text(model, dataset, seed, length=150,
                                 temperature=temp, device=device)
            print(f"Seed: '{seed}'")
            print(sample)
            print()
        print()
    
    # ==================== Summary ====================
    print(f"{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"\nKey Observations:")
    print(f"1. Lower temperature (0.5) → More conservative, repetitive")
    print(f"2. Medium temperature (0.8) → Balanced creativity")
    print(f"3. Higher temperature (1.2) → More random, creative but chaotic")
    print(f"\nThe model learned to generate text autoregressively,")
    print(f"predicting one character at a time based on previous context!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
