"""
Training Script for Sequence-to-Sequence Models
Includes training loop, evaluation, and checkpoint saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import math
from pathlib import Path


class Seq2SeqDataset(Dataset):
    """
    Custom Dataset for Seq2Seq training
    
    Args:
        src_data: List of source sequences (token indices)
        trg_data: List of target sequences (token indices)
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
    """
    
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.trg_data[idx])


def collate_fn(batch, pad_idx=0):
    """
    Collate function for batching variable-length sequences
    
    Args:
        batch: List of (src, trg) tuples
        pad_idx: Padding token index
        
    Returns:
        src_batch: Padded source sequences
        trg_batch: Padded target sequences
        src_lengths: Actual lengths of source sequences
        trg_lengths: Actual lengths of target sequences
    """
    src_batch, trg_batch = zip(*batch)
    
    # Get lengths
    src_lengths = torch.tensor([len(s) for s in src_batch])
    trg_lengths = torch.tensor([len(t) for t in trg_batch])
    
    # Pad sequences
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
    
    return src_batch, trg_batch, src_lengths, trg_lengths


class Seq2SeqTrainer:
    """
    Trainer class for Seq2Seq models
    
    Args:
        model: Seq2Seq model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        pad_idx: Padding token index
        clip: Gradient clipping value
    """
    
    def __init__(self, model, optimizer, criterion, device, pad_idx=0, clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.pad_idx = pad_idx
        self.clip = clip
        
    def train_epoch(self, dataloader, teacher_forcing_ratio=0.5):
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            epoch_loss: Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):
            src = src.to(self.device)
            trg = trg.to(self.device)
            src_lengths = src_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'encoder'):
                # Seq2SeqAttention model
                output, _ = self.model(src, trg, teacher_forcing_ratio, src_lengths)
            else:
                output = self.model(src, trg, teacher_forcing_ratio, src_lengths)
            
            # Reshape output and target for loss calculation
            # output: (batch_size, trg_len, output_dim)
            # trg: (batch_size, trg_len)
            output_dim = output.shape[-1]
            
            # Skip the first token (<sos>) in target
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = self.criterion(output, trg)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # Update parameters
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        Evaluate the model
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            epoch_loss: Average loss for the epoch
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):
                src = src.to(self.device)
                trg = trg.to(self.device)
                src_lengths = src_lengths.to(self.device)
                
                # Forward pass (no teacher forcing during evaluation)
                if hasattr(self.model, 'encoder'):
                    output, _ = self.model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
                else:
                    output = self.model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
                
                # Reshape for loss calculation
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # Calculate loss
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints', 
              teacher_forcing_ratio=0.5, save_every=1):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            teacher_forcing_ratio: Initial teacher forcing ratio
            save_every: Save checkpoint every N epochs
            
        Returns:
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            
            # Evaluate
            val_loss = self.evaluate(val_loader)
            
            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            # Print progress
            print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir / 'best_model.pt'
                self.save_checkpoint(best_checkpoint_path, epoch, train_loss, val_loss)
                print(f'\t[Saved Best Model]')
            
            # Decay teacher forcing ratio (optional)
            teacher_forcing_ratio = max(0.5 * teacher_forcing_ratio, 0.1)
        
        return train_losses, val_losses
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def initialize_weights(model):
    """Initialize model weights"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    from encoder import BasicEncoder
    from decoder import AttentionDecoder
    from seq2seq_model import Seq2SeqAttention
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    INPUT_DIM = 10000
    OUTPUT_DIM = 10000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    PAD_IDX = 0
    
    # Create model
    encoder = BasicEncoder(
        input_size=INPUT_DIM,
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    decoder = AttentionDecoder(
        output_size=OUTPUT_DIM,
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # Initialize weights
    initialize_weights(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Create trainer
    trainer = Seq2SeqTrainer(model, optimizer, criterion, device, PAD_IDX, clip=1.0)
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    num_samples = 1000
    src_data = [np.random.randint(3, INPUT_DIM, size=np.random.randint(10, 30)).tolist() 
                for _ in range(num_samples)]
    trg_data = [np.random.randint(3, OUTPUT_DIM, size=np.random.randint(10, 30)).tolist() 
                for _ in range(num_samples)]
    
    train_dataset = Seq2SeqDataset(src_data[:800], trg_data[:800])
    val_dataset = Seq2SeqDataset(src_data[800:], trg_data[800:])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=5,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5
    )
    
    print("\nTraining completed!")
