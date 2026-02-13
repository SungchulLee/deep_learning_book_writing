"""Tutorial 27: Model Checkpointing - Saving and loading models"""
import torch
import torch.nn as nn
import torch.optim as optim
import os

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def main():
    # Create temporary directory for checkpoints
    os.makedirs('/home/claude/checkpoints', exist_ok=True)
    
    header("1. Saving Model State Dict")
    model = SimpleModel()
    print("Model created:")
    print(model)
    
    torch.save(model.state_dict(), '/home/claude/checkpoints/model_weights.pth')
    print("\nModel weights saved to 'model_weights.pth'")
    print("This saves only the parameters, not the architecture!")
    
    header("2. Loading Model State Dict")
    new_model = SimpleModel()  # Must create architecture first
    new_model.load_state_dict(torch.load('/home/claude/checkpoints/model_weights.pth'))
    print("Weights loaded into new model")
    
    # Verify weights match
    param1 = list(model.parameters())[0]
    param2 = list(new_model.parameters())[0]
    print(f"Weights match: {torch.equal(param1, param2)}")
    
    header("3. Saving Entire Model")
    torch.save(model, '/home/claude/checkpoints/full_model.pth')
    print("Full model saved (architecture + weights)")
    
    loaded_model = torch.load('/home/claude/checkpoints/full_model.pth')
    print("Full model loaded")
    print("Note: This requires the model class definition to be available!")
    
    header("4. Saving Training Checkpoint")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    epoch = 10
    loss = 0.123
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, '/home/claude/checkpoints/training_checkpoint.pth')
    print("Training checkpoint saved with:")
    print(f"  - Epoch: {epoch}")
    print(f"  - Model weights")
    print(f"  - Optimizer state")
    print(f"  - Loss: {loss}")
    
    header("5. Resuming Training")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    
    checkpoint = torch.load('/home/claude/checkpoints/training_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Last loss: {last_loss}")
    print("Ready to continue training!")
    
    header("6. Save Best Model Only")
    best_loss = float('inf')
    current_loss = 0.1
    
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save(model.state_dict(), '/home/claude/checkpoints/best_model.pth')
        print(f"New best model saved! Loss: {best_loss:.4f}")
    
    header("7. Model Versioning")
    epoch = 5
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }, f'/home/claude/checkpoints/model_epoch_{epoch}.pth')
    print(f"Checkpoint saved: model_epoch_{epoch}.pth")
    print("Useful for comparing different training stages!")
    
    header("8. Saving for Inference Only")
    model.eval()  # Set to evaluation mode
    torch.save(model.state_dict(), '/home/claude/checkpoints/inference_model.pth')
    print("Inference-only model saved")
    print("Remember to call model.eval() before inference!")
    
    header("9. Cross-Platform Compatibility")
    print("""
    For maximum compatibility:
    
    # Save
    torch.save(model.state_dict(), 'model.pth', _use_new_zipfile_serialization=True)
    
    # Load with device mapping
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    
    # Then move to desired device
    model = model.to(device)
    """)
    
    header("10. Best Practices")
    print("""
    Model Checkpointing Best Practices:
    
    1. Save state_dict (not full model) for flexibility
    2. Save training state (epoch, optimizer, loss)
    3. Keep only N best checkpoints to save space
    4. Use descriptive filenames (epoch, metric, date)
    5. Checkpoint after every epoch or N steps
    6. Verify checkpoint loads correctly after saving
    7. Use map_location='cpu' for cross-device loading
    8. Save additional metadata (hyperparameters, etc.)
    9. Test your loading code regularly
    10. Keep separate inference-only checkpoints
    
    Example naming: model_epoch50_loss0.123_acc0.95.pth
    """)
    
    # Clean up
    import shutil
    if os.path.exists('/home/claude/checkpoints'):
        shutil.rmtree('/home/claude/checkpoints')
    print("\nCheckpoint files cleaned up.")

if __name__ == "__main__":
    main()
