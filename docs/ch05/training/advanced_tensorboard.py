"""
Advanced MNIST Training with Enhanced TensorBoard Features

This script demonstrates advanced TensorBoard usage including:
- Hyperparameter logging
- Confusion matrix visualization
- Per-class metrics
- Model weight and gradient histograms
- Learning rate scheduling
- Model checkpointing

Usage:
    python advanced_tensorboard.py
    tensorboard --logdir=runs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# Import our custom modules
from config import Config
from utils import (train_one_epoch, evaluate_model, plot_confusion_matrix,
                   plot_per_class_accuracy, save_checkpoint, count_parameters,
                   print_classification_report)


# ============================================================================
# MODEL DEFINITION (Same as before)
# ============================================================================
class NeuralNet(nn.Module):
    """Simple fully connected neural network for MNIST classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# ============================================================================
# ADVANCED TRAINING FUNCTION
# ============================================================================
def train_with_advanced_logging(config):
    """
    Train the model with comprehensive TensorBoard logging.
    
    Args:
        config (Config): Configuration object containing all hyperparameters
    """
    # Print configuration
    config.print_config()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(config.tensorboard_log_dir)
    print(f"\nTensorBoard logs will be saved to: {config.tensorboard_log_dir}")
    print("To view: tensorboard --logdir=runs\n")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("Loading MNIST dataset...")
    
    # Define transforms (can add augmentation here)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # Optional normalization
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_dir,
        train=False,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_test,
        num_workers=config.num_workers
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # ========================================================================
    # LOG SAMPLE IMAGES TO TENSORBOARD
    # ========================================================================
    examples = iter(test_loader)
    example_data, example_labels = next(examples)
    
    # Create image grid
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('MNIST_Sample_Images', img_grid)
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    print("Initializing model...")
    model = NeuralNet(config.input_size, config.hidden_size, config.num_classes)
    model = model.to(config.device)
    
    # Print model summary
    print(f"\nModel Architecture:\n{model}")
    total_params = count_parameters(model)
    
    # ========================================================================
    # LOG MODEL GRAPH TO TENSORBOARD
    # ========================================================================
    writer.add_graph(model, example_data.reshape(-1, 28*28).to(config.device))
    
    # ========================================================================
    # LOSS FUNCTION AND OPTIMIZER
    # ========================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = config.get_optimizer(model.parameters())
    
    # Learning rate scheduler (optional but recommended)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=3,  # Reduce LR every 3 epochs
        gamma=0.5     # Multiply LR by 0.5
    )
    
    # ========================================================================
    # LOG HYPERPARAMETERS TO TENSORBOARD
    # ========================================================================
    # This creates a nice table in TensorBoard's HPARAMS tab
    hparam_dict = {
        'hidden_size': config.hidden_size,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'optimizer': config.optimizer_type,
        'total_parameters': total_params
    }
    
    # We'll fill in metrics after training
    metric_dict = {
        'final_test_accuracy': 0,
        'final_test_loss': 0
    }
    
    # ========================================================================
    # TRAINING LOOP WITH ADVANCED LOGGING
    # ========================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_accuracy = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        print(f"{'='*70}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}\n")
        
        # Log learning rate to TensorBoard
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # ====================================================================
        # TRAIN FOR ONE EPOCH
        # ====================================================================
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device,
            epoch, config.num_epochs, writer, config.log_interval
        )
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # ====================================================================
        # EVALUATE ON TEST SET
        # ====================================================================
        test_loss, test_acc, predictions, labels, probabilities = evaluate_model(
            model, test_loader, criterion, config.device, writer, epoch
        )
        
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # ====================================================================
        # LOG MODEL WEIGHTS AND GRADIENTS (Every epoch)
        # ====================================================================
        for name, param in model.named_parameters():
            # Log weight distributions
            writer.add_histogram(f'Weights/{name}', param, epoch)
            
            # Log gradient distributions (if gradients exist)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # ====================================================================
        # LOG CONFUSION MATRIX (Every epoch)
        # ====================================================================
        fig_cm = plot_confusion_matrix(
            labels, predictions,
            class_names=[str(i) for i in range(10)]
        )
        writer.add_figure('Confusion_Matrix', fig_cm, epoch)
        plt.close(fig_cm)
        
        # ====================================================================
        # LOG PER-CLASS ACCURACY (Every epoch)
        # ====================================================================
        fig_per_class = plot_per_class_accuracy(
            labels, predictions,
            class_names=[str(i) for i in range(10)]
        )
        writer.add_figure('Per_Class_Accuracy', fig_per_class, epoch)
        plt.close(fig_per_class)
        
        # ====================================================================
        # LOG PRECISION-RECALL CURVES (Every epoch)
        # ====================================================================
        for i in range(10):
            # Binary labels: True if sample is class i
            class_labels = labels == i
            # Probabilities for class i
            class_probs = probabilities[:, i]
            
            # Add PR curve
            writer.add_pr_curve(
                f'PR_Curve/Class_{i}',
                class_labels,
                class_probs,
                global_step=epoch
            )
        
        # ====================================================================
        # PRINT DETAILED CLASSIFICATION REPORT
        # ====================================================================
        if epoch == config.num_epochs - 1:  # Print on last epoch
            print_classification_report(labels, predictions)
        
        # ====================================================================
        # SAVE BEST MODEL CHECKPOINT
        # ====================================================================
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if config.save_model:
                save_checkpoint(
                    model, optimizer, epoch, test_loss, test_acc,
                    config.model_save_path
                )
        
        # ====================================================================
        # UPDATE LEARNING RATE
        # ====================================================================
        scheduler.step()
    
    # ========================================================================
    # LOG FINAL HYPERPARAMETERS WITH METRICS
    # ========================================================================
    metric_dict['final_test_accuracy'] = test_accuracies[-1]
    metric_dict['final_test_loss'] = test_losses[-1]
    
    writer.add_hparams(
        hparam_dict,
        metric_dict
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nBest Test Accuracy: {best_accuracy*100:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]*100:.2f}%")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"\nTensorBoard logs saved to: {config.tensorboard_log_dir}")
    print("To view results, run: tensorboard --logdir=runs")
    print("="*70 + "\n")
    
    # Close TensorBoard writer
    writer.close()
    
    return model, train_losses, test_losses, train_accuracies, test_accuracies


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main function to run the training with advanced logging."""
    
    # ========================================================================
    # SELECT CONFIGURATION
    # ========================================================================
    # You can use different configuration presets:
    # - Config() - default configuration
    # - FastExperiment() - quick test
    # - DeepNetworkExperiment() - larger network
    # - HighLearningRateExperiment() - higher learning rate
    # - etc.
    
    from config import Config, DeepNetworkExperiment
    
    # Use default configuration
    config = Config()
    
    # Or use a preset experiment
    # config = DeepNetworkExperiment()
    
    # Or customize on the fly
    # config.num_epochs = 5
    # config.hidden_size = 1000
    # config.tensorboard_log_dir = 'runs/custom_experiment'
    
    # ========================================================================
    # RUN TRAINING
    # ========================================================================
    model, train_losses, test_losses, train_accs, test_accs = train_with_advanced_logging(config)
    
    # ========================================================================
    # OPTIONAL: PLOT TRAINING HISTORY
    # ========================================================================
    if config.num_epochs > 1:
        from utils import plot_training_history
        fig = plot_training_history(
            train_losses, train_accs,
            test_losses, test_accs,
            save_path='training_history.png'
        )
        plt.show()


if __name__ == '__main__':
    main()


# ============================================================================
# ADDITIONAL TIPS FOR USING TENSORBOARD
# ============================================================================
"""
1. COMPARING MULTIPLE RUNS:
   - Run experiments with different hyperparameters
   - Each run gets saved to a different directory
   - View all together: tensorboard --logdir=runs
   - Use regex filtering in TensorBoard UI

2. REMOTE SERVER USAGE:
   - On server: tensorboard --logdir=runs --host=0.0.0.0 --port=6006
   - On local: ssh -L 6006:localhost:6006 user@server
   - Access: http://localhost:6006

3. TENSORBOARD.DEV (SHARING RESULTS):
   - Upload logs: tensorboard dev upload --logdir=runs
   - Get shareable link
   - Great for collaboration!

4. PLUGIN FEATURES:
   - SCALARS: Training metrics over time
   - IMAGES: Visualize data and predictions
   - GRAPHS: Model architecture
   - DISTRIBUTIONS: Weight and gradient distributions
   - HISTOGRAMS: Detailed distribution over time
   - HPARAMS: Hyperparameter comparison table
   - PR CURVES: Precision-recall curves
   - PROJECTOR: High-dimensional embeddings (advanced)

5. PERFORMANCE TIPS:
   - Don't log too frequently (every 100 steps is usually good)
   - Limit histogram logging (every epoch is sufficient)
   - Use writer.flush() periodically for real-time updates
   - Close writer with writer.close() when done

6. DEBUGGING WITH TENSORBOARD:
   - Check gradient norms for exploding/vanishing gradients
   - Monitor weight distributions for proper initialization
   - Use histograms to detect dead neurons (all zeros)
   - Compare train/test metrics for overfitting
"""
