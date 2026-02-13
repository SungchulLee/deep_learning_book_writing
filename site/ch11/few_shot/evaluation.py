"""
Evaluation Utilities for Few-Shot Learning

Functions for evaluating few-shot learning models, including metrics,
confidence intervals, and standard evaluation protocols.
"""

import torch
import numpy as np
from scipy import stats


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: (n,) - Predicted class labels
        labels: (n,) - True class labels
    
    Returns:
        accuracy: Float in [0, 1]
    """
    correct = (predictions == labels).float()
    return correct.mean().item()


def compute_confidence_interval(accuracies, confidence=0.95):
    """
    Compute confidence interval for accuracy measurements.
    
    Args:
        accuracies: List or array of accuracy values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        mean: Mean accuracy
        ci: Confidence interval (half-width)
    """
    accuracies = np.array(accuracies)
    mean = np.mean(accuracies)
    std_error = stats.sem(accuracies)
    
    # Compute confidence interval
    ci = std_error * stats.t.ppf((1 + confidence) / 2, len(accuracies) - 1)
    
    return mean, ci


def evaluate_few_shot_model(model, dataloader, n_episodes=600):
    """
    Evaluate a few-shot learning model on multiple episodes.
    
    Args:
        model: Few-shot learning model with forward(support, support_labels, query)
        dataloader: DataLoader that yields (support, support_labels, query, query_labels)
        n_episodes: Number of episodes to evaluate
    
    Returns:
        mean_accuracy: Mean accuracy across episodes
        ci: 95% confidence interval
        accuracies: List of per-episode accuracies
    """
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for episode_idx, (support, support_labels, query, query_labels) in enumerate(dataloader):
            if episode_idx >= n_episodes:
                break
            
            # Remove batch dimension if present
            if support.dim() == 5:
                support = support.squeeze(0)
                support_labels = support_labels.squeeze(0)
                query = query.squeeze(0)
                query_labels = query_labels.squeeze(0)
            
            # Forward pass
            logits = model(support, support_labels, query)
            
            # Compute predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Compute accuracy for this episode
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    # Compute statistics
    mean_acc, ci = compute_confidence_interval(accuracies)
    
    return mean_acc, ci, accuracies


def evaluate_with_multiple_runs(model, data, labels, n_way, k_shot, n_query, n_episodes=600):
    """
    Evaluate model by creating episodes on-the-fly.
    
    Args:
        model: Few-shot model
        data: All available data
        labels: All available labels
        n_way: Number of classes per episode
        k_shot: Support examples per class
        n_query: Query examples per class
        n_episodes: Number of episodes to evaluate
    
    Returns:
        mean_accuracy, confidence_interval, all_accuracies
    """
    from data_loader import create_episode
    
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for _ in range(n_episodes):
            # Create episode
            support, support_labels, query, query_labels = create_episode(
                data, labels, n_way, k_shot, n_query
            )
            
            # Forward pass
            logits = model(support, support_labels, query)
            predictions = torch.argmax(logits, dim=1)
            
            # Compute accuracy
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    return mean_acc, ci, accuracies


def evaluate_cross_domain(model, source_data, source_labels, target_data, target_labels,
                          n_way, k_shot, n_query, n_episodes=600):
    """
    Evaluate cross-domain few-shot learning.
    
    Train on source domain, test on target domain to measure generalization.
    """
    model.eval()
    accuracies = []
    
    from data_loader import create_episode
    
    with torch.no_grad():
        for _ in range(n_episodes):
            # Create episode from target domain
            support, support_labels, query, query_labels = create_episode(
                target_data, target_labels, n_way, k_shot, n_query
            )
            
            # Evaluate
            logits = model(support, support_labels, query)
            predictions = torch.argmax(logits, dim=1)
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    return mean_acc, ci, accuracies


def compute_confusion_matrix(predictions, labels, n_classes):
    """
    Compute confusion matrix for few-shot predictions.
    
    Args:
        predictions: (n,) - Predicted labels
        labels: (n,) - True labels
        n_classes: Number of classes
    
    Returns:
        confusion_matrix: (n_classes, n_classes) matrix
    """
    confusion = torch.zeros(n_classes, n_classes)
    
    for pred, true in zip(predictions, labels):
        confusion[true, pred] += 1
    
    return confusion


def evaluate_per_class_accuracy(predictions, labels, n_classes):
    """
    Compute per-class accuracy.
    
    Returns:
        per_class_acc: (n_classes,) - Accuracy for each class
    """
    per_class_acc = []
    
    for c in range(n_classes):
        class_mask = (labels == c)
        if class_mask.sum() > 0:
            class_predictions = predictions[class_mask]
            class_labels = labels[class_mask]
            accuracy = compute_accuracy(class_predictions, class_labels)
            per_class_acc.append(accuracy)
        else:
            per_class_acc.append(0.0)
    
    return torch.tensor(per_class_acc)


class FewShotEvaluator:
    """
    Comprehensive evaluator for few-shot learning models.
    """
    def __init__(self, model):
        self.model = model
        self.results = {
            'accuracies': [],
            'losses': [],
            'per_class_accuracies': []
        }
    
    def evaluate_episode(self, support, support_labels, query, query_labels):
        """
        Evaluate on a single episode.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(support, support_labels, query)
            predictions = torch.argmax(logits, dim=1)
            
            accuracy = compute_accuracy(predictions, query_labels)
            loss = torch.nn.functional.cross_entropy(logits, query_labels).item()
            
            n_classes = len(torch.unique(support_labels))
            per_class_acc = evaluate_per_class_accuracy(predictions, query_labels, n_classes)
            
            self.results['accuracies'].append(accuracy)
            self.results['losses'].append(loss)
            self.results['per_class_accuracies'].append(per_class_acc)
        
        return accuracy, loss
    
    def get_summary(self):
        """
        Get summary statistics of evaluation.
        """
        mean_acc, ci_acc = compute_confidence_interval(self.results['accuracies'])
        mean_loss = np.mean(self.results['losses'])
        
        # Average per-class accuracy
        if self.results['per_class_accuracies']:
            avg_per_class = torch.stack(self.results['per_class_accuracies']).mean(dim=0)
        else:
            avg_per_class = None
        
        summary = {
            'mean_accuracy': mean_acc,
            'accuracy_95_ci': ci_acc,
            'mean_loss': mean_loss,
            'per_class_accuracy': avg_per_class,
            'n_episodes': len(self.results['accuracies'])
        }
        
        return summary
    
    def reset(self):
        """
        Reset evaluation results.
        """
        self.results = {
            'accuracies': [],
            'losses': [],
            'per_class_accuracies': []
        }


def print_evaluation_results(mean_acc, ci, n_episodes):
    """
    Pretty print evaluation results.
    """
    print("=" * 50)
    print("Few-Shot Learning Evaluation Results")
    print("=" * 50)
    print(f"Number of episodes: {n_episodes}")
    print(f"Mean accuracy: {mean_acc*100:.2f}%")
    print(f"95% Confidence interval: ±{ci*100:.2f}%")
    print(f"Accuracy range: [{(mean_acc-ci)*100:.2f}%, {(mean_acc+ci)*100:.2f}%]")
    print("=" * 50)


# Example usage
if __name__ == "__main__":
    from prototypical_networks import PrototypicalNetwork, ConvEncoder
    from data_loader import create_episode
    
    # Create dummy data
    n_samples = 500
    n_classes = 20
    data = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Create model
    encoder = ConvEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = PrototypicalNetwork(encoder)
    
    # Evaluate
    print("Evaluating 5-way 1-shot...")
    mean_acc, ci, accuracies = evaluate_with_multiple_runs(
        model, data, labels,
        n_way=5, k_shot=1, n_query=15,
        n_episodes=100
    )
    print_evaluation_results(mean_acc, ci, 100)
    
    # Evaluate 5-way 5-shot
    print("\nEvaluating 5-way 5-shot...")
    mean_acc, ci, accuracies = evaluate_with_multiple_runs(
        model, data, labels,
        n_way=5, k_shot=5, n_query=15,
        n_episodes=100
    )
    print_evaluation_results(mean_acc, ci, 100)
    
    # Use FewShotEvaluator
    evaluator = FewShotEvaluator(model)
    
    for i in range(10):
        support, support_labels, query, query_labels = create_episode(
            data, labels, n_way=5, k_shot=1, n_query=15
        )
        evaluator.evaluate_episode(support, support_labels, query, query_labels)
    
    summary = evaluator.get_summary()
    print("\nEvaluator Summary:")
    print(f"Mean accuracy: {summary['mean_accuracy']*100:.2f}%")
    print(f"95% CI: ±{summary['accuracy_95_ci']*100:.2f}%")
    print(f"Mean loss: {summary['mean_loss']:.4f}")
