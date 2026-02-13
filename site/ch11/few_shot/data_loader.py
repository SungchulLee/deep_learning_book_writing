"""
Episodic Data Loader for Few-Shot Learning

Creates episodes (tasks) for training and evaluation in few-shot learning.
Each episode consists of a support set and query set sampled from N classes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict


class EpisodicDataset(Dataset):
    """
    Dataset that generates episodes for few-shot learning.
    
    Each episode is an N-way K-shot task with a support set and query set.
    """
    def __init__(self, data, labels, n_way, k_shot, n_query, n_episodes):
        """
        Args:
            data: (N, *input_shape) - All available data
            labels: (N,) - Labels for all data
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to generate
        """
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Organize data by class
        self.classes = torch.unique(labels).tolist()
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label.item()].append(idx)
    
    def __len__(self):
        return self.n_episodes
    
    def __getitem__(self, idx):
        """
        Generate one episode.
        
        Returns:
            support_set: (n_way * k_shot, *input_shape)
            support_labels: (n_way * k_shot,)
            query_set: (n_way * n_query, *input_shape)
            query_labels: (n_way * n_query,)
        """
        # Randomly select n_way classes
        episode_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx, class_label in enumerate(episode_classes):
            # Get all indices for this class
            class_indices = self.class_to_indices[class_label]
            
            # Sample k_shot + n_query examples
            selected_indices = np.random.choice(
                class_indices,
                self.k_shot + self.n_query,
                replace=False
            )
            
            # Split into support and query
            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:]
            
            # Add to support set
            support_data.append(self.data[support_indices])
            support_labels.extend([class_idx] * self.k_shot)
            
            # Add to query set
            query_data.append(self.data[query_indices])
            query_labels.extend([class_idx] * self.n_query)
        
        # Concatenate all classes
        support_set = torch.cat(support_data, dim=0)
        support_labels = torch.tensor(support_labels)
        query_set = torch.cat(query_data, dim=0)
        query_labels = torch.tensor(query_labels)
        
        return support_set, support_labels, query_set, query_labels


class MiniImageNetLoader:
    """
    Data loader for mini-ImageNet or similar datasets.
    Organizes data for episodic few-shot learning.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
        # In practice, load actual dataset here
        # For now, we'll create dummy data
    
    def get_dataloader(self, split='train', n_way=5, k_shot=5, n_query=15, n_episodes=100, batch_size=4):
        """
        Create episodic dataloader.
        
        Args:
            split: 'train', 'val', or 'test'
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes
            batch_size: Batch size (number of episodes per batch)
        """
        # Load data (dummy data for example)
        if split == 'train':
            n_samples = 1000
            n_classes = 64
        elif split == 'val':
            n_samples = 300
            n_classes = 16
        else:  # test
            n_samples = 300
            n_classes = 20
        
        # Create dummy data
        data = torch.randn(n_samples, 3, 84, 84)  # Standard mini-ImageNet size
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # Create episodic dataset
        dataset = EpisodicDataset(data, labels, n_way, k_shot, n_query, n_episodes)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for simplicity
        )
        
        return dataloader


class OmniglotLoader:
    """
    Data loader for Omniglot dataset (handwritten characters).
    Standard benchmark for few-shot learning.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
    
    def get_dataloader(self, split='train', n_way=5, k_shot=1, n_query=15, n_episodes=100):
        """
        Create episodic dataloader for Omniglot.
        """
        # Load data (dummy for example)
        if split == 'train':
            n_samples = 1000
            n_classes = 1200  # Background set
        else:
            n_samples = 500
            n_classes = 423  # Evaluation set
        
        # Create dummy data (28x28 grayscale images)
        data = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # Create episodic dataset
        dataset = EpisodicDataset(data, labels, n_way, k_shot, n_query, n_episodes)
        
        return DataLoader(dataset, batch_size=1, shuffle=True)


def create_episode(data, labels, n_way, k_shot, n_query):
    """
    Utility function to create a single episode from data.
    
    Args:
        data: (N, *input_shape) - All data
        labels: (N,) - All labels
        n_way: Number of classes
        k_shot: Support examples per class
        n_query: Query examples per class
    
    Returns:
        support_set, support_labels, query_set, query_labels
    """
    # Get unique classes
    unique_classes = torch.unique(labels)
    
    # Sample n_way classes
    episode_classes = unique_classes[torch.randperm(len(unique_classes))[:n_way]]
    
    support_data = []
    support_labels = []
    query_data = []
    query_labels = []
    
    for class_idx, class_label in enumerate(episode_classes):
        # Get indices for this class
        class_mask = (labels == class_label)
        class_data = data[class_mask]
        
        # Shuffle and split
        perm = torch.randperm(len(class_data))
        support_indices = perm[:k_shot]
        query_indices = perm[k_shot:k_shot + n_query]
        
        # Add to sets
        support_data.append(class_data[support_indices])
        support_labels.extend([class_idx] * k_shot)
        query_data.append(class_data[query_indices])
        query_labels.extend([class_idx] * n_query)
    
    support_set = torch.cat(support_data, dim=0)
    support_labels = torch.tensor(support_labels)
    query_set = torch.cat(query_data, dim=0)
    query_labels = torch.tensor(query_labels)
    
    return support_set, support_labels, query_set, query_labels


# Example usage
if __name__ == "__main__":
    # Example 1: Create episodic dataset
    n_samples = 500
    n_classes = 20
    
    # Dummy data (28x28 grayscale images)
    data = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Create episodic dataset (5-way 1-shot)
    dataset = EpisodicDataset(
        data=data,
        labels=labels,
        n_way=5,
        k_shot=1,
        n_query=15,
        n_episodes=100
    )
    
    # Get one episode
    support_set, support_labels, query_set, query_labels = dataset[0]
    print(f"Support set shape: {support_set.shape}")  # (5, 1, 28, 28)
    print(f"Support labels: {support_labels}")
    print(f"Query set shape: {query_set.shape}")  # (75, 1, 28, 28)
    print(f"Query labels: {query_labels}")
    
    # Example 2: Use dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, batch in enumerate(dataloader):
        support_sets, support_label_sets, query_sets, query_label_sets = batch
        print(f"\nBatch {batch_idx}:")
        print(f"Support sets shape: {support_sets.shape}")  # (4, 5, 1, 28, 28)
        print(f"Query sets shape: {query_sets.shape}")  # (4, 75, 1, 28, 28)
        
        if batch_idx == 0:
            break
    
    # Example 3: Mini-ImageNet loader
    mini_loader = MiniImageNetLoader()
    train_loader = mini_loader.get_dataloader(
        split='train',
        n_way=5,
        k_shot=5,
        n_query=15,
        n_episodes=100,
        batch_size=4
    )
    
    for batch in train_loader:
        support, support_labels, query, query_labels = batch
        print(f"\nMini-ImageNet batch:")
        print(f"Support shape: {support.shape}")
        print(f"Query shape: {query.shape}")
        break
