"""
Balanced DataLoader — Cats vs Dogs
===================================
Demonstrates ``WeightedRandomSampler`` to balance a heavily imbalanced
dataset (100 cats vs 1 000 dogs) so that each training batch contains
a roughly equal mix of both classes.

Usage
-----
    python balanced_dataloader.py
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def create_balanced_data_loader():
    """
    Creates a DataLoader with a WeightedRandomSampler to handle class
    imbalance.  Features are generated synthetically for a dataset with
    100 cats and 1 000 dogs.

    Returns
    -------
    DataLoader
        A data loader configured with weighted random sampling.
    """
    # Synthetic features
    cat_features = torch.randn(100, 10)       # 100 cats
    dog_features = torch.randn(1000, 10)      # 1 000 dogs

    features = torch.cat((cat_features, dog_features), dim=0)   # (1100, 10)
    labels   = torch.cat((torch.zeros(100), torch.ones(1000)))  # (1100,)

    dataset = TensorDataset(features, labels)

    # Inverse-frequency weighting
    class_counts   = torch.tensor(
        [(labels == 0).sum(), (labels == 1).sum()], dtype=torch.float32
    )
    class_weights  = 1.0 / class_counts                         # (1/100, 1/1000)
    sample_weights = torch.tensor(
        [class_weights[int(label)] for label in labels]          # (1100,)
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(dataset, batch_size=100, sampler=sampler)


def main():
    loader = create_balanced_data_loader()

    for batch_idx, (batch_features, batch_labels) in enumerate(loader):
        n_cats = (batch_labels == 0).sum().item()
        n_dogs = (batch_labels == 1).sum().item()
        print(f"Batch {batch_idx + 1}: Cats={n_cats}, Dogs={n_dogs}")


if __name__ == "__main__":
    main()
