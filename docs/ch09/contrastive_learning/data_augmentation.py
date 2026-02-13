"""
Data Augmentation Strategies for Self-Supervised Learning
Includes augmentations used in SimCLR, MoCo, and other contrastive methods.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import ImageFilter, ImageOps
import numpy as np


class GaussianBlur:
    """Gaussian blur augmentation"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """Solarization augmentation"""
    def __init__(self, threshold=128):
        self.threshold = threshold
    
    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)


class SimCLRAugmentation:
    """
    Data augmentation pipeline used in SimCLR
    Creates two correlated views of the same image
    """
    def __init__(self, image_size=224, s=1.0):
        """
        Args:
            image_size: final image size
            s: strength of color distortion
        """
        # Color jitter parameters
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply augmentation twice to create two views"""
        return self.transform(x), self.transform(x)


class MoCoAugmentation:
    """
    Data augmentation pipeline used in MoCo v2
    """
    def __init__(self, image_size=224):
        # Query augmentation (stronger)
        self.query_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Key augmentation (same as query for MoCo v2)
        self.key_transform = self.query_transform
    
    def __call__(self, x):
        """Apply augmentation to create query and key"""
        return self.query_transform(x), self.key_transform(x)


class MoCoV3Augmentation:
    """
    Data augmentation pipeline used in MoCo v3
    Includes additional augmentations
    """
    def __init__(self, image_size=224):
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply two different augmentations"""
        return self.transform1(x), self.transform2(x)


class BYOLAugmentation:
    """
    Data augmentation pipeline used in BYOL
    Uses asymmetric augmentation
    """
    def __init__(self, image_size=224):
        # View 1: stronger augmentation
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # View 2: weaker augmentation
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply asymmetric augmentation"""
        return self.transform1(x), self.transform2(x)


class MAEAugmentation:
    """
    Data augmentation for MAE
    MAE uses minimal augmentation compared to contrastive methods
    """
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply simple augmentation"""
        return self.transform(x)


class MultiCropAugmentation:
    """
    Multi-crop augmentation used in SwAV and DINO
    Creates multiple crops of different sizes
    """
    def __init__(
        self,
        image_size=224,
        n_global_crops=2,
        n_local_crops=6,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.4)
    ):
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        
        # Global crop augmentation
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Local crop augmentation
        local_size = int(image_size * 0.4)  # Smaller size for local crops
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Create global and local crops"""
        crops = []
        
        # Global crops
        for _ in range(self.n_global_crops):
            crops.append(self.global_transform(x))
        
        # Local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(x))
        
        return crops


class TwoCropsTransform:
    """
    Generic wrapper for creating two augmented views
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def get_augmentation(method='simclr', image_size=224):
    """
    Get augmentation pipeline for specified method
    
    Args:
        method: 'simclr', 'moco', 'mocov3', 'byol', 'mae', 'multicrop'
        image_size: image size
    
    Returns:
        augmentation transform
    """
    if method == 'simclr':
        return SimCLRAugmentation(image_size)
    elif method == 'moco':
        return MoCoAugmentation(image_size)
    elif method == 'mocov3':
        return MoCoV3Augmentation(image_size)
    elif method == 'byol':
        return BYOLAugmentation(image_size)
    elif method == 'mae':
        return MAEAugmentation(image_size)
    elif method == 'multicrop':
        return MultiCropAugmentation(image_size)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from PIL import Image
    
    # Create a dummy image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Test SimCLR augmentation
    print("Testing SimCLR augmentation...")
    simclr_aug = SimCLRAugmentation(image_size=224)
    view1, view2 = simclr_aug(img)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    
    # Test MoCo augmentation
    print("\nTesting MoCo augmentation...")
    moco_aug = MoCoAugmentation(image_size=224)
    query, key = moco_aug(img)
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    
    # Test MAE augmentation
    print("\nTesting MAE augmentation...")
    mae_aug = MAEAugmentation(image_size=224)
    aug_img = mae_aug(img)
    print(f"Augmented image shape: {aug_img.shape}")
    
    # Test multi-crop augmentation
    print("\nTesting Multi-crop augmentation...")
    multicrop_aug = MultiCropAugmentation(image_size=224, n_global_crops=2, n_local_crops=4)
    crops = multicrop_aug(img)
    print(f"Number of crops: {len(crops)}")
    print(f"Global crop shapes: {[crops[i].shape for i in range(2)]}")
    print(f"Local crop shapes: {[crops[i].shape for i in range(2, 6)]}")
    
    print("\nAll augmentation tests passed!")
